"""Render and save a few ClearBuds-style multi-mic scenes for listening.

Per example: a random 2D polygon FG room with 4 mics at the origin, target voice at the
origin, K~U[1,3] babble talkers in the FG room, and one env-noise source in a separate
larger BG room. MVDR-beamforms with all-ones steering (target at array center) + oracle
noise covariance, then saves the raw mic, the beamformed mix, and the per-component
beamformed signals so you can hear what each stage contributes.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torchcodec.encoders import AudioEncoder

from beamforming import (
    BABBLE_K_RANGE, BABBLE_RADIUS_RANGE, BG_ABSORPTION_RANGE, BG_RADIUS_RANGE,
    BG_VOL_RANGE, BG_WALL_HALF_MAX, BG_WALL_HALF_MIN,
    FG_VOL_RANGE, FG_WALL_HALF_MAX, FG_WALL_HALF_MIN,
    MIC_RADIUS_RANGE, N_MICS_DEFAULT,
    beamform_mvdr, compute_mvdr_weights, peak_normalize,
    place_circular_array, render_to_mics, sample_room,
)
from dataset import DATASETS, SR, list_speech_files, load_mono, random_segment
from metrics import calc_sdr_torch


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    speech_roots = [DATASETS[d] for d in args.dataset]
    print(f"dataset: {', '.join(args.dataset)}")
    speech_files = list_speech_files(speech_roots)
    noise_files = sorted(Path(args.wham_root).rglob("*.wav"))
    assert speech_files, f"no .flac under {speech_roots}"
    assert noise_files, f"no .wav under {args.wham_root}"

    n_samples = int(SR * args.segment_seconds)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sdr_gaps = []
    for i in range(args.n_examples):
        # Foreground room + array
        fg_corners, fg_absorption = sample_room(FG_WALL_HALF_MIN, FG_WALL_HALF_MAX)
        mic_radius = random.uniform(*MIC_RADIUS_RANGE)
        mic_positions = place_circular_array(args.n_mics, mic_radius)

        # Target at origin (array center)
        target_audio = random_segment(load_mono(random.choice(speech_files), SR), n_samples)
        target_mic = render_to_mics(target_audio, [0.0, 0.0],
                                    mic_positions, fg_corners, fg_absorption, n_samples)
        target_mic = peak_normalize(target_mic, random.uniform(*FG_VOL_RANGE))

        # K babble talkers at random angles in the same room
        babble_mics_per_src = []
        babble_angles = []
        k = random.randint(*BABBLE_K_RANGE)
        for _ in range(k):
            audio = random_segment(load_mono(random.choice(speech_files), SR), n_samples)
            r = random.uniform(*BABBLE_RADIUS_RANGE)
            theta = random.uniform(0, 2 * np.pi)
            babble_angles.append(np.degrees(theta))
            sig = render_to_mics(audio, [r * np.cos(theta), r * np.sin(theta)],
                                 mic_positions, fg_corners, fg_absorption, n_samples)
            babble_mics_per_src.append(peak_normalize(sig, random.uniform(*FG_VOL_RANGE)))
        babble_mic = sum(babble_mics_per_src)

        # Background noise — separate larger room, distant source
        bg_corners, bg_absorption = sample_room(BG_WALL_HALF_MIN, BG_WALL_HALF_MAX, BG_ABSORPTION_RANGE)
        bg_audio = random_segment(load_mono(random.choice(noise_files), SR), n_samples)
        bg_r = random.uniform(*BG_RADIUS_RANGE)
        bg_theta = random.uniform(0, 2 * np.pi)
        env_mic = render_to_mics(bg_audio, [bg_r * np.cos(bg_theta), bg_r * np.sin(bg_theta)],
                                 mic_positions, bg_corners, bg_absorption, n_samples)
        env_mic = peak_normalize(env_mic, random.uniform(*BG_VOL_RANGE))

        multi_mic = target_mic + babble_mic + env_mic       # (M, T)

        # MVDR with all-ones steering, oracle noise = babble + env
        mvdr_w = compute_mvdr_weights(babble_mic + env_mic)
        beam        = beamform_mvdr(multi_mic, mvdr_w)
        beam_target = beamform_mvdr(target_mic, mvdr_w)
        beam_babble = beamform_mvdr(babble_mic, mvdr_w)
        beam_env    = beamform_mvdr(env_mic,    mvdr_w)

        # Diagnostics — SI-SDR against the reverberant target as each signal sees it
        mic0_raw = multi_mic[0]
        sdr_mic0 = calc_sdr_torch(mic0_raw.unsqueeze(0), target_mic[0].unsqueeze(0)).item()
        sdr_beam = calc_sdr_torch(beam.unsqueeze(0),     beam_target.unsqueeze(0)).item()
        gap = sdr_beam - sdr_mic0
        sdr_gaps.append(gap)

        print(
            f"[{i:02d}] mics={args.n_mics}@{mic_radius*100:.1f}cm  absorption={fg_absorption:.2f}  "
            f"babble_θ=[{', '.join(f'{a:.0f}°' for a in babble_angles)}]  bg_dist={bg_r:.1f}m\n"
            f"     SI-SDR(mic0→reverb_target) = {sdr_mic0:+6.2f} dB\n"
            f"     SI-SDR(beam →reverb_target) = {sdr_beam:+6.2f} dB   gain = {gap:+.2f} dB"
        )

        components = {
            "1_target_anechoic":   target_audio,
            "2_mic0_raw":          mic0_raw,
            "3_beamformed":        beam,
            "4_beamformed_target": beam_target,
            "5_beamformed_babble": beam_babble,
            "6_beamformed_env":    beam_env,
        }
        peak = max(x.abs().max().item() for x in components.values())
        norm = 0.99 / peak if peak > 0.99 else 1.0
        ex_dir = args.out_dir / f"example_{i:02d}"
        ex_dir.mkdir(exist_ok=True)
        for name, x in components.items():
            AudioEncoder((x * norm).unsqueeze(0), sample_rate=SR).to_file(
                str(ex_dir / f"{name}.wav"))

    if sdr_gaps:
        print(f"\nmean SI-SDR gain (beam − mic0): {sum(sdr_gaps)/len(sdr_gaps):+.2f} dB "
              f"over {len(sdr_gaps)} examples")
        print(f"saved -> {args.out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", nargs="+", choices=sorted(DATASETS.keys()), required=True,
                   metavar="NAME",
                   help=f"Speech dataset(s). Choices: {sorted(DATASETS.keys())}")
    p.add_argument("--wham-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("preview_beam"))
    p.add_argument("--n-examples", type=int, default=5)
    p.add_argument("--n-mics", type=int, default=N_MICS_DEFAULT)
    p.add_argument("--segment-seconds", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
