"""Render multi-mic beamformed scenes — ClearBuds-style reproduction.

Matches the ClearBuds (clearbuds_waveform/generate_dataset.py) data recipe:
  - 2D polygon room, walls in ±[15, 20] m  (i.e. 30-40 m floor)
  - 4-mic circular array, radius 5-12.5 cm, centered at origin
  - Target voice at array center (0, 0)
  - K=3 babble talkers at uniform random angle, radius 1-5 m
  - Background noise rendered in a SEPARATE larger room (walls ±[20, 40] m)
    at radius 10-20 m — keeps reverb diffuse, matches their pipeline
  - Per-source peak normalization: foreground ∈ [0.15, 0.4], background ∈ [0.2, 0.5]
  - Absorption ∈ [0.1, 0.99] uniform, max_order=10 (fixed)
  - 3-second mixtures

The MVDR+post-filter pipeline (our AR-glasses architecture) sits on top:
  MVDR with all-ones steering (target at array center → omnidirectional, all
  mics see target with zero delay) + oracle noise covariance.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import torch
from torchcodec.encoders import AudioEncoder

from dataset import DATASETS, SR, list_speech_files, load_mono, random_segment
from metrics import calc_sdr_torch

# ---- Room geometry (ClearBuds-style 2D polygon) ----
FG_WALL_HALF_MIN = 15.0           # FG room walls in ±[15, 20] m
FG_WALL_HALF_MAX = 20.0
BG_WALL_HALF_MIN = 20.0           # BG room walls in ±[20, 40] m  (separate larger room)
BG_WALL_HALF_MAX = 40.0
FG_ABSORPTION_RANGE = (0.1, 0.99)
BG_ABSORPTION_RANGE = (0.5, 0.99) # more absorptive → diffuse-sounding distant noise (ClearBuds)
MAX_ORDER = 10

# ---- Mic array ----
MIC_RADIUS_RANGE = (0.05, 0.125)  # circular array radius (m)
N_MICS_DEFAULT = 4
MIC_PHI0 = 0.0                    # starting angle of first mic

# ---- Sources ----
BABBLE_K_RANGE = (1, 3)           # random number of babble talkers per example
BABBLE_RADIUS_RANGE = (1.0, 5.0)
BG_RADIUS_RANGE = (10.0, 20.0)

# ---- Volumes (peak normalization, ClearBuds-style) ----
FG_VOL_RANGE = (0.15, 0.4)
BG_VOL_RANGE = (0.2, 0.5)

# ---- MVDR ----
MVDR_N_FFT = 1024
MVDR_HOP = 256
MVDR_DIAG_LOAD = 1e-3
MVDR_DIAG_FLOOR = 1e-8


def sample_room(half_min, half_max, absorption_range=FG_ABSORPTION_RANGE):
    """ClearBuds-style: 4-corner 2D polygon with walls in ±[half_min, half_max]."""
    lw = -random.uniform(half_min, half_max)
    rw =  random.uniform(half_min, half_max)
    bw = -random.uniform(half_min, half_max)
    tw =  random.uniform(half_min, half_max)
    corners = np.array([[lw, bw], [lw, tw], [rw, tw], [rw, bw]]).T   # (2, 4)
    absorption = random.uniform(*absorption_range)
    return corners, absorption


def place_circular_array(n_mics, mic_radius):
    """ClearBuds-style 2D circular array centered at origin. Returns (2, n_mics)."""
    return pra.circular_2D_array(center=[0.0, 0.0], M=n_mics, phi0=MIC_PHI0, radius=mic_radius)


def render_to_mics(audio, src_pos, mic_positions, corners, absorption, n_samples):
    """Render a single source to all mics in a 2D polygon room. Returns (M, T_render) cropped."""
    room = pra.Room.from_corners(corners, fs=SR, max_order=MAX_ORDER,
                                 materials=pra.Material(absorption))
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))
    room.add_source(list(src_pos), signal=audio.numpy())
    room.image_source_model()
    room.simulate()
    sig = torch.from_numpy(room.mic_array.signals).float()
    return sig[:, :n_samples]


def peak_normalize(sig, target_peak):
    """Scale signal so its abs().max() == target_peak. Pass-through if signal is near-zero."""
    peak = sig.abs().max()
    if peak < 1e-9:
        return sig
    return sig * (target_peak / peak)


def compute_mvdr_weights(noise_mic, n_fft=MVDR_N_FFT, hop=MVDR_HOP,
                         diag_load=MVDR_DIAG_LOAD, diag_floor=MVDR_DIAG_FLOOR):
    """MVDR weights with all-ones steering (target at array center → omnidirectional).

    noise_mic: (M, T) noise-only signal (oracle = babble + env). Returns (F, M) complex weights.
    """
    M = noise_mic.shape[0]
    window = torch.hann_window(n_fft)
    N = torch.stft(noise_mic, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                   window=window, return_complex=True, center=True)        # (M, F, L)
    L_frames = N.shape[-1]
    N_fml = N.permute(1, 0, 2)                                              # (F, M, L)
    sigma = (N_fml @ N_fml.conj().transpose(-2, -1)) / L_frames             # (F, M, M)

    # Scale-adaptive diagonal load (low-freq bins are near-rank-1 → singular at float32)
    trace = sigma.diagonal(dim1=-2, dim2=-1).real.mean(-1)                  # (F,)
    reg = (diag_load * trace).clamp_min(diag_floor)                         # (F,)
    eye = torch.eye(M, dtype=sigma.dtype).unsqueeze(0)
    sigma = sigma + reg[:, None, None] * eye

    # Target at array center → zero-delay across all mics → steering = all-ones
    F_bins = N.shape[1]
    d = torch.ones(F_bins, M, dtype=sigma.dtype)                            # (F, M)
    d_col = d.unsqueeze(-1)
    sigma_inv_d = torch.linalg.solve(sigma, d_col)
    denom = (d_col.conj().transpose(-2, -1) @ sigma_inv_d).squeeze(-1).squeeze(-1)
    return sigma_inv_d.squeeze(-1) / denom.unsqueeze(-1)                    # (F, M)


def beamform_mvdr(multi_mic, weights, n_fft=MVDR_N_FFT, hop=MVDR_HOP):
    """Apply per-bin MVDR weights to a (M, T) signal. Returns (T,)."""
    length = multi_mic.shape[1]
    window = torch.hann_window(n_fft)
    X = torch.stft(multi_mic, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                   window=window, return_complex=True, center=True)        # (M, F, L)
    X_fml = X.permute(1, 0, 2)                                              # (F, M, L)
    Y = (weights.conj().unsqueeze(-1) * X_fml).sum(dim=1)                   # (F, L)
    return torch.istft(Y, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                       window=window, center=True, length=length)


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
        babble_mic = sum(babble_mics_per_src)   # plain sum (ClearBuds doesn't divide by sqrt(K))

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
    p.add_argument("--segment-seconds", type=float, default=3.0)   # ClearBuds default
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
