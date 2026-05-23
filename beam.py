"""Render multi-mic beamformed scenes for the post-filter training preview.

3D ShoeBox + linear mic array + target + 1-3 babble talkers + env noise.
Per-source rendering keeps SIR/SNR controllable. Beamforming is MVDR in the
STFT domain with an anechoic-geometry steering vector and an oracle noise
covariance (from babble + env signals — in deployment this becomes a
VAD-gated estimate, but the rest of the code stays the same).

Outputs 6 wavs per example so the user can listen to each component pre- and
post-beamform, plus prints SI-SDR(mic0→reverb_target) vs SI-SDR(beam→same)
so the gap (what beamforming buys *before* the post-filter sees anything) is
visible.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import torch
from torchcodec.encoders import AudioEncoder

from dataset import (
    DATASETS, BABBLE_GAIN_RANGE, SR,
    list_speech_files, load_mono, random_segment, rms,
)
from metrics import calc_sdr_torch
from rir import ROOM_DIM_RANGE, WALL_MARGIN

CEILING_RANGE = (2.5, 3.5)        # 3D ceiling height range (meters)
Z_MARGIN = 0.3                    # floor/ceiling margin for source placement (meters)
SOURCE_Z_JITTER = 0.2             # source z ~ mic_height ± this (meters; head-height variation)
SPEED_OF_SOUND = 343.0            # m/s — used for MVDR steering-vector geometric delays
MVDR_N_FFT = 1024                 # 64 ms @ 16 kHz; 75% overlap (hop=256)
MVDR_HOP = 256
MVDR_DIAG_LOAD = 1e-6             # diagonal-loading ε for noise-cov regularization

TARGET_ANGLE_JITTER = 8.0         # target sits at boresight (90°) ± this
TARGET_DIST_RANGE = (1.0, 2.0)
BABBLE_K_RANGE = (1, 3)
BABBLE_DIST_RANGE = (1.0, 3.5)
BABBLE_NEAR_TARGET_DEG = 15.0     # at least one babble forced within ± this of target (co-directional)
BABBLE_FAR_ANGLE_RANGE = (30.0, 150.0)
ENV_DIST_RANGE = (1.5, 4.0)
ENV_ANGLE_RANGE = (30.0, 150.0)


def sample_room(rt60_range):
    """Sample a 3D ShoeBox + valid Sabine materials. Retry on infeasible (room, RT60) combos."""
    while True:
        room_dim = [
            random.uniform(*ROOM_DIM_RANGE),
            random.uniform(*ROOM_DIM_RANGE),
            random.uniform(*CEILING_RANGE),
        ]
        rt60 = random.uniform(*rt60_range)
        try:
            e_abs, max_order = pra.inverse_sabine(rt60, room_dim)
            return room_dim, e_abs, max_order, rt60
        except ValueError:
            continue


def place_array(room_dim, n_mics, spacing, height):
    """Linear array along x-axis at `height`. Returns (mic_positions (3, M), center (cx, cy, ch))."""
    cx = random.uniform(WALL_MARGIN, room_dim[0] - WALL_MARGIN)
    cy = random.uniform(WALL_MARGIN, room_dim[1] - WALL_MARGIN)
    ch = float(height)
    offsets = (np.arange(n_mics) - (n_mics - 1) / 2.0) * spacing
    xs = cx + offsets
    ys = np.full(n_mics, cy)
    zs = np.full(n_mics, ch)
    return np.stack([xs, ys, zs], axis=0), (cx, cy, ch)


def place_source(center, angle_deg, dist, room_dim):
    """Polar (horizontal angle, distance) + z-jitter around mic height -> clamped (x, y, z)."""
    cx, cy, ch = center
    rad = np.deg2rad(angle_deg)
    x = cx + dist * np.cos(rad)
    y = cy + dist * np.sin(rad)
    z = ch + random.uniform(-SOURCE_Z_JITTER, SOURCE_Z_JITTER)
    x = float(np.clip(x, WALL_MARGIN, room_dim[0] - WALL_MARGIN))
    y = float(np.clip(y, WALL_MARGIN, room_dim[1] - WALL_MARGIN))
    z = float(np.clip(z, Z_MARGIN, room_dim[2] - Z_MARGIN))
    return [x, y, z]


def render_to_mics(audio, src_pos, mic_positions, room_dim, e_abs, max_order):
    """Per-source render: one-source pra ShoeBox, return (M, T) mic signals cropped to input length."""
    room = pra.ShoeBox(room_dim, fs=SR, materials=pra.Material(e_abs), max_order=max_order)
    room.add_microphone_array(mic_positions)
    room.add_source(src_pos, signal=audio.numpy())
    room.simulate()
    sig = torch.from_numpy(room.mic_array.signals).float()       # (M, T_render)
    return sig[:, :audio.size(0)]


def compute_mvdr_weights(noise_mic, mic_positions, target_angle_deg,
                         n_fft=MVDR_N_FFT, hop=MVDR_HOP, diag_load=MVDR_DIAG_LOAD):
    """MVDR weights per frequency bin from an oracle noise estimate + anechoic steering vector.

    noise_mic       : (M, T) noise-only signal (in preview: babble_mic + env_mic; in production:
                              VAD-gated noise segments).
    mic_positions   : (3, M) absolute 3D mic coords.
    target_angle_deg: azimuth in degrees; 90° = +y axis = boresight.
    Returns (F, M) complex weights where F = n_fft // 2 + 1.
    """
    M = noise_mic.shape[0]
    window = torch.hann_window(n_fft)
    N = torch.stft(noise_mic, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                   window=window, return_complex=True, center=True)        # (M, F, L)
    L_frames = N.shape[-1]

    # Noise covariance Σ_NN(f) = N(f) N(f)^H / L  for each freq bin
    N_fml = N.permute(1, 0, 2)                                              # (F, M, L)
    sigma = (N_fml @ N_fml.conj().transpose(-2, -1)) / L_frames             # (F, M, M)
    eye = torch.eye(M, dtype=sigma.dtype).unsqueeze(0)
    sigma = sigma + diag_load * eye

    # Steering vector for plane-wave model: d_k(m) = exp(-j 2π f_k τ_m), τ_m = (r_m · ĥ) / c
    rad = np.deg2rad(target_angle_deg)
    direction = torch.tensor([np.cos(rad), np.sin(rad), 0.0], dtype=torch.float32)
    mics = torch.from_numpy(mic_positions).float()
    rel = mics - mics.mean(dim=1, keepdim=True)                             # (3, M)
    tau = (direction @ rel) / SPEED_OF_SOUND                                # (M,)
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / SR)                           # (F,)
    phase = -2 * np.pi * freqs.unsqueeze(1) * tau.unsqueeze(0)              # (F, M)
    d = torch.exp(1j * phase).to(sigma.dtype)                               # (F, M)

    # w_k = Σ⁻¹ d_k / (d_k^H Σ⁻¹ d_k)
    d_col = d.unsqueeze(-1)                                                 # (F, M, 1)
    sigma_inv_d = torch.linalg.solve(sigma, d_col)                          # (F, M, 1)
    denom = (d_col.conj().transpose(-2, -1) @ sigma_inv_d).squeeze(-1).squeeze(-1)  # (F,)
    return sigma_inv_d.squeeze(-1) / denom.unsqueeze(-1)                    # (F, M)


def beamform_mvdr(multi_mic, weights, n_fft=MVDR_N_FFT, hop=MVDR_HOP):
    """Apply per-bin MVDR weights to a (M, T) multi-mic signal. Returns (T,)."""
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
    assert speech_files, f"no .flac files under {speech_roots}"
    assert noise_files, f"no .wav files under {args.wham_root}"

    n_samples = int(SR * args.segment_seconds)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sdr_gaps = []
    for i in range(args.n_examples):
        room_dim, e_abs, max_order, rt60 = sample_room(args.rt60_range)
        mic_positions, center = place_array(room_dim, args.n_mics, args.mic_spacing, args.mic_height)

        # Source geometry
        target_angle = 90.0 + random.uniform(-TARGET_ANGLE_JITTER, TARGET_ANGLE_JITTER)
        target_dist  = random.uniform(*TARGET_DIST_RANGE)

        k = random.randint(*BABBLE_K_RANGE)
        # First babble forced co-directional with target (the hard residual)
        babble_angles = [target_angle + random.uniform(-BABBLE_NEAR_TARGET_DEG, BABBLE_NEAR_TARGET_DEG)]
        babble_dists  = [random.uniform(*BABBLE_DIST_RANGE)]
        for _ in range(k - 1):
            babble_angles.append(random.uniform(*BABBLE_FAR_ANGLE_RANGE))
            babble_dists.append(random.uniform(*BABBLE_DIST_RANGE))

        env_angle = random.uniform(*ENV_ANGLE_RANGE)
        env_dist  = random.uniform(*ENV_DIST_RANGE)

        # Audio
        target_audio = random_segment(load_mono(random.choice(speech_files), SR), n_samples)
        babble_audios = [random_segment(load_mono(random.choice(speech_files), SR), n_samples)
                         for _ in range(k)]
        env_audio = random_segment(load_mono(random.choice(noise_files), SR), n_samples)

        # Per-source mic signals — each source rendered in its own ShoeBox (separable for SIR/SNR)
        target_mic = render_to_mics(
            target_audio, place_source(center, target_angle, target_dist, room_dim),
            mic_positions, room_dim, e_abs, max_order)

        babble_mics_per_src = [
            render_to_mics(b, place_source(center, a, d, room_dim),
                           mic_positions, room_dim, e_abs, max_order)
            for b, a, d in zip(babble_audios, babble_angles, babble_dists)
        ]
        env_mic = render_to_mics(
            env_audio, place_source(center, env_angle, env_dist, room_dim),
            mic_positions, room_dim, e_abs, max_order)

        # Stack + RMS-normalize babble across the K talkers (matches DynamicSpeechDataset)
        babble_gains = [random.uniform(*BABBLE_GAIN_RANGE) for _ in range(k)]
        babble_mic = sum(b * g for b, g in zip(babble_mics_per_src, babble_gains)) / (k ** 0.5)

        # SIR / SNR scaling — derive scale from the across-mic mean, apply uniformly across mics
        sir_db = random.uniform(*args.sir_db_range)
        snr_db = random.uniform(*args.snr_db_range)
        babble_scale = rms(target_mic.mean(0)) / (rms(babble_mic.mean(0)) * 10 ** (sir_db / 20))
        env_scale    = rms(target_mic.mean(0)) / (rms(env_mic.mean(0))    * 10 ** (snr_db / 20))
        babble_mic = babble_mic * babble_scale
        env_mic    = env_mic    * env_scale

        multi_mic = target_mic + babble_mic + env_mic       # (M, T)

        # MVDR weights are computed once per scene from oracle noise (babble + env), then
        # applied to multi_mic and each per-source path (linear operator → consistent decomp).
        mvdr_w = compute_mvdr_weights(babble_mic + env_mic, mic_positions, target_angle)
        beam        = beamform_mvdr(multi_mic,  mvdr_w)
        beam_target = beamform_mvdr(target_mic, mvdr_w)
        beam_babble = beamform_mvdr(babble_mic, mvdr_w)
        beam_env    = beamform_mvdr(env_mic,    mvdr_w)

        # Diagnostics — SI-SDR uses the *reverberant* target as reference (target-only signal at
        # the array), so the metric isolates "did the beamformer suppress interference?" from
        # "is reverb still there?". Vs anechoic target the propagation-delay misalignment and
        # reverb mismatch swamp the gap. Anechoic target stays the eventual training label —
        # this is only the beamformer's sanity check.
        mic0_raw = multi_mic[0]
        sdr_mic0 = calc_sdr_torch(mic0_raw.unsqueeze(0), target_mic[0].unsqueeze(0)).item()
        sdr_beam = calc_sdr_torch(beam.unsqueeze(0),     beam_target.unsqueeze(0)).item()
        gap = sdr_beam - sdr_mic0
        sdr_gaps.append(gap)

        babble_angle_str = ", ".join(f"{a:.1f}°" for a in babble_angles)
        print(
            f"[{i:02d}] RT60={rt60:.2f}s  "
            f"room=({room_dim[0]:.1f}×{room_dim[1]:.1f}×{room_dim[2]:.1f})m  "
            f"target={target_angle:.1f}°  babble=[{babble_angle_str}]  env={env_angle:.1f}°\n"
            f"     SIR={sir_db:+.1f}dB  SNR={snr_db:+.1f}dB  K={k}\n"
            f"     SI-SDR(mic0→reverb_target) = {sdr_mic0:+6.2f} dB\n"
            f"     SI-SDR(beam →reverb_target) = {sdr_beam:+6.2f} dB   gain = {gap:+.2f} dB"
        )

        # Save 6 wavs with shared peak normalization for fair listening
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
                   help=f"Speech dataset(s) for target+babble. Choices: {sorted(DATASETS.keys())}")
    p.add_argument("--wham-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("preview_beam"))
    p.add_argument("--n-examples", type=int, default=5)
    p.add_argument("--n-mics", type=int, default=2)
    p.add_argument("--mic-spacing", type=float, default=0.14, help="meters")
    p.add_argument("--mic-height", type=float, default=1.5, help="z of mic array, meters")
    p.add_argument("--segment-seconds", type=float, default=4.0)
    p.add_argument("--sir-db-range", type=float, nargs=2, default=[-5.0, 10.0])
    p.add_argument("--snr-db-range", type=float, nargs=2, default=[0.0, 15.0])
    p.add_argument("--rt60-range", type=float, nargs=2, default=[0.2, 0.6])
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
