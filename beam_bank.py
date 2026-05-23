"""Pre-render a multi-mic RIR bank for the beam + post-filter training pipeline.

Each scene = random 3D ShoeBox + random mic-array placement + RIRs from a grid
of (angle, distance) source positions to each mic. At training time the
BeamformedSpeechDataset (future) samples a scene, picks target/babble/env
positions by (angle, distance), looks up the per-mic RIRs, fft-convolves
audio with each, scales SIR/SNR, sums per-mic, then beamforms.

Bank format (one .pt file, list of N_scenes dicts):
    room_dim:      [Lx, Ly, Lz]              # meters
    rt60:          float                     # seconds
    array_center:  [cx, cy, ch]              # mic-array centroid
    mic_positions: (3, M) tensor             # absolute 3D coords
    angles:        (N_a,) tensor (degrees)   # desired grid (90 = boresight)
    distances:     (N_d,) tensor (meters)    # desired grid
    source_xyz:    (N_a, N_d, 3) tensor      # achieved (clamped) positions
    rirs:          (N_a, N_d, M, rir_len)    # per-grid-point per-mic impulse response
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import torch
from tqdm import tqdm

from beam import place_array, place_source, sample_room
from dataset import SR


def render_scene_rirs(room_dim, e_abs, max_order, mic_positions, source_xyz_flat, rir_len):
    """One ShoeBox, all sources at once. source_xyz_flat: (N, 3). Returns (N, M, rir_len)."""
    n_pos, m_mics = source_xyz_flat.shape[0], mic_positions.shape[1]
    room = pra.ShoeBox(room_dim, fs=SR, materials=pra.Material(e_abs), max_order=max_order)
    room.add_microphone_array(mic_positions)
    for pos in source_xyz_flat:
        room.add_source(pos.tolist())
    room.compute_rir()

    rirs = torch.zeros(n_pos, m_mics, rir_len)
    for i in range(n_pos):
        for m in range(m_mics):
            r = torch.from_numpy(room.rir[m][i]).float()
            n = min(rir_len, r.size(0))
            rirs[i, m, :n] = r[:n]
    return rirs


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    angles = torch.arange(args.angle_start, args.angle_stop + 1e-6, args.angle_step)
    distances = torch.tensor(args.distances)
    rir_len = int(args.rir_seconds * SR)
    n_a, n_d = len(angles), len(distances)
    bytes_per_scene = n_a * n_d * args.n_mics * rir_len * 4

    print(f"per-scene grid: {n_a} angles × {n_d} distances = {n_a * n_d} positions")
    print(f"mics: {args.n_mics} @ {args.mic_spacing}m, RIR len: {rir_len} samples ({args.rir_seconds}s)")
    print(f"~{bytes_per_scene / 1e6:.1f} MB per scene -> ~{bytes_per_scene * args.n_scenes / 1e9:.2f} GB total")

    bank = []
    for _ in tqdm(range(args.n_scenes), desc="scenes"):
        room_dim, e_abs, max_order, rt60 = sample_room(args.rt60_range)
        mic_positions, array_center = place_array(room_dim, args.n_mics,
                                                  args.mic_spacing, args.mic_height)

        # Per-grid-point source position: polar -> clamped xyz (z-jitter per grid point)
        source_xyz = torch.zeros(n_a, n_d, 3)
        for ai in range(n_a):
            for di in range(n_d):
                source_xyz[ai, di] = torch.tensor(
                    place_source(array_center, float(angles[ai]),
                                 float(distances[di]), room_dim)
                )

        rirs = render_scene_rirs(
            room_dim, e_abs, max_order, mic_positions,
            source_xyz.reshape(-1, 3), rir_len
        ).reshape(n_a, n_d, args.n_mics, rir_len)

        bank.append({
            "room_dim":      room_dim,
            "rt60":          rt60,
            "array_center":  list(array_center),
            "mic_positions": torch.from_numpy(mic_positions).float(),
            "angles":        angles.clone(),
            "distances":     distances.clone(),
            "source_xyz":    source_xyz,
            "rirs":          rirs,
        })

    args.out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, args.out_pt)
    size_mb = args.out_pt.stat().st_size / (1024 ** 2)
    print(f"saved {len(bank)} scenes -> {args.out_pt}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-scenes", type=int, default=20)
    p.add_argument("--out-pt", type=Path, default=Path("datasets/rir_bank_beam.pt"))
    p.add_argument("--n-mics", type=int, default=2)
    p.add_argument("--mic-spacing", type=float, default=0.14)
    p.add_argument("--mic-height", type=float, default=1.5)
    p.add_argument("--rt60-range", type=float, nargs=2, default=[0.2, 0.6])
    p.add_argument("--angle-start", type=float, default=30.0)
    p.add_argument("--angle-stop",  type=float, default=150.0)
    p.add_argument("--angle-step",  type=float, default=5.0)
    p.add_argument("--distances",   type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.5])
    p.add_argument("--rir-seconds", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
