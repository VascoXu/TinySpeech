"""Pre-render a multi-mic RIR bank for the beam + post-filter training pipeline.

Each scene contains two paired rooms — same circular mic array (centered at origin),
different acoustics — matching preview.py's per-scene rendering exactly:

  - FG (foreground) room: 2D polygon, walls in ±[15, 20] m. Holds the target RIR
    (one position at origin) and a grid of babble RIRs at (angle, radius).
  - BG (background) room: separate larger 2D polygon, walls in ±[20, 40] m.
    Holds a grid of env-noise RIRs at (angle, radius).

At training time, SpatialAudioDataset samples a scene, convolves the dry target
with the target_rir, picks random grid points for K babble talkers + 1 env source,
peak-normalizes each, sums per-mic, MVDR-beamforms, and yields (beamformed, target).

Bank format (one .pt file, list of N_scenes dicts):
    fg_corners:     (2, 4)    # FG-room polygon corners
    fg_absorption:  float
    bg_corners:     (2, 4)    # BG-room polygon corners
    bg_absorption:  float
    mic_radius:     float
    mic_positions:  (2, M)    # 2D, circular, centered at origin
    target_rir:     (M, rir_len)              # FG, source at origin
    babble_angles:  (N_a,)    degrees
    babble_radii:   (N_r,)    meters
    babble_rirs:    (N_a, N_r, M, rir_len)    # FG
    bg_angles:      (N_a_bg,) degrees
    bg_radii:       (N_r_bg,) meters
    bg_rirs:        (N_a_bg, N_r_bg, M, rir_len)  # BG
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import torch
from tqdm import tqdm

from beamforming import (
    BABBLE_RADIUS_RANGE, BG_ABSORPTION_RANGE, BG_RADIUS_RANGE,
    BG_WALL_HALF_MAX, BG_WALL_HALF_MIN,
    FG_WALL_HALF_MAX, FG_WALL_HALF_MIN,
    MAX_ORDER, MIC_RADIUS_RANGE, N_MICS_DEFAULT,
    place_circular_array, sample_room,
)
from dataset import SR


def _polar_grid(angles_deg, radii):
    """Cartesian product of (angle, radius) -> list of (x, y) tuples."""
    out = []
    for a in angles_deg:
        th = float(a) * np.pi / 180.0
        for r in radii:
            out.append((float(r) * np.cos(th), float(r) * np.sin(th)))
    return out


def render_grid_rirs(corners, absorption, mic_positions, source_xy, rir_len):
    """All sources in one room — single image-source pass, then read room.rir[m][i].

    source_xy: list of (x, y). Returns (N, M, rir_len) padded/truncated to rir_len.
    """
    room = pra.Room.from_corners(corners, fs=SR, max_order=MAX_ORDER,
                                 materials=pra.Material(absorption))
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))
    for pos in source_xy:
        room.add_source(list(pos))
    room.compute_rir()

    n_pos, m_mics = len(source_xy), mic_positions.shape[1]
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

    babble_angles = torch.linspace(0.0, 360.0 - 360.0 / args.n_fg_angles, args.n_fg_angles)
    babble_radii  = torch.linspace(BABBLE_RADIUS_RANGE[0], BABBLE_RADIUS_RANGE[1], args.n_fg_radii)
    bg_angles     = torch.linspace(0.0, 360.0 - 360.0 / args.n_bg_angles, args.n_bg_angles)
    bg_radii      = torch.linspace(BG_RADIUS_RANGE[0], BG_RADIUS_RANGE[1], args.n_bg_radii)

    n_a, n_r        = len(babble_angles), len(babble_radii)
    n_a_bg, n_r_bg  = len(bg_angles), len(bg_radii)
    rir_len = int(args.rir_seconds * SR)

    floats_per_scene = (1 + n_a * n_r + n_a_bg * n_r_bg) * args.n_mics * rir_len
    bytes_per_scene = floats_per_scene * 4

    print(f"FG grid: {n_a} angles × {n_r} radii = {n_a*n_r} babble positions  "
          f"(+ 1 target @ origin) in room walls ±[{FG_WALL_HALF_MIN},{FG_WALL_HALF_MAX}] m")
    print(f"BG grid: {n_a_bg} angles × {n_r_bg} radii = {n_a_bg*n_r_bg} env positions  "
          f"in room walls ±[{BG_WALL_HALF_MIN},{BG_WALL_HALF_MAX}] m")
    print(f"mics: {args.n_mics} circular @ radius ∈ {MIC_RADIUS_RANGE} m  "
          f"RIR len: {rir_len} samples ({args.rir_seconds}s)")
    print(f"~{bytes_per_scene/1e6:.1f} MB per scene -> "
          f"~{bytes_per_scene*args.n_scenes/1e9:.2f} GB total")

    bank = []
    for _ in tqdm(range(args.n_scenes), desc="scenes"):
        mic_radius = random.uniform(*MIC_RADIUS_RANGE)
        mic_positions = place_circular_array(args.n_mics, mic_radius)

        # ---- FG room: target at origin + babble grid ----
        fg_corners, fg_absorption = sample_room(FG_WALL_HALF_MIN, FG_WALL_HALF_MAX)
        fg_xy = [(0.0, 0.0)] + _polar_grid(babble_angles, babble_radii)
        fg_rirs = render_grid_rirs(fg_corners, fg_absorption, mic_positions, fg_xy, rir_len)
        target_rir  = fg_rirs[0]                                          # (M, rir_len)
        babble_rirs = fg_rirs[1:].reshape(n_a, n_r, args.n_mics, rir_len)

        # ---- BG room: env grid ----
        bg_corners, bg_absorption = sample_room(BG_WALL_HALF_MIN, BG_WALL_HALF_MAX, BG_ABSORPTION_RANGE)
        bg_xy = _polar_grid(bg_angles, bg_radii)
        bg_rirs = render_grid_rirs(bg_corners, bg_absorption, mic_positions,
                                   bg_xy, rir_len).reshape(n_a_bg, n_r_bg, args.n_mics, rir_len)

        bank.append({
            "fg_corners":     torch.from_numpy(fg_corners).float(),
            "fg_absorption":  fg_absorption,
            "bg_corners":     torch.from_numpy(bg_corners).float(),
            "bg_absorption":  bg_absorption,
            "mic_radius":     mic_radius,
            "mic_positions":  torch.from_numpy(mic_positions).float(),
            "target_rir":     target_rir,
            "babble_angles":  babble_angles.clone(),
            "babble_radii":   babble_radii.clone(),
            "babble_rirs":    babble_rirs,
            "bg_angles":      bg_angles.clone(),
            "bg_radii":       bg_radii.clone(),
            "bg_rirs":        bg_rirs,
        })

    args.out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, args.out_pt)
    size_mb = args.out_pt.stat().st_size / (1024 ** 2)
    print(f"saved {len(bank)} scenes -> {args.out_pt}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-scenes", type=int, default=20)
    p.add_argument("--out-pt", type=Path, default=Path("datasets/rir_bank_beam.pt"))
    p.add_argument("--n-mics", type=int, default=N_MICS_DEFAULT)
    p.add_argument("--n-fg-angles", type=int, default=16)
    p.add_argument("--n-fg-radii",  type=int, default=5)
    p.add_argument("--n-bg-angles", type=int, default=8)
    p.add_argument("--n-bg-radii",  type=int, default=3)
    p.add_argument("--rir-seconds", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
