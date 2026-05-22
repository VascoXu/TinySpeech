"""Pre-render a bank of random-room RIRs and convolve dry audio through them at training time."""
import argparse
import random
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import torch
from tqdm import tqdm


# Room sampling ranges — 2D ShoeBox, mono mic. RT60 covers small rooms to medium reverb halls.
ROOM_DIM_RANGE = (5.0, 10.0)   # meters per side
RT60_RANGE = (0.15, 0.6)       # seconds
WALL_MARGIN = 0.5              # min distance from any wall, meters


def _render_one(sr: int, rir_seconds: float) -> torch.Tensor:
    # Some (room, rt60) combos are physically infeasible (short room can't sustain long reverb);
    # pra.inverse_sabine raises in that case — just resample until we get a valid combo.
    while True:
        room_dim = [random.uniform(*ROOM_DIM_RANGE), random.uniform(*ROOM_DIM_RANGE)]
        rt60 = random.uniform(*RT60_RANGE)
        try:
            e_abs, max_order = pra.inverse_sabine(rt60, room_dim)
            break
        except ValueError:
            continue
    room = pra.ShoeBox(room_dim, fs=sr, materials=pra.Material(e_abs), max_order=max_order)

    src = [random.uniform(WALL_MARGIN, d - WALL_MARGIN) for d in room_dim]
    mic = [random.uniform(WALL_MARGIN, d - WALL_MARGIN) for d in room_dim]
    rir_len = int(sr * rir_seconds)
    # Drive a unit impulse through the room to recover the impulse response at the mic.
    impulse = np.concatenate([[1.0], np.zeros(rir_len - 1)])
    room.add_source(src, signal=impulse)
    room.add_microphone(mic)
    room.image_source_model()
    room.simulate()

    out = torch.zeros(rir_len)
    sig = torch.from_numpy(room.mic_array.signals[0]).float()
    n = min(rir_len, sig.size(0))
    out[:n] = sig[:n]
    return out


def render_rir_bank(n_rirs: int, sr: int = 16000, rir_seconds: float = 1.0) -> torch.Tensor:
    return torch.stack([_render_one(sr, rir_seconds)
                        for _ in tqdm(range(n_rirs), desc="rendering RIRs")])


def fft_convolve(signal: torch.Tensor, rir: torch.Tensor, n_samples: int) -> torch.Tensor:
    # FFT-based 1D conv; returns first n_samples (direct path aligned with original onset).
    n_fft = 1 << (signal.size(0) + rir.size(0) - 2).bit_length()
    S = torch.fft.rfft(signal, n=n_fft)
    R = torch.fft.rfft(rir, n=n_fft)
    return torch.fft.irfft(S * R, n=n_fft)[:n_samples]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-rirs", type=int, default=2000)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--rir-seconds", type=float, default=1.0)
    p.add_argument("--out-pt", type=Path, default=Path("datasets/rir_bank.pt"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    random.seed(args.seed)
    bank = render_rir_bank(args.n_rirs, args.sr, args.rir_seconds)
    args.out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, args.out_pt)
    print(f"saved {tuple(bank.shape)} -> {args.out_pt}")
