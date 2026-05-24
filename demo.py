"""Run a trained post-filter checkpoint on a .wav and save the denoised output."""
import argparse
import time
from pathlib import Path

import torch
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from model import TasNet
from dataset import SR


def save_wav(path: Path, x: torch.Tensor, norm: float) -> None:
    AudioEncoder((x * norm).unsqueeze(0), sample_rate=SR).to_file(str(path))


def main(args):
    device = torch.device("cpu")

    samples = AudioDecoder(str(args.input), sample_rate=SR).get_all_samples().data  # (C, T)
    n_channels = samples.shape[0]
    duration = samples.shape[1] / SR
    noisy = samples[0] if n_channels == 1 else samples.mean(dim=0)

    model = TasNet(causal=True, sr=SR).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {args.checkpoint}  (epoch {ckpt.get('epoch', '?')}, "
          f"val si-sdr {ckpt.get('val_sisdr', float('nan')):+.2f} dB)")

    # prediction with timing
    x = noisy.unsqueeze(0)
    with torch.no_grad():
        model(x)  # warmup
        t0 = time.perf_counter()
        estimate = model(x)[0, 0]  # (T,)
        dt_ms = (time.perf_counter() - t0) * 1000
    print(f"forward: {dt_ms:.1f} ms on {duration:.1f} s of audio  ({dt_ms / duration:.1f} ms / s) on cpu")

    # scale both signals so their peak normalizes to 0.99
    peak = max(noisy.abs().max().item(), estimate.abs().max().item())
    norm = 0.99 / peak if peak > 0.99 else 1.0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_wav(args.out_dir / "1_noisy.wav",    noisy,    norm)
    save_wav(args.out_dir / "2_estimate.wav", estimate, norm)
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Noisy input .wav")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    main(p.parse_args())
