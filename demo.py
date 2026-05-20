"""Run a trained checkpoint on a .wav and save the denoised target + the other stream."""
import argparse
import time
from pathlib import Path

import torch
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from model import TasNet
from dataset import SR

# target stream is empirical per checkpoint — verify with sample.py
TARGET_STREAM = 1


def save_wav(path: Path, x: torch.Tensor, norm: float) -> None:
    AudioEncoder((x * norm).unsqueeze(0), sample_rate=SR).to_file(str(path))


def main(args):
    # CPU-only: AR-glasses target is on-device CPU, GPU numbers are not representative.
    device = torch.device("cpu")

    noisy = AudioDecoder(str(args.input), sample_rate=SR).get_all_samples().data.mean(dim=0)

    model = TasNet(num_spk=2, causal=True, sr=SR).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {args.checkpoint}  (epoch {ckpt.get('epoch', '?')}, "
          f"val si-sdr {ckpt.get('val_sisdr', float('nan')):+.2f} dB)")

    # prediction with timing
    x = noisy.unsqueeze(0)
    with torch.no_grad():
        model(x)  # warmup
        t0 = time.perf_counter()
        estimates = model(x)[0]  # (2, T)
        dt_ms = (time.perf_counter() - t0) * 1000
    audio_s = noisy.size(0) / SR
    print(f"forward: {dt_ms:.1f} ms on {audio_s:.1f} s of audio  ({dt_ms / audio_s:.1f} ms / s) on cpu")

    # scale all signals so their peak normalizes to 0.99
    peak = max(noisy.abs().max().item(), estimates.abs().max().item())
    norm = 0.99 / peak if peak > 0.99 else 1.0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_wav(args.out_dir / "1_noisy.wav",            noisy,                          norm)
    save_wav(args.out_dir / "2_estimate_target.wav",  estimates[TARGET_STREAM],       norm)
    save_wav(args.out_dir / "3_estimate_other.wav",   estimates[1 - TARGET_STREAM],   norm)
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Noisy input .wav")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    main(p.parse_args())
