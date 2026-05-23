"""Side-by-side log-magnitude spectrograms: noisy input, clean target, model estimate."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dataset import ProcessedSpeechDataset
from metrics import calc_sdr_torch
from model import TasNet

SR = 16000
TARGET_STREAM = 1   # empirical per checkpoint — matches eval.py / demo.py
N_FFT = 1024
HOP = 256
DB_RANGE = 80   # displayed dynamic range below the panel peak


def stft_db(x: torch.Tensor) -> torch.Tensor:
    """(T,) waveform -> (F, frames) log-magnitude spectrogram in dB."""
    spec = torch.stft(x, n_fft=N_FFT, hop_length=HOP,
                      window=torch.hann_window(N_FFT, device=x.device),
                      return_complex=True)
    return 20 * torch.log10(spec.abs().clamp_min(1e-7))


def main(args):
    device = torch.device(args.device)
    model = TasNet(num_spk=2, causal=True, sr=SR).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])

    ds = ProcessedSpeechDataset(args.test_pt)
    noisy, sources = ds[args.index]
    target = sources[0]

    with torch.no_grad():
        ests = model(noisy.unsqueeze(0).to(device))[0]   # (2, T)
        est = ests[TARGET_STREAM].cpu()

    sdr_in  = calc_sdr_torch(noisy.unsqueeze(0), target.unsqueeze(0)).item()
    sdr_out = calc_sdr_torch(est.unsqueeze(0),   target.unsqueeze(0)).item()

    panels = {
        f"noisy (SI-SDR {sdr_in:+.2f} dB)": stft_db(noisy),
        "clean target":                      stft_db(target),
        f"estimate (SI-SDR {sdr_out:+.2f} dB)": stft_db(est),
    }
    vmax = max(s.max() for s in panels.values()).item()
    vmin = vmax - DB_RANGE

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, spec) in zip(axes, panels.items()):
        ax.imshow(spec.numpy(), aspect="auto", origin="lower",
                  cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved -> {args.out}  (index {args.index}, SI-SDRi {sdr_out - sdr_in:+.2f} dB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--test-pt", type=Path, required=True)
    p.add_argument("--index", type=int, default=0,
                   help="Sample index in the test set")
    p.add_argument("--out", type=Path, default=Path("data/spectrogram.png"))
    p.add_argument("--device", type=str, default="cuda:0")
    main(p.parse_args())
