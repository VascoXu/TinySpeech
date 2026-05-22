"""Draw random babble mixtures from DynamicSpeechDataset, run the model, save (noisy, clean, estimate)."""
import argparse
import random
from pathlib import Path

import torch
from torchcodec.encoders import AudioEncoder

from model import TasNet
from dataset import DynamicSpeechDataset, SR
from metrics import calc_sdr_torch


def pick_target_stream(estimates: torch.Tensor, target: torch.Tensor):
    """Oracle stream selection: pick whichever output channel best matches target.

    estimates: (C, T), target: (T,). Returns (best_idx, sdrs_list).
    """
    sdrs = [calc_sdr_torch(estimates[i:i+1], target.unsqueeze(0)).item()
            for i in range(estimates.size(0))]
    best = max(range(len(sdrs)), key=sdrs.__getitem__)
    return best, sdrs


def main(args):
    random.seed(args.seed)

    ds = DynamicSpeechDataset(
        speech_root=args.speech_root,
        wham_root=args.wham_root,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TasNet(num_spk=2, causal=True, sr=SR).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.checkpoint}  (epoch {ckpt.get('epoch', '?')}, "
          f"val si-sdr {ckpt.get('val_sisdr', float('nan')):+.2f} dB)")

    for i in range(args.n_examples):
        noisy, sources = ds[0]  # idx is ignored; fresh random mix each call
        target, interference = sources[0], sources[1]

        with torch.no_grad():
            estimates = model(noisy.unsqueeze(0).to(device))[0].cpu()  # (2, T)

        best, sdrs = pick_target_stream(estimates, target)
        other = 1 - best
        estimate = estimates[best]
        out_sdr = sdrs[best]

        in_sdr = calc_sdr_torch(noisy.unsqueeze(0), target.unsqueeze(0)).item()

        peak = max(t.abs().max().item() for t in [noisy, target, interference, estimate, estimates[other]])
        norm = 0.99 / peak if peak > 0.99 else 1.0

        ex_dir = args.out_dir / f"example_{i:02d}"
        ex_dir.mkdir(parents=True, exist_ok=True)
        files = [
            ("1_noisy", noisy),
            ("2_target", target),
            ("3_interference", interference),
            ("4_estimate_target", estimate),
            ("5_estimate_other", estimates[other]),
        ]
        for name, x in files:
            AudioEncoder((x * norm).unsqueeze(0), sample_rate=SR).to_file(str(ex_dir / f"{name}.wav"))

        print(f"[{i:02d}] in {in_sdr:+6.2f} dB  out {out_sdr:+6.2f} dB (stream {best})  "
              f"Δ {out_sdr - in_sdr:+.2f} dB  -> {ex_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--speech-root", type=Path, required=True)
    p.add_argument("--wham-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("samples"))
    p.add_argument("--n-examples", type=int, default=5)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
