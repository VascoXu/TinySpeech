"""Evaluate a checkpoint on a fixed test set: SI-SDR, SI-SDRi, PESQ, STOI."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from tqdm import tqdm

from dataset import ProcessedSpeechDataset
from metrics import calc_sdr_torch
from model import TasNet

SR = 16000


def fmt_row(name: str, x: torch.Tensor) -> str:
    """One-line distribution stats: mean / median / std / min / max (NaN-safe)."""
    valid = x[~x.isnan()]
    n_nan = int(x.isnan().sum())
    row = (f"  {name:8} mean {valid.mean():6.2f}  median {valid.median():6.2f}  "
           f"std {valid.std():5.2f}  min {valid.min():6.2f}  max {valid.max():6.2f}")
    return row + (f"  ({n_nan} nan)" if n_nan else "")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    sdr_in, sdr_out, pesq, stoi = [], [], [], []
    for noisy, target in tqdm(loader, desc="eval"):
        noisy, target = noisy.to(device), target.to(device)
        est = model(noisy)[:, 0]                                       # (B, T)

        sdr_in.append(calc_sdr_torch(noisy, target).cpu())
        sdr_out.append(calc_sdr_torch(est, target).cpu())
        pesq.append(perceptual_evaluation_speech_quality(est, target, fs=SR, mode="wb").cpu())
        stoi.append(short_time_objective_intelligibility(est, target, fs=SR).cpu())

    sdr_in, sdr_out = torch.cat(sdr_in), torch.cat(sdr_out)
    return {
        "si_sdr":  sdr_out,
        "si_sdri": sdr_out - sdr_in,
        "pesq":    torch.cat(pesq),
        "stoi":    torch.cat(stoi),
    }


def main(args):
    device = torch.device(args.device)
    model = TasNet(causal=True, sr=SR).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} params ({n_params/1e6:.2f}M), "
          f"checkpoint epoch {ckpt.get('epoch', '?')}")

    ds = ProcessedSpeechDataset(args.test_pt)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    results = evaluate(model, loader, device)
    print(f"\ntest set: {args.test_pt} ({len(ds)} examples)")
    print(fmt_row("SI-SDR",  results["si_sdr"]))
    print(fmt_row("SI-SDRi", results["si_sdri"]))
    print(fmt_row("PESQ",    results["pesq"]))
    print(fmt_row("STOI",    results["stoi"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--test-pt", type=Path, required=True,
                   help="Pre-rendered (noisy, target) test set — see beam_prepare.py / prepare.py")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    main(p.parse_args())
