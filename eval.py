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
TARGET_STREAM = 0



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
    for noisy, sources in tqdm(loader, desc="eval"):
        noisy, sources = noisy.to(device), sources.to(device)
        target = sources[:, 0]                                 # (B, T)
        est = model(noisy)[:, TARGET_STREAM]                         # (B, T)
        sdr = calc_sdr_torch(est, target)                            # (B,)

        sdr_in.append(calc_sdr_torch(noisy, target).cpu())
        sdr_out.append(sdr.cpu())
        pesq.append(perceptual_evaluation_speech_quality(est, target, fs=SR, mode="wb").cpu())
        stoi.append(short_time_objective_intelligibility(est, target, fs=SR).cpu())

    sdr_in, sdr_out = torch.cat(sdr_in), torch.cat(sdr_out)
    return {
        "si_sdr":  sdr_out,
        "si_sdri": sdr_out - sdr_in,
        "pesq":    torch.cat(pesq),
        "stoi":    torch.cat(stoi),
    }


# ---------- TEMP DIAGNOSTIC (delete me) ----------------------------------------
def dump_worst(model, ds, sisdri, n, out_dir, device):
    """Save noisy/clean/stream0/stream1 wavs per-folder for the N worst SI-SDRi samples."""
    from torchcodec.encoders import AudioEncoder
    out_dir.mkdir(parents=True, exist_ok=True)
    worst_idx = sisdri.argsort()[:n].tolist()
    print(f"\nworst {n} samples by SI-SDRi -> {out_dir}/")
    for rank, idx in enumerate(worst_idx):
        noisy, sources = ds[idx]
        clean = sources[0]
        with torch.no_grad():
            ests = model(noisy.unsqueeze(0).to(device))[0].cpu()       # (2, T)
        sdr_0 = calc_sdr_torch(ests[0].unsqueeze(0), clean.unsqueeze(0)).item()
        sdr_1 = calc_sdr_torch(ests[1].unsqueeze(0), clean.unsqueeze(0)).item()
        sdri  = sisdri[idx].item()

        peak = max(noisy.abs().max(), clean.abs().max(),
                   ests[0].abs().max(), ests[1].abs().max()).item()
        norm = 0.99 / peak if peak > 0.99 else 1.0

        sample_dir = out_dir / f"rank{rank:02d}_idx{idx:04d}_sisdri{sdri:+.1f}dB"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for name, x in [("noisy", noisy), ("clean", clean),
                        ("stream0", ests[0]), ("stream1", ests[1])]:
            AudioEncoder((x * norm).unsqueeze(0), sample_rate=SR).to_file(
                str(sample_dir / f"{name}.wav"))
        print(f"  rank {rank:02d}  idx {idx:4d}  SI-SDRi {sdri:+6.2f}   "
              f"stream0 SDR {sdr_0:+6.2f}   stream1 SDR {sdr_1:+6.2f}")
# ---------- END TEMP DIAGNOSTIC -------------------------------------------------


def main(args):
    device = torch.device(args.device)
    model = TasNet(num_spk=2, causal=True, sr=SR).to(device)
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

    # TEMP DIAGNOSTIC: dump worst-N samples for listening (delete me)
    if args.dump_worst > 0:
        dump_worst(model, ds, results["si_sdri"], args.dump_worst, args.worst_out, device)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--test-pt", type=Path, required=True,
                   help="Pre-rendered (noisy, sources) test set — see prepare.py")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    # TEMP DIAGNOSTIC args (delete me)
    p.add_argument("--dump-worst", type=int, default=0,
                   help="(TEMP) Save audio of the N worst SI-SDRi samples for listening")
    p.add_argument("--worst-out", type=Path, default=Path("data/eval_worst"),
                   help="(TEMP) Output directory for --dump-worst")
    main(p.parse_args())
