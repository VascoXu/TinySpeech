"""Train TinySpeech: causal Conv-TasNet for single-channel babble denoising at 16 kHz."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import TasNet
from metrics import calc_sdr_torch
from dataset import DATASETS, DynamicSpeechDataset, ProcessedSpeechDataset
from losses import L1MultiResSTFTLoss, PITLoss

GRAD_CLIP = 5.0      # max grad L2 norm
LR_DECAY = 0.95      # exponential per-epoch LR decay


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for noisy, sources in tqdm(loader, desc="train", leave=False):
        noisy, sources = noisy.to(device), sources.to(device)
        estimates = model(noisy)  # (B, 2, T)
        loss = loss_fn(estimates, sources)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    losses, sisdrs = [], []
    for noisy, sources in tqdm(loader, desc="val", leave=False):
        noisy, sources = noisy.to(device), sources.to(device)
        estimates = model(noisy)  # (B, 2, T)
        losses.append(loss_fn(estimates, sources).item())
        target = sources[:, 0]
        sdr_0 = calc_sdr_torch(estimates[:, 0], target)
        sdr_1 = calc_sdr_torch(estimates[:, 1], target)
        sisdrs.append(torch.maximum(sdr_0, sdr_1).mean().item())
    return sum(losses) / len(losses), sum(sisdrs) / len(sisdrs)


def main(args):
    device = torch.device(args.device)
    print(f"device: {device}")

    rir_bank = torch.load(args.rir_bank, map_location="cpu") if args.rir_bank else None
    if rir_bank is not None:
        print(f"reverb: {tuple(rir_bank.shape)} RIRs loaded from {args.rir_bank}")

    speech_roots = [DATASETS[d] for d in args.dataset]
    print(f"dataset: {', '.join(args.dataset)}")
    train_set = DynamicSpeechDataset(
        speech_root=speech_roots,
        wham_root=args.wham_root,
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        rir_bank=rir_bank,
        reverb_prob=args.reverb_prob,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = TasNet(num_spk=2, causal=True, sr=args.sample_rate).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} params ({n_params/1e6:.2f}M)")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=LR_DECAY)
    loss_fn = PITLoss(L1MultiResSTFTLoss(reduction="none")).to(device)

    if args.smoke_test:
        rf_ms = model.receptive_field * model.stride / args.sample_rate * 1000
        print(f"TCN receptive field: {model.receptive_field} frames ({rf_ms:.0f} ms past context)")
        noisy, sources = next(iter(train_loader))
        print(f"batch shapes  noisy: {tuple(noisy.shape)}  sources: {tuple(sources.shape)}")
        noisy, sources = noisy.to(device), sources.to(device)
        estimates = model(noisy)
        print(f"estimates shape: {tuple(estimates.shape)}")
        loss = loss_fn(estimates, sources)
        loss.backward()
        assert not torch.isnan(loss), "loss is NaN"
        target = sources[:, 0]
        sdr_0 = calc_sdr_torch(estimates[:, 0].detach(), target)
        sdr_1 = calc_sdr_torch(estimates[:, 1].detach(), target)
        sisdr = torch.maximum(sdr_0, sdr_1).mean().item()
        print(f"loss: {loss.item():.4f}  si-sdr (untrained, best stream): {sisdr:+.2f} dB")
        print("smoke test passed.")
        return

    assert args.val_pt is not None, "--val-pt is required for training"
    val_set = ProcessedSpeechDataset(args.val_pt)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_sisdr = -float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_sisdr = ckpt.get("best_sisdr", -float("inf"))
        print(f"resumed from {args.resume}: starting at epoch {start_epoch}, "
              f"prev best si-sdr {best_sisdr:+.2f} dB")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_sisdr = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        print(f"epoch {epoch:03d}  train {train_loss:.4f}  "
              f"val {val_loss:.4f}  si-sdr {val_sisdr:+.2f} dB")

        is_new_best = val_sisdr > best_sisdr
        if is_new_best:
            best_sisdr = val_sisdr

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "val_sisdr": val_sisdr,
            "best_sisdr": best_sisdr,
        }
        torch.save(ckpt, args.ckpt_dir / "last.pt")
        if is_new_best:
            torch.save(ckpt, args.ckpt_dir / "best.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", nargs="+", choices=sorted(DATASETS.keys()), required=True,
                   metavar="NAME",
                   help=f"Speech dataset(s) to train on (mix for OOD robustness). "
                        f"Choices: {sorted(DATASETS.keys())}")
    p.add_argument("--wham-root", type=Path, required=True)
    p.add_argument("--val-pt", type=Path, default=None,
                   help="Pre-rendered fixed validation set (.pt of (noisy, clean) tensor pairs)")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run one fwd/bwd pass on a single train batch and exit")
    p.add_argument("--device", type=str, default="cuda:3",
                   help='torch device (e.g. "cuda:3", "cuda:0", "cpu")')
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--resume", type=Path, default=None,
                   help="Resume from a checkpoint (loads model/optimizer/scheduler/best_sisdr; "
                        "continues from saved epoch+1 up to --epochs)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--segment-seconds", type=float, default=4.0)
    p.add_argument("--rir-bank", type=Path, default=None,
                   help="Pre-rendered RIR bank .pt (see rir.py). Enables room reverb on target + babble.")
    p.add_argument("--reverb-prob", type=float, default=1.0,
                   help="Probability of applying reverb when --rir-bank is set (rest is dry).")
    main(p.parse_args())
