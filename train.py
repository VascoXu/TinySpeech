"""Train TinySpeech: causal Conv-TasNet post-filter on top of MVDR-beamformed audio.

Loss follows the ClearBuds recipe (5·L1 + 0.1·SC + 0.1·log-mag, multi-res STFT) but on a
single output stream — we predict the target voice only. See the conversation history /
README for why C=1 here (target-only) rather than ClearBuds' C=2 (multi-mic target).

Inputs are *beamformed* single-channel signals from SpatialAudioDataset; targets are the
(reverberant or anechoic) target voice.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DATASETS, ProcessedSpeechDataset, SpatialAudioDataset
from losses import MultiResSTFTLoss
from metrics import calc_sdr_torch
from model import TasNet

GRAD_CLIP = 5.0      # max grad L2 norm
LR_DECAY = 0.95      # exponential per-epoch LR decay
W_L1 = 5.0           # ClearBuds: solver.py:205


def compute_loss(estimate, target, stft_loss):
    """ClearBuds-style loss recipe on a single output stream."""
    # estimate: (B, 1, T)   target: (B, T)
    return W_L1 * F.l1_loss(estimate[:, 0], target) + stft_loss(estimate[:, 0], target)


def train_one_epoch(model, loader, optimizer, stft_loss, device):
    model.train()
    total = 0.0
    for noisy, target in tqdm(loader, desc="train", leave=False):
        noisy, target = noisy.to(device), target.to(device)
        estimate = model(noisy)                            # (B, 1, T)
        loss = compute_loss(estimate, target, stft_loss)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, stft_loss, device):
    model.eval()
    losses, sisdrs = [], []
    for noisy, target in tqdm(loader, desc="val", leave=False):
        noisy, target = noisy.to(device), target.to(device)
        estimate = model(noisy)
        losses.append(compute_loss(estimate, target, stft_loss).item())
        sisdrs.append(calc_sdr_torch(estimate[:, 0], target).mean().item())
    return sum(losses) / len(losses), sum(sisdrs) / len(sisdrs)


def main(args):
    device = torch.device(args.device)
    print(f"device: {device}")

    speech_roots = [DATASETS[d] for d in args.dataset]
    print(f"dataset: {', '.join(args.dataset)}  target_type: {args.target_type}")
    print(f"beam bank: {args.beam_bank}")
    train_set = SpatialAudioDataset(
        bank_path=args.beam_bank,
        speech_root=speech_roots,
        wham_root=args.wham_root,
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        target_type=args.target_type,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = TasNet(causal=True, sr=args.sample_rate).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} params ({n_params/1e6:.2f}M)")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=LR_DECAY)
    stft_loss = MultiResSTFTLoss().to(device)

    if args.smoke_test:
        rf_ms = model.receptive_field * model.stride / args.sample_rate * 1000
        print(f"TCN receptive field: {model.receptive_field} frames ({rf_ms:.0f} ms past context)")
        noisy, target = next(iter(train_loader))
        print(f"batch shapes  noisy: {tuple(noisy.shape)}  target: {tuple(target.shape)}")
        noisy, target = noisy.to(device), target.to(device)
        estimate = model(noisy)
        print(f"estimate shape: {tuple(estimate.shape)}")
        loss = compute_loss(estimate, target, stft_loss)
        loss.backward()
        assert not torch.isnan(loss), "loss is NaN"
        sisdr = calc_sdr_torch(estimate[:, 0].detach(), target).mean().item()
        print(f"loss: {loss.item():.4f}  si-sdr (untrained): {sisdr:+.2f} dB")
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
        train_loss = train_one_epoch(model, train_loader, optimizer, stft_loss, device)
        val_loss, val_sisdr = validate(model, val_loader, stft_loss, device)
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
                   help="Pre-rendered fixed validation set (.pt of (noisy, target) tensor pairs)")
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
    p.add_argument("--segment-seconds", type=float, default=3.0)
    p.add_argument("--beam-bank", type=Path, required=True,
                   help="Multi-mic RIR bank .pt rendered by render_bank.py")
    p.add_argument("--target-type", choices=["anechoic", "reverberant"], default="reverberant",
                   help="reverberant = target-only path through MVDR (denoise-only task, "
                        "exact noise = noisy - target via linearity of MVDR); "
                        "anechoic = dry voice (post-filter does denoise + dereverb)")
    main(p.parse_args())
