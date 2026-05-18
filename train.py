"""Train TinySpeech: causal Conv-TasNet for single-channel babble denoising at 16 kHz."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from conv_tasnet import TasNet
from sdr import calc_sdr_torch
from dataset import DynamicMixDataset, FixedMixDataset
from losses import L1MultiResSTFTLoss


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for noisy, clean in tqdm(loader, desc="train", leave=False):
        noisy, clean = noisy.to(device), clean.to(device)
        estimate = model(noisy).squeeze(1)
        loss = loss_fn(estimate, clean)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    losses, sisdrs = [], []
    for noisy, clean in tqdm(loader, desc="val", leave=False):
        noisy, clean = noisy.to(device), clean.to(device)
        estimate = model(noisy).squeeze(1)
        losses.append(loss_fn(estimate, clean).item())
        sisdrs.append(calc_sdr_torch(estimate, clean).mean().item())
    return sum(losses) / len(losses), sum(sisdrs) / len(sisdrs)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_set = DynamicMixDataset(
        speech_root=args.speech_root,
        wham_root=args.wham_root,
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = TasNet(num_spk=1, causal=True, sr=args.sample_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    loss_fn = L1MultiResSTFTLoss().to(device)

    if args.smoke_test:
        rf_ms = model.receptive_field * model.stride / args.sample_rate * 1000
        print(f"params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"TCN receptive field: {model.receptive_field} frames ({rf_ms:.0f} ms past context)")
        noisy, clean = next(iter(train_loader))
        print(f"batch shapes  noisy: {tuple(noisy.shape)}  clean: {tuple(clean.shape)}")
        noisy, clean = noisy.to(device), clean.to(device)
        estimate = model(noisy).squeeze(1)
        print(f"estimate shape: {tuple(estimate.shape)}")
        loss = loss_fn(estimate, clean)
        loss.backward()
        assert not torch.isnan(loss), "loss is NaN"
        sisdr = calc_sdr_torch(estimate.detach(), clean).mean().item()
        print(f"loss: {loss.item():.4f}  si-sdr (untrained): {sisdr:+.2f} dB")
        print("smoke test passed.")
        return

    assert args.val_pt is not None, "--val-pt is required for training"
    val_set = FixedMixDataset(args.val_pt)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_sisdr = -float("inf")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_sisdr = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        print(f"epoch {epoch:03d}  train {train_loss:.4f}  "
              f"val {val_loss:.4f}  si-sdr {val_sisdr:+.2f} dB")

        ckpt = {"model": model.state_dict(), "epoch": epoch, "val_sisdr": val_sisdr}
        torch.save(ckpt, args.ckpt_dir / "last.pt")
        if val_sisdr > best_sisdr:
            best_sisdr = val_sisdr
            torch.save(ckpt, args.ckpt_dir / "best.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--speech-root", type=Path, required=True)
    p.add_argument("--wham-root", type=Path, required=True)
    p.add_argument("--val-pt", type=Path, default=None,
                   help="Pre-rendered fixed validation set (.pt of (noisy, clean) tensor pairs)")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run one fwd/bwd pass on a single train batch and exit")
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--segment-seconds", type=float, default=4.0)
    main(p.parse_args())
