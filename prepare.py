"""Pack a fixed (noisy, sources) tensor file from DynamicSpeechDataset for reproducible eval."""
import argparse
import random
from pathlib import Path

import torch
from tqdm import tqdm

from dataset import DynamicSpeechDataset


def main(args):
    random.seed(args.seed)
    rir_bank = torch.load(args.rir_bank, map_location="cpu") if args.rir_bank else None
    ds = DynamicSpeechDataset(
        speech_root=args.speech_root,
        wham_root=args.wham_root,
        segment_seconds=args.segment_seconds,
        rir_bank=rir_bank,
    )

    noisy, sources = [], []
    for _ in tqdm(range(args.n_examples), desc="rendering"):
        n, s = ds[0]
        noisy.append(n)
        sources.append(s)

    args.out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"noisy": torch.stack(noisy), "sources": torch.stack(sources)}, args.out_pt)
    print(f"saved {args.n_examples} examples -> {args.out_pt}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--speech-root", type=Path, required=True)
    p.add_argument("--wham-root", type=Path, required=True)
    p.add_argument("--out-pt", type=Path, default=Path("datasets/val_vctk.pt"))
    p.add_argument("--n-examples", type=int, default=200)
    p.add_argument("--segment-seconds", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rir-bank", type=Path, default=None,
                   help="Pre-rendered RIR bank .pt (see rir.py). Bakes reverb into the val set.")
    main(p.parse_args())
