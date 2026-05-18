"""Render a fixed (noisy, clean) tensor pack from DynamicMixDataset for reproducible eval."""
import argparse
import random
from pathlib import Path

import torch
from tqdm import tqdm

from dataset import DynamicMixDataset


def main(args):
    random.seed(args.seed)
    ds = DynamicMixDataset(
        speech_root=args.speech_root,
        wham_root=args.wham_root,
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
    )

    noisy, clean = [], []
    for _ in tqdm(range(args.n_examples), desc="rendering"):
        n, c = ds[0]  # idx is ignored
        noisy.append(n)
        clean.append(c)

    args.out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"noisy": torch.stack(noisy), "clean": torch.stack(clean)},
        args.out_pt,
    )
    print(f"saved {args.n_examples} examples -> {args.out_pt}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--speech-root", type=Path, required=True)
    p.add_argument("--wham-root", type=Path, required=True)
    p.add_argument("--out-pt", type=Path, required=True)
    p.add_argument("--n-examples", type=int, default=200)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--segment-seconds", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
