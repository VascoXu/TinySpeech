"""Pack a fixed (beamformed, sources) tensor file from SpatialAudioDataset for reproducible eval.

Matches the format produced by prepare.py (a {noisy, sources} dict loadable by
ProcessedSpeechDataset), so it drops into the same train.py / eval.py paths without
any consumer changes — the only difference is that `noisy` here is a beamformed
single-channel signal instead of a raw single-mic mixture.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dataset import DATASETS, SpatialAudioDataset


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    speech_roots = [DATASETS[d] for d in args.dataset]
    print(f"dataset: {', '.join(args.dataset)}  target_type: {args.target_type}")
    print(f"beam bank: {args.beam_bank}")
    ds = SpatialAudioDataset(
        bank_path=args.beam_bank,
        speech_root=speech_roots,
        wham_root=args.wham_root,
        segment_seconds=args.segment_seconds,
        target_type=args.target_type,
    )

    noisy, targets = [], []
    for _ in tqdm(range(args.n_examples), desc="rendering"):
        n, t = ds[0]
        noisy.append(n)
        targets.append(t)

    args.out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"noisy": torch.stack(noisy), "target": torch.stack(targets)}, args.out_pt)
    print(f"saved {args.n_examples} examples -> {args.out_pt}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", nargs="+", choices=sorted(DATASETS.keys()), required=True,
                   metavar="NAME",
                   help=f"Speech dataset(s) for target+babble. Choices: {sorted(DATASETS.keys())}")
    p.add_argument("--wham-root", type=Path, required=True)
    p.add_argument("--beam-bank", type=Path, required=True,
                   help="Multi-mic RIR bank .pt rendered by beam_bank.py")
    p.add_argument("--out-pt", type=Path, default=Path("datasets/val_beam.pt"))
    p.add_argument("--n-examples", type=int, default=200)
    p.add_argument("--segment-seconds", type=float, default=3.0)
    p.add_argument("--target-type", choices=["anechoic", "reverberant"], default="anechoic")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
