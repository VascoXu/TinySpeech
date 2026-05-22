"""Render a few training-distribution mixtures, save each component separately for listening."""
import argparse
import random
from pathlib import Path

import torch
from torchcodec.encoders import AudioEncoder

from dataset import (
    load_mono, random_segment, scale_to_snr, list_speech_files,
    SR, SNR_DB_DEFAULT, GAIN_DB_DEFAULT, BABBLE_N_DEFAULT, BABBLE_GAIN_RANGE,
)
from rir import fft_convolve


def save_wav(path: Path, x: torch.Tensor) -> None:
    AudioEncoder(x.unsqueeze(0), sample_rate=SR).to_file(str(path))


def main(args):
    random.seed(args.seed)
    speech_files = list_speech_files(args.speech_root)
    noise_files = sorted(Path(args.wham_root).rglob("*.wav"))
    assert speech_files, f"no .flac under {args.speech_root}"
    assert noise_files, f"no .wav under {args.wham_root}"
    rir_bank = torch.load(args.rir_bank, map_location="cpu") if args.rir_bank else None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_samples = int(SR * args.segment_seconds)

    for i in range(args.n_examples):
        # mirror DynamicSpeechDataset.__getitem__ but keep each intermediate signal exposed
        rir = rir_bank[random.randrange(len(rir_bank))] if rir_bank is not None else None

        target = random_segment(load_mono(random.choice(speech_files), SR), n_samples)
        if rir is not None:
            target = fft_convolve(target, rir, n_samples)

        k = random.randint(*BABBLE_N_DEFAULT)
        babble = torch.zeros_like(target)
        for _ in range(k):
            voice = random_segment(load_mono(random.choice(speech_files), SR), n_samples)
            if rir is not None:
                voice = fft_convolve(voice, rir, n_samples)
            babble = babble + voice * random.uniform(*BABBLE_GAIN_RANGE)
        babble = babble / (k ** 0.5)

        env = random_segment(load_mono(random.choice(noise_files), SR), n_samples)

        snr_db = random.uniform(*SNR_DB_DEFAULT)
        babble = scale_to_snr(target, babble, snr_db)
        env = scale_to_snr(target, env, snr_db)

        gain_db = random.uniform(*GAIN_DB_DEFAULT)
        gain = 10 ** (gain_db / 20.0)
        clean = target * gain
        noisy = (target + babble + env) * gain

        components = {
            "1_target": target * gain,
            "2_babble": babble * gain,
            "3_env":    env * gain,
            "4_noisy":  noisy,
            "5_clean":  clean,
        }
        peak = max(x.abs().max().item() for x in components.values())
        norm = 0.99 / peak if peak > 0.99 else 1.0

        ex_dir = args.out_dir / f"example_{i:02d}"
        ex_dir.mkdir(exist_ok=True)
        for name, x in components.items():
            save_wav(ex_dir / f"{name}.wav", x * norm)

        print(f"[{i:02d}] K={k}  SNR={snr_db:+.1f}dB  gain={gain_db:+.1f}dB  -> {ex_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--speech-root", type=Path, required=True)
    p.add_argument("--wham-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("preview"))
    p.add_argument("--n-examples", type=int, default=5)
    p.add_argument("--segment-seconds", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rir-bank", type=Path, default=None,
                   help="Pre-rendered RIR bank .pt (see rir.py). Adds reverb to target + babble.")
    main(p.parse_args())
