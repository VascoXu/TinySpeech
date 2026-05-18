"""Render a few training-distribution mixtures and save each intermediate as WAV, for listening."""
import argparse
import random
from pathlib import Path

import torch
import torchaudio.functional as AF
from torchcodec.encoders import AudioEncoder

from dataset import _load_mono, _random_segment, _scale_to_snr, _list_speech_files, SR, HPF_HZ


def save_wav(path: Path, x: torch.Tensor) -> None:
    AudioEncoder(x.unsqueeze(0), sample_rate=SR).to_file(str(path))


def main(args):
    random.seed(args.seed)
    speech_files = _list_speech_files(args.speech_root)
    noise_files = sorted(Path(args.wham_root).rglob("*.wav"))
    assert speech_files, f"No .flac under {args.speech_root}"
    assert noise_files, f"No .wav under {args.wham_root}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_samples = int(SR * args.segment_seconds)

    for i in range(args.n_examples):
        target = _random_segment(_load_mono(random.choice(speech_files), SR), n_samples)

        k = random.randint(3, 10)
        babble = torch.zeros_like(target)
        for _ in range(k):
            voice = _random_segment(_load_mono(random.choice(speech_files), SR), n_samples)
            babble = babble + voice * random.uniform(0.5, 1.0)
        babble = babble / (k ** 0.5)

        env = _random_segment(_load_mono(random.choice(noise_files), SR), n_samples)

        snr_db = random.uniform(-5.0, 15.0)
        atten_db = random.uniform(6.0, 12.0)
        babble_s = _scale_to_snr(target, babble, snr_db + atten_db)
        env_s = _scale_to_snr(target, env, snr_db)
        mix = target + babble_s + env_s

        gain_db = random.uniform(-15.0, 5.0)
        gain = 10 ** (gain_db / 20.0)
        noisy = AF.highpass_biquad(mix, SR, HPF_HZ) * gain
        clean = AF.highpass_biquad(target, SR, HPF_HZ) * gain

        # Group-normalize so PCM16 save doesn't clip; preserves relative levels within an example.
        peak = max(t.abs().max().item() for t in [target, babble_s, env_s, noisy, clean])
        norm = 0.99 / peak if peak > 0.99 else 1.0

        ex_dir = args.out_dir / f"example_{i:02d}"
        ex_dir.mkdir(exist_ok=True)
        save_wav(ex_dir / "1_target.wav",       target   * norm)
        save_wav(ex_dir / "2_babble.wav",       babble_s * norm)
        save_wav(ex_dir / "3_env.wav",          env_s    * norm)
        save_wav(ex_dir / "4_noisy.wav",        noisy    * norm)
        save_wav(ex_dir / "5_clean_target.wav", clean    * norm)

        print(f"[{i:02d}] K={k}  SNR={snr_db:+.1f}dB  "
              f"babble_atten={atten_db:.1f}dB  gain={gain_db:+.1f}dB  -> {ex_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--speech-root", type=Path, required=True)
    p.add_argument("--wham-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("preview_mixes"))
    p.add_argument("--n-examples", type=int, default=5)
    p.add_argument("--segment-seconds", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
