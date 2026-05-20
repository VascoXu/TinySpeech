"""Datasets for TinySpeech: dynamic on-the-fly mixing for train, pre-rendered for val/test."""
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder

SR = 16000
SNR_DB_DEFAULT = (-5.0, 15.0)         # voice-vs-interference SNR drawn uniformly
GAIN_DB_DEFAULT = (-15.0, 5.0)        # final per-sample gain (simulates mic-distance variation)
BABBLE_N_DEFAULT = (3, 10)            # number of stacked interfering voices
BABBLE_GAIN_RANGE = (0.5, 1.0)        # per-voice gain inside the babble stack


def _load_mono(path: Path, sample_rate: int) -> torch.Tensor:
    samples = AudioDecoder(str(path), sample_rate=sample_rate).get_all_samples()
    return samples.data.mean(dim=0)


def _random_segment(wav: torch.Tensor, n_samples: int) -> torch.Tensor:
    if wav.size(0) >= n_samples:
        start = random.randint(0, wav.size(0) - n_samples)
        return wav[start:start + n_samples]
    out = torch.zeros(n_samples, dtype=wav.dtype)
    out[:wav.size(0)] = wav
    return out


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean().clamp_min(1e-10).sqrt()


def _scale_to_snr(target: torch.Tensor, interferer: torch.Tensor, snr_db: float) -> torch.Tensor:
    return interferer * (_rms(target) / (_rms(interferer) * 10 ** (snr_db / 20)))


def _list_speech_files(root: Path) -> list:
    """List .flac files under root; drops VCTK 0.92 *_mic2.flac so each utterance is counted once."""
    return sorted(p for p in Path(root).rglob("*.flac") if not p.name.endswith("_mic2.flac"))


class DynamicMixDataset(Dataset):
    """On-the-fly mixing of clean target + K babble voices + WHAM environmental noise.

    Full-band (no HPF), ClearBuds-style framing. Each __getitem__ draws fresh random
    parameters: target speaker, K interfering speakers, SNR, gain, env noise clip.
    """

    def __init__(self,
                 speech_root: Path,
                 wham_root: Path,
                 sample_rate: int = SR,
                 segment_seconds: float = 4.0,
                 epoch_size: int = 20000,
                 snr_db_range=SNR_DB_DEFAULT,
                 gain_db_range=GAIN_DB_DEFAULT,
                 babble_n_range=BABBLE_N_DEFAULT,
                 silence_prepend_prob: float = 0.0,
                 silence_max_seconds: float = 2.0):
        self.sr = sample_rate
        self.n_samples = int(sample_rate * segment_seconds)
        self.epoch_size = epoch_size
        self.snr_range = snr_db_range
        self.gain_range = gain_db_range
        self.babble_n_range = babble_n_range
        self.silence_prepend_prob = silence_prepend_prob
        self.silence_max_samples = int(silence_max_seconds * self.sr)

        self.speech_files = _list_speech_files(speech_root)
        self.noise_files = sorted(Path(wham_root).rglob("*.wav"))
        if not self.speech_files:
            raise RuntimeError(f"No .flac files under {speech_root}")
        if not self.noise_files:
            raise RuntimeError(f"No .wav files under {wham_root}")

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, _idx: int):
        # Optional leading silence keeps voice activity at end; guarantees >=1s of voice.
        silence_samples = 0
        if random.random() < self.silence_prepend_prob:
            max_sil = min(self.silence_max_samples, self.n_samples - self.sr)
            silence_samples = random.randint(0, max_sil)
        voice_samples = self.n_samples - silence_samples

        voice = _random_segment(
            _load_mono(random.choice(self.speech_files), self.sr), voice_samples)
        target = torch.zeros(self.n_samples, dtype=voice.dtype)
        target[silence_samples:] = voice

        k = random.randint(*self.babble_n_range)
        babble = torch.zeros_like(target)
        for _ in range(k):
            v = _random_segment(
                _load_mono(random.choice(self.speech_files), self.sr), self.n_samples)
            babble = babble + v * random.uniform(*BABBLE_GAIN_RANGE)
        babble = babble / (k ** 0.5)  # rough RMS normalization across stack

        env = _random_segment(
            _load_mono(random.choice(self.noise_files), self.sr), self.n_samples)

        # Scale interferers against voice (not silence-padded target) so SNR stays meaningful.
        snr_db = random.uniform(*self.snr_range)
        babble = _scale_to_snr(voice, babble, snr_db)
        env = _scale_to_snr(voice, env, snr_db)
        interference = babble + env

        gain = 10 ** (random.uniform(*self.gain_range) / 20.0)
        target_out = (target * gain).float()
        interference_out = (interference * gain).float()
        noisy = target_out + interference_out
        sources = torch.stack([target_out, interference_out], dim=0)  # (2, T)
        return noisy, sources


class FixedMixDataset(Dataset):
    """Loads pre-rendered (noisy, sources) pairs from a .pt file.

    The .pt file is a dict {"noisy": (N, T), "sources": (N, 2, T)} where
    sources[:, 0] is the clean target and sources[:, 1] is the interference.
    """

    def __init__(self, pt_path: Path):
        blob = torch.load(pt_path, map_location="cpu")
        self.noisy = blob["noisy"]
        self.sources = blob["sources"]
        assert self.noisy.size(0) == self.sources.size(0)

    def __len__(self) -> int:
        return self.noisy.size(0)

    def __getitem__(self, idx: int):
        return self.noisy[idx], self.sources[idx]
