"""Datasets for TinySpeech: dynamic on-the-fly mixing for train, pre-rendered for val/test."""
import random
from pathlib import Path

import torch
import torchaudio.functional as AF
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder

SR = 16000
HPF_HZ = 1200.0


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
    """On-the-fly mixing: target + babble + environmental noise, per call.

    Each __getitem__ draws fresh random parameters. The babble is a sum of K random
    LibriSpeech utterances attenuated by an extra `babble_atten_db` to emulate MVDR
    off-axis suppression. Both noisy and clean are high-passed at hpf_hz so the model
    never has to invent low-frequency content that the upstream pipeline removes.
    """

    def __init__(self,
                 speech_root: Path,
                 wham_root: Path,
                 sample_rate: int = SR,
                 segment_seconds: float = 4.0,
                 epoch_size: int = 20000,
                 snr_db_range=(-5.0, 15.0),
                 babble_atten_db_range=(6.0, 12.0),
                 gain_db_range=(-15.0, 5.0),
                 babble_n_range=(3, 10),
                 hpf_hz: float = HPF_HZ):
        self.sr = sample_rate
        self.n_samples = int(sample_rate * segment_seconds)
        self.epoch_size = epoch_size
        self.snr_range = snr_db_range
        self.atten_range = babble_atten_db_range
        self.gain_range = gain_db_range
        self.babble_n_range = babble_n_range
        self.hpf_hz = hpf_hz

        self.speech_files = _list_speech_files(speech_root)
        self.noise_files = sorted(Path(wham_root).rglob("*.wav"))
        if not self.speech_files:
            raise RuntimeError(f"No .flac files under {speech_root}")
        if not self.noise_files:
            raise RuntimeError(f"No .wav files under {wham_root}")

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, _idx: int):
        target = _random_segment(
            _load_mono(random.choice(self.speech_files), self.sr), self.n_samples)

        k = random.randint(*self.babble_n_range)
        babble = torch.zeros_like(target)
        for _ in range(k):
            voice = _random_segment(
                _load_mono(random.choice(self.speech_files), self.sr), self.n_samples)
            babble = babble + voice * random.uniform(0.5, 1.0)
        babble = babble / (k ** 0.5)  # keep total babble RMS roughly constant in K

        env = _random_segment(
            _load_mono(random.choice(self.noise_files), self.sr), self.n_samples)

        snr_db = random.uniform(*self.snr_range)
        atten_db = random.uniform(*self.atten_range)
        babble = _scale_to_snr(target, babble, snr_db + atten_db)
        env = _scale_to_snr(target, env, snr_db)
        mix = target + babble + env

        gain = 10 ** (random.uniform(*self.gain_range) / 20.0)
        noisy = AF.highpass_biquad(mix,    self.sr, self.hpf_hz) * gain
        clean = AF.highpass_biquad(target, self.sr, self.hpf_hz) * gain
        return noisy.float(), clean.float()


class FixedMixDataset(Dataset):
    """Loads pre-rendered (noisy, clean) tensor pairs from a .pt file.

    The .pt file is a dict {"noisy": (N, T), "clean": (N, T)} produced once with a
    fixed seed so val/test runs are reproducible across training runs.
    """

    def __init__(self, pt_path: Path):
        blob = torch.load(pt_path, map_location="cpu")
        self.noisy = blob["noisy"]
        self.clean = blob["clean"]
        assert self.noisy.shape == self.clean.shape

    def __len__(self) -> int:
        return self.noisy.size(0)

    def __getitem__(self, idx: int):
        return self.noisy[idx], self.clean[idx]
