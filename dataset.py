"""Audio mixing datasets: random fresh mixes for training, packed for reproducible eval."""
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder

from rir import fft_convolve

SR = 16000
SNR_DB_DEFAULT = (-5.0, 15.0)         # voice-vs-interference SNR drawn uniformly
GAIN_DB_DEFAULT = (-15.0, 5.0)        # final per-sample gain (simulates mic-distance variation)
BABBLE_N_DEFAULT = (3, 10)            # number of stacked interfering voices
BABBLE_GAIN_RANGE = (0.5, 1.0)        # per-voice gain inside the babble stack


def load_mono(path: Path, sample_rate: int) -> torch.Tensor:
    samples = AudioDecoder(str(path), sample_rate=sample_rate).get_all_samples()
    return samples.data.mean(dim=0)


def random_segment(wav: torch.Tensor, n_samples: int) -> torch.Tensor:
    if wav.size(0) >= n_samples:
        start = random.randint(0, wav.size(0) - n_samples)
        return wav[start:start + n_samples]
    out = torch.zeros(n_samples, dtype=wav.dtype)
    out[:wav.size(0)] = wav
    return out


def rms(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean().clamp_min(1e-10).sqrt()


def scale_to_snr(target: torch.Tensor, interferer: torch.Tensor, snr_db: float) -> torch.Tensor:
    return interferer * (rms(target) / (rms(interferer) * 10 ** (snr_db / 20)))


def list_speech_files(root: Path) -> list:
    # VCTK 0.92 ships two mic renditions per utterance; keep only mic1 so each utterance is unique.
    return sorted(p for p in Path(root).rglob("*.flac") if not p.name.endswith("_mic2.flac"))


class DynamicSpeechDataset(Dataset):
    """Generates target voice + K babble voices + env noise."""
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
                 silence_max_seconds: float = 2.0,
                 rir_bank: torch.Tensor = None,
                 reverb_prob: float = 1.0):
        self.sr = sample_rate
        self.n_samples = int(sample_rate * segment_seconds)
        self.epoch_size = epoch_size
        self.snr_range = snr_db_range
        self.gain_range = gain_db_range
        self.babble_n_range = babble_n_range
        self.silence_prepend_prob = silence_prepend_prob
        self.silence_max_samples = int(silence_max_seconds * self.sr)
        self.rir_bank = rir_bank
        self.reverb_prob = reverb_prob

        self.speech_files = list_speech_files(speech_root)
        self.noise_files = sorted(Path(wham_root).rglob("*.wav"))
        assert self.speech_files, f"no .flac files under {speech_root}"
        assert self.noise_files, f"no .wav files under {wham_root}"

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, _):
        # optional leading silence; keep >=1 s of voice
        sil = (random.randint(0, min(self.silence_max_samples, self.n_samples - self.sr))
               if random.random() < self.silence_prepend_prob else 0)
        n_voice = self.n_samples - sil

        # one room per scene — target and babble share the RIR; env stays dry (WHAM is already reverberant)
        rir = (self.rir_bank[random.randrange(len(self.rir_bank))]
               if self.rir_bank is not None and random.random() < self.reverb_prob
               else None)

        def load_voice(n):
            x = random_segment(load_mono(random.choice(self.speech_files), self.sr), n)
            return fft_convolve(x, rir, n) if rir is not None else x

        voice = load_voice(n_voice)
        target = torch.zeros(self.n_samples, dtype=voice.dtype)
        target[sil:] = voice

        k = random.randint(*self.babble_n_range)
        babble = torch.zeros_like(target)
        for _ in range(k):
            babble = babble + load_voice(self.n_samples) * random.uniform(*BABBLE_GAIN_RANGE)
        babble = babble / (k ** 0.5)  # rough RMS normalization across stack

        env = random_segment(load_mono(random.choice(self.noise_files), self.sr), self.n_samples)

        # SNR is measured against the dry voice signal, not the silence-padded target.
        snr_db = random.uniform(*self.snr_range)
        interference = scale_to_snr(voice, babble, snr_db) + scale_to_snr(voice, env, snr_db)

        gain = 10 ** (random.uniform(*self.gain_range) / 20.0)
        target = (target * gain).float()
        interference = (interference * gain).float()
        return target + interference, torch.stack([target, interference])


class ProcessedSpeechDataset(Dataset):
    """Pre-rendered (noisy, sources) pairs from a .pt file (see prepare.py)."""

    def __init__(self, pt_path: Path):
        blob = torch.load(pt_path, map_location="cpu")
        self.noisy = blob["noisy"]
        self.sources = blob["sources"]
        assert self.noisy.size(0) == self.sources.size(0)

    def __len__(self) -> int:
        return self.noisy.size(0)

    def __getitem__(self, idx: int):
        return self.noisy[idx], self.sources[idx]
