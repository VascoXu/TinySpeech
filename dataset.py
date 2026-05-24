"""Datasets and speech-loading helpers for the beamforming + post-filter pipeline.

`SpatialAudioDataset` produces (beamformed, target) pairs by drawing a random pre-rendered
scene from a RIR bank (see render_bank.py), convolving fresh audio at runtime, and
MVDR-beamforming. `ProcessedSpeechDataset` is the matching reader for pre-rendered fixed
val/test sets (see render_val.py).
"""
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder

SR = 16000

# Dataset registry — short name -> speech root. Lets train.py take --dataset vctk librispeech
# instead of typing out full paths every invocation.
DATASETS = {
    "vctk":        Path("datasets/VCTK-Corpus-0.92/wav48_silence_trimmed"),
    "librispeech": Path("datasets/LibriSpeech/train-clean-360"),
}


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


def list_speech_files(roots) -> list:
    # Accepts a single Path or an iterable of Paths (for multi-corpus training).
    # VCTK 0.92 ships two mic renditions per utterance; keep only mic1 so each utterance is unique
    # (the filter is a no-op on LibriSpeech and other corpora).
    if isinstance(roots, (str, Path)):
        roots = [roots]
    files = []
    for r in roots:
        files.extend(p for p in Path(r).rglob("*.flac") if not p.name.endswith("_mic2.flac"))
    return sorted(files)


def _convolve_to_mics(audio: torch.Tensor, per_mic_rirs: torch.Tensor, n_samples: int) -> torch.Tensor:
    """audio: (T,)  +  per_mic_rirs: (M, rir_len)  ->  (M, n_samples)."""
    from beamforming import fft_convolve  # local import: keeps dataset module importable from beamforming
    return torch.stack([
        fft_convolve(audio, per_mic_rirs[m], n_samples)
        for m in range(per_mic_rirs.size(0))
    ])


class SpatialAudioDataset(Dataset):
    """Live multi-mic + MVDR training set — yields (beamformed, target) pairs.

    Each example:
      - Pick a random scene from the pre-rendered RIR bank (see render_bank.py).
      - Target dry voice convolved with the per-mic RIR at array origin (FG room).
      - K babble talkers (K ~ U[1, 3]) convolved with random (angle, radius) grid points (FG room).
      - 1 env-noise source convolved with a random (angle, radius) grid point (BG room).
      - Per-source peak normalization: FG ∈ [0.15, 0.4], BG ∈ [0.2, 0.5].
      - With probability `silence_prob` (default 0.05): zero out the target voice so the model
        learns to mute when no target is present (prevents noise-floor amplification at silent
        boundaries — a known Conv-TasNet failure mode with ReLU masks).
      - Sum per-mic, MVDR-beamform with all-ones steering + oracle noise covariance.

    Target options:
        "anechoic"    — the dry voice. Post-filter learns denoise + dereverb.
        "reverberant" — the target-only path through MVDR. Time-aligned with the
                        beamformed input; the post-filter task becomes denoise-only.
    """

    def __init__(self,
                 bank_path: Path,
                 speech_root,
                 wham_root: Path,
                 sample_rate: int = SR,
                 segment_seconds: float = 3.0,
                 epoch_size: int = 10000,
                 target_type: str = "anechoic",
                 silence_prob: float = 0.05):
        from beamforming import BABBLE_K_RANGE, FG_VOL_RANGE, BG_VOL_RANGE  # local: avoids circular hazard
        assert target_type in ("anechoic", "reverberant"), f"unknown target_type: {target_type}"

        # Bank is tens of MB per scene, kept resident. Each DataLoader worker fork
        # gets its own copy via copy-on-write — set num_workers conservatively.
        self.bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        self.speech_files = list_speech_files(speech_root)
        self.noise_files = sorted(Path(wham_root).rglob("*.wav"))
        assert self.speech_files, f"no .flac under {speech_root}"
        assert self.noise_files, f"no .wav under {wham_root}"

        self.sr = sample_rate
        self.n_samples = int(sample_rate * segment_seconds)
        self.epoch_size = epoch_size
        self.target_type = target_type
        self.silence_prob = silence_prob
        self._babble_k_range = BABBLE_K_RANGE
        self._fg_vol_range = FG_VOL_RANGE
        self._bg_vol_range = BG_VOL_RANGE

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, _):
        from beamforming import beamform_mvdr, compute_mvdr_weights, peak_normalize
        scene = self.bank[random.randrange(len(self.bank))]
        n = self.n_samples

        # Target at array origin (FG room)
        target_audio = random_segment(load_mono(random.choice(self.speech_files), self.sr), n)
        target_mic = _convolve_to_mics(target_audio, scene["target_rir"], n)
        target_mic = peak_normalize(target_mic, random.uniform(*self._fg_vol_range))

        # Silence-target augmentation
        if random.random() < self.silence_prob:
            target_audio = torch.zeros_like(target_audio)
            target_mic = torch.zeros_like(target_mic)

        # K babble talkers at random grid points (FG room)
        n_a, n_r = scene["babble_rirs"].shape[:2]
        babble_mic = torch.zeros_like(target_mic)
        k = random.randint(*self._babble_k_range)
        for _ in range(k):
            audio = random_segment(load_mono(random.choice(self.speech_files), self.sr), n)
            ai, ri = random.randrange(n_a), random.randrange(n_r)
            sig = _convolve_to_mics(audio, scene["babble_rirs"][ai, ri], n)
            babble_mic = babble_mic + peak_normalize(sig, random.uniform(*self._fg_vol_range))

        # Env at a random grid point (BG room)
        n_a_bg, n_r_bg = scene["bg_rirs"].shape[:2]
        env_audio = random_segment(load_mono(random.choice(self.noise_files), self.sr), n)
        ai, ri = random.randrange(n_a_bg), random.randrange(n_r_bg)
        env_mic = _convolve_to_mics(env_audio, scene["bg_rirs"][ai, ri], n)
        env_mic = peak_normalize(env_mic, random.uniform(*self._bg_vol_range))

        multi_mic = target_mic + babble_mic + env_mic

        # MVDR with all-ones steering (target at origin) + oracle noise covariance
        mvdr_w = compute_mvdr_weights(babble_mic + env_mic)
        beamformed = beamform_mvdr(multi_mic, mvdr_w)

        if self.target_type == "anechoic":
            target = target_audio
        else:
            target = beamform_mvdr(target_mic, mvdr_w)

        return beamformed.float(), target.float()


class ProcessedSpeechDataset(Dataset):
    """Pre-rendered (noisy, target) pairs from a .pt file (see render_val.py)."""

    def __init__(self, pt_path: Path):
        blob = torch.load(pt_path, map_location="cpu")
        self.noisy = blob["noisy"]
        self.target = blob["target"]
        assert self.noisy.size(0) == self.target.size(0)

    def __len__(self) -> int:
        return self.noisy.size(0)

    def __getitem__(self, idx: int):
        return self.noisy[idx], self.target[idx]
