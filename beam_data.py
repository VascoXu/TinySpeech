"""Live multi-mic + MVDR training dataset for the post-filter (ClearBuds-style).

Each example:
  - Pick a random scene from the pre-rendered RIR bank (see beam_bank.py).
  - Target dry voice convolved with the per-mic RIR at array origin (FG room).
  - K babble talkers (K ~ U[1, 3]) convolved with random (angle, radius) grid points (FG room).
  - 1 env-noise source convolved with a random (angle, radius) grid point (BG room).
  - Per-source peak normalization: FG ∈ [0.15, 0.4], BG ∈ [0.2, 0.5].
  - With probability `silence_prob` (default 0.05): zero out the target voice so the model
    learns to mute when no target is present (prevents noise-floor amplification at silent
    boundaries — a known Conv-TasNet failure mode with ReLU masks).
  - Sum per-mic, MVDR-beamform with all-ones steering + oracle noise covariance.
  - Return (beamformed, target).

Target options:
    "anechoic"    — the dry voice (project spec). Post-filter learns denoise + dereverb.
    "reverberant" — the target-only path through MVDR. Time-aligned with the
                    beamformed input; the post-filter task becomes denoise-only.
"""
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from beam import (
    BABBLE_K_RANGE, BG_VOL_RANGE, FG_VOL_RANGE,
    beamform_mvdr, compute_mvdr_weights, peak_normalize,
)
from dataset import SR, list_speech_files, load_mono, random_segment
from rir import fft_convolve


def _convolve_to_mics(audio: torch.Tensor, per_mic_rirs: torch.Tensor, n_samples: int) -> torch.Tensor:
    """audio: (T,)  +  per_mic_rirs: (M, rir_len)  ->  (M, n_samples)."""
    return torch.stack([
        fft_convolve(audio, per_mic_rirs[m], n_samples)
        for m in range(per_mic_rirs.size(0))
    ])


class SpatialAudioDataset(Dataset):
    """(beamformed_input, target) pairs. See module docstring."""

    def __init__(self,
                 bank_path: Path,
                 speech_root,
                 wham_root: Path,
                 sample_rate: int = SR,
                 segment_seconds: float = 3.0,
                 epoch_size: int = 10000,
                 target_type: str = "anechoic",
                 silence_prob: float = 0.05):
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

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, _):
        scene = self.bank[random.randrange(len(self.bank))]
        n = self.n_samples

        # Target at array origin (FG room)
        target_audio = random_segment(load_mono(random.choice(self.speech_files), self.sr), n)
        target_mic = _convolve_to_mics(target_audio, scene["target_rir"], n)
        target_mic = peak_normalize(target_mic, random.uniform(*FG_VOL_RANGE))

        # Silence-target augmentation: occasionally zero out the target so the model learns
        # "no target voice → output silence" instead of amplifying the mic noise floor
        # (a known Conv-TasNet pathology with the ReLU mask — see conversation notes).
        if random.random() < self.silence_prob:
            target_audio = torch.zeros_like(target_audio)
            target_mic = torch.zeros_like(target_mic)

        # K babble talkers at random grid points (FG room)
        n_a, n_r = scene["babble_rirs"].shape[:2]
        babble_mic = torch.zeros_like(target_mic)
        k = random.randint(*BABBLE_K_RANGE)
        for _ in range(k):
            audio = random_segment(load_mono(random.choice(self.speech_files), self.sr), n)
            ai, ri = random.randrange(n_a), random.randrange(n_r)
            sig = _convolve_to_mics(audio, scene["babble_rirs"][ai, ri], n)
            babble_mic = babble_mic + peak_normalize(sig, random.uniform(*FG_VOL_RANGE))

        # Env at a random grid point (BG room)
        n_a_bg, n_r_bg = scene["bg_rirs"].shape[:2]
        env_audio = random_segment(load_mono(random.choice(self.noise_files), self.sr), n)
        ai, ri = random.randrange(n_a_bg), random.randrange(n_r_bg)
        env_mic = _convolve_to_mics(env_audio, scene["bg_rirs"][ai, ri], n)
        env_mic = peak_normalize(env_mic, random.uniform(*BG_VOL_RANGE))

        multi_mic = target_mic + babble_mic + env_mic

        # MVDR with all-ones steering (target at origin) + oracle noise covariance
        mvdr_w = compute_mvdr_weights(babble_mic + env_mic)
        beamformed = beamform_mvdr(multi_mic, mvdr_w)

        if self.target_type == "anechoic":
            target = target_audio
        else:
            target = beamform_mvdr(target_mic, mvdr_w)

        return beamformed.float(), target.float()
