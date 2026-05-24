"""Room simulation + MVDR beamforming utilities.

Geometry:
  - 2D polygon rooms for both foreground (target + babble) and background (env noise)
  - 4-mic circular array centered at the origin
  - Target voice always at (0, 0); babble + env at random (angle, radius)
  - Per-source peak normalization, separate volume ranges for FG vs BG

MVDR uses all-ones steering (target at array center → zero-delay across mics) and an
oracle noise covariance (Σ_NN = noise_only_mixture @ noise_only_mixture.H / N_frames).
"""
import random

import numpy as np
import pyroomacoustics as pra
import torch

from dataset import SR

# ---- Room geometry ----
FG_WALL_HALF_MIN = 15.0           # FG room walls in ±[15, 20] m
FG_WALL_HALF_MAX = 20.0
BG_WALL_HALF_MIN = 20.0           # BG room walls in ±[20, 40] m  (separate larger room)
BG_WALL_HALF_MAX = 40.0
FG_ABSORPTION_RANGE = (0.1, 0.99)
BG_ABSORPTION_RANGE = (0.5, 0.99) # higher absorption → diffuse-sounding distant noise
MAX_ORDER = 10

# ---- Mic array ----
MIC_RADIUS_RANGE = (0.05, 0.125)  # circular array radius (m)
N_MICS_DEFAULT = 4
MIC_PHI0 = 0.0                    # starting angle of first mic

# ---- Sources ----
BABBLE_K_RANGE = (1, 3)           # random number of babble talkers per example
BABBLE_RADIUS_RANGE = (1.0, 5.0)
BG_RADIUS_RANGE = (10.0, 20.0)

# ---- Volumes (peak normalization) ----
FG_VOL_RANGE = (0.15, 0.4)
BG_VOL_RANGE = (0.2, 0.5)

# ---- MVDR ----
MVDR_N_FFT = 1024
MVDR_HOP = 256
MVDR_DIAG_LOAD = 1e-3
MVDR_DIAG_FLOOR = 1e-8


def sample_room(half_min, half_max, absorption_range=FG_ABSORPTION_RANGE):
    """4-corner 2D polygon with walls in ±[half_min, half_max]."""
    lw = -random.uniform(half_min, half_max)
    rw =  random.uniform(half_min, half_max)
    bw = -random.uniform(half_min, half_max)
    tw =  random.uniform(half_min, half_max)
    corners = np.array([[lw, bw], [lw, tw], [rw, tw], [rw, bw]]).T   # (2, 4)
    absorption = random.uniform(*absorption_range)
    return corners, absorption


def place_circular_array(n_mics, mic_radius):
    """2D circular mic array centered at origin. Returns (2, n_mics)."""
    return pra.circular_2D_array(center=[0.0, 0.0], M=n_mics, phi0=MIC_PHI0, radius=mic_radius)


def render_to_mics(audio, src_pos, mic_positions, corners, absorption, n_samples):
    """Render a single source to all mics in a 2D polygon room. Returns (M, T_render) cropped."""
    room = pra.Room.from_corners(corners, fs=SR, max_order=MAX_ORDER,
                                 materials=pra.Material(absorption))
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))
    room.add_source(list(src_pos), signal=audio.numpy())
    room.image_source_model()
    room.simulate()
    sig = torch.from_numpy(room.mic_array.signals).float()
    return sig[:, :n_samples]


def peak_normalize(sig, target_peak):
    """Scale signal so its abs().max() == target_peak. Pass-through if signal is near-zero."""
    peak = sig.abs().max()
    if peak < 1e-9:
        return sig
    return sig * (target_peak / peak)


def fft_convolve(signal: torch.Tensor, rir: torch.Tensor, n_samples: int) -> torch.Tensor:
    """FFT-based 1D conv; returns first n_samples (direct path aligned with original onset)."""
    n_fft = 1 << (signal.size(0) + rir.size(0) - 2).bit_length()
    S = torch.fft.rfft(signal, n=n_fft)
    R = torch.fft.rfft(rir, n=n_fft)
    return torch.fft.irfft(S * R, n=n_fft)[:n_samples]


def compute_mvdr_weights(noise_mic, n_fft=MVDR_N_FFT, hop=MVDR_HOP,
                         diag_load=MVDR_DIAG_LOAD, diag_floor=MVDR_DIAG_FLOOR):
    """MVDR weights with all-ones steering (target at array center → omnidirectional).

    noise_mic: (M, T) noise-only signal (oracle = babble + env). Returns (F, M) complex weights.
    """
    M = noise_mic.shape[0]
    window = torch.hann_window(n_fft)
    N = torch.stft(noise_mic, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                   window=window, return_complex=True, center=True)        # (M, F, L)
    L_frames = N.shape[-1]
    N_fml = N.permute(1, 0, 2)                                              # (F, M, L)
    sigma = (N_fml @ N_fml.conj().transpose(-2, -1)) / L_frames             # (F, M, M)

    # Scale-adaptive diagonal load (low-freq bins are near-rank-1 → singular at float32)
    trace = sigma.diagonal(dim1=-2, dim2=-1).real.mean(-1)                  # (F,)
    reg = (diag_load * trace).clamp_min(diag_floor)                         # (F,)
    eye = torch.eye(M, dtype=sigma.dtype).unsqueeze(0)
    sigma = sigma + reg[:, None, None] * eye

    # Target at array center → zero-delay across all mics → steering = all-ones
    F_bins = N.shape[1]
    d = torch.ones(F_bins, M, dtype=sigma.dtype)                            # (F, M)
    d_col = d.unsqueeze(-1)
    sigma_inv_d = torch.linalg.solve(sigma, d_col)
    denom = (d_col.conj().transpose(-2, -1) @ sigma_inv_d).squeeze(-1).squeeze(-1)
    return sigma_inv_d.squeeze(-1) / denom.unsqueeze(-1)                    # (F, M)


def beamform_mvdr(multi_mic, weights, n_fft=MVDR_N_FFT, hop=MVDR_HOP):
    """Apply per-bin MVDR weights to a (M, T) signal. Returns (T,)."""
    length = multi_mic.shape[1]
    window = torch.hann_window(n_fft)
    X = torch.stft(multi_mic, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                   window=window, return_complex=True, center=True)        # (M, F, L)
    X_fml = X.permute(1, 0, 2)                                              # (F, M, L)
    Y = (weights.conj().unsqueeze(-1) * X_fml).sum(dim=1)                   # (F, L)
    return torch.istft(Y, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                       window=window, center=True, length=length)
