"""Multi-resolution STFT loss: spectral convergence + log-magnitude L1 across 3 STFT sizes.

Returns a single scalar = W_SC · mean(SC across resolutions) + W_MAG · mean(log-mag L1).
Used together with a waveform L1 term computed in train.py; the combined training recipe
is 5·L1 + 0.1·SC + 0.1·log-mag.

  - Spectral convergence (SC): ||S_target − S_est||_F / ||S_target||_F. Penalizes the
    structural error in magnitude; robust to overall level mismatch.
  - Log-mag L1: mean |log|S_target| − log|S_est||. Focuses on perceptual loudness scale.

Both terms reduce globally (one scalar across batch × freq × time), not per-sample.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Three STFT resolutions: short-fast, medium, long-fine. Catches transients and harmonics together.
FFT_SIZES = (1024, 2048, 512)
HOP_SIZES = (120, 240, 50)
WIN_SIZES = (600, 1200, 240)

W_SC = 0.1   # weight on spectral-convergence loss
W_MAG = 0.1  # weight on log-magnitude L1 loss


class _STFTLoss(nn.Module):
    """Single-resolution STFT loss: returns (spectral_convergence, log_magnitude_L1)."""

    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def _mag(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window,
            return_complex=True,
        )
        return spec.abs().clamp_min(1e-7)

    def forward(self, estimate: torch.Tensor, target: torch.Tensor):
        est_mag = self._mag(estimate)
        ref_mag = self._mag(target)
        # Global Frobenius ratio: one scalar over batch + freq + time (not per-sample).
        sc = torch.norm(ref_mag - est_mag, p="fro") / (torch.norm(ref_mag, p="fro") + 1e-8)
        mag = F.l1_loss(torch.log(est_mag), torch.log(ref_mag))
        return sc, mag


class MultiResSTFTLoss(nn.Module):
    """Sum of single-resolution STFT losses, scaled by W_SC and W_MAG."""

    def __init__(self,
                 fft_sizes=FFT_SIZES, hop_sizes=HOP_SIZES, win_sizes=WIN_SIZES,
                 w_sc: float = W_SC, w_mag: float = W_MAG):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)
        self.resolutions = nn.ModuleList([
            _STFTLoss(n, h, w) for n, h, w in zip(fft_sizes, hop_sizes, win_sizes)
        ])
        self.w_sc, self.w_mag = w_sc, w_mag

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # estimate, target: (B, T). Returns a scalar.
        sc_total, mag_total = 0.0, 0.0
        for stft in self.resolutions:
            sc, mag = stft(estimate, target)
            sc_total = sc_total + sc
            mag_total = mag_total + mag
        n = len(self.resolutions)
        return self.w_sc * (sc_total / n) + self.w_mag * (mag_total / n)
