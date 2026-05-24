"""Multi-resolution STFT loss — ClearBuds recipe.

Reference: clearbuds_waveform/src/stft_loss.py + solver.py:205 ::
    loss = 5 * F.l1_loss(gt, estimate) + 0.2 * sc + 0.2 * mag
where (sc, mag) come from MultiResolutionSTFTLoss(factor_sc=0.5, factor_mag=0.5), i.e.
they already include a 0.5 prefactor. Effective combined weights are 5·L1 + 0.1·SC + 0.1·log-mag.

L1 is computed externally in train.py so it can be applied across both output streams (target +
noise), matching ClearBuds' multi-mic L1 broadcast at solver.py:201. The multi-res STFT loss
is applied to the target stream only (matching solver.py:203).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Three STFT resolutions: short-fast, medium, long-fine. Catches transients and harmonics together.
FFT_SIZES = (1024, 2048, 512)
HOP_SIZES = (120, 240, 50)
WIN_SIZES = (600, 1200, 240)

W_SC = 0.1   # 0.2 outer × 0.5 inner (factor_sc) in ClearBuds
W_MAG = 0.1  # 0.2 outer × 0.5 inner (factor_mag) in ClearBuds


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
        # ClearBuds: global Frobenius ratio (one scalar over batch + freq + time)
        sc = torch.norm(ref_mag - est_mag, p="fro") / (torch.norm(ref_mag, p="fro") + 1e-8)
        mag = F.l1_loss(torch.log(est_mag), torch.log(ref_mag))
        return sc, mag


class MultiResSTFTLoss(nn.Module):
    """Sum of single-resolution STFT losses, scaled by ClearBuds weights (W_SC, W_MAG)."""

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
