"""L1 waveform + multi-resolution STFT loss (ClearBuds / Parallel WaveGAN recipe)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _STFTLoss(nn.Module):
    """Single-resolution STFT loss: spectral convergence + log-magnitude L1."""

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
        return spec.abs()

    def forward(self, estimate: torch.Tensor, target: torch.Tensor):
        est_mag = self._mag(estimate)
        ref_mag = self._mag(target)
        sc = torch.norm(ref_mag - est_mag) / (torch.norm(ref_mag) + 1e-8)
        mag = F.l1_loss(torch.log(est_mag + 1e-7), torch.log(ref_mag + 1e-7))
        return sc, mag


class L1MultiResSTFTLoss(nn.Module):
    """L1 waveform loss + multi-resolution STFT loss.

    Default weights match the ClearBuds recipe: 5*L1 + 0.2*SC + 0.2*log-mag.
    """

    def __init__(self,
                 fft_sizes=(512, 1024, 2048),
                 hop_sizes=(50, 120, 240),
                 win_sizes=(240, 600, 1200),
                 w_l1: float = 5.0,
                 w_sc: float = 0.2,
                 w_mag: float = 0.2):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)
        self.resolutions = nn.ModuleList([
            _STFTLoss(n, h, w) for n, h, w in zip(fft_sizes, hop_sizes, win_sizes)
        ])
        self.w_l1, self.w_sc, self.w_mag = w_l1, w_sc, w_mag

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sc_sum, mag_sum = 0.0, 0.0
        for stft in self.resolutions:
            sc, mag = stft(estimate, target)
            sc_sum = sc_sum + sc
            mag_sum = mag_sum + mag
        n = len(self.resolutions)
        l1 = F.l1_loss(estimate, target)
        return self.w_l1 * l1 + self.w_sc * (sc_sum / n) + self.w_mag * (mag_sum / n)
