"""L1 waveform + multi-resolution STFT loss, with optional PIT wrapper for 2-source training."""
import torch
import torch.nn as nn

# Three STFT resolutions: short-fast, medium, long-fine. Catches transients and harmonics together.
FFT_SIZES = (512, 1024, 2048)
HOP_SIZES = (50, 120, 240)
WIN_SIZES = (240, 600, 1200)

# ClearBuds recipe: heavy L1 keeps phase, light SC + log-mag shape the spectrum.
W_L1, W_SC, W_MAG = 5.0, 0.2, 0.2


class _STFTLoss(nn.Module):
    """Single-resolution STFT loss returning per-sample (B,) spectral convergence + log-mag L1."""

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
        # (B, T) -> magnitude spectrograms (B, F, T_frames)
        est_mag = self._mag(estimate)
        ref_mag = self._mag(target)
        diff_norm = ((ref_mag - est_mag) ** 2).sum(dim=(-2, -1)).sqrt()
        ref_norm = (ref_mag ** 2).sum(dim=(-2, -1)).sqrt() + 1e-8
        sc = diff_norm / ref_norm
        mag = (torch.log(est_mag + 1e-7) - torch.log(ref_mag + 1e-7)).abs().mean(dim=(-2, -1))
        return sc, mag


class L1MultiResSTFTLoss(nn.Module):
    """L1 waveform + multi-resolution STFT loss. ClearBuds-style weights (see W_L1/W_SC/W_MAG).

    reduction='mean' returns a scalar; reduction='none' returns per-sample (B,) — required when
    wrapping in PITLoss so each example's permutation can be picked independently.
    """

    def __init__(self,
                 fft_sizes=FFT_SIZES,
                 hop_sizes=HOP_SIZES,
                 win_sizes=WIN_SIZES,
                 w_l1: float = W_L1,
                 w_sc: float = W_SC,
                 w_mag: float = W_MAG,
                 reduction: str = "mean"):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)
        assert reduction in ("mean", "none")
        self.resolutions = nn.ModuleList([
            _STFTLoss(n, h, w) for n, h, w in zip(fft_sizes, hop_sizes, win_sizes)
        ])
        self.w_l1, self.w_sc, self.w_mag = w_l1, w_sc, w_mag
        self.reduction = reduction

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # estimate, target: (B, T) -> per-sample loss of shape (B,)
        l1 = (estimate - target).abs().mean(dim=-1)
        sc_sum, mag_sum = 0.0, 0.0
        for stft in self.resolutions:
            sc, mag = stft(estimate, target)
            sc_sum = sc_sum + sc
            mag_sum = mag_sum + mag
        n = len(self.resolutions)
        per_sample = self.w_l1 * l1 + self.w_sc * (sc_sum / n) + self.w_mag * (mag_sum / n)
        return per_sample if self.reduction == "none" else per_sample.mean()


class PITLoss(nn.Module):
    """Permutation-invariant training wrapper for 2-source losses.

    For each sample in (B, 2, T) estimates and sources, evaluates both source-to-estimate
    assignments under base_loss (expected to return shape (B,)), picks the lower one per
    sample, and averages across the batch.
    """

    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, estimates: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
        # estimates, sources: (B, 2, T)
        e0, e1 = estimates[:, 0], estimates[:, 1]
        s0, s1 = sources[:, 0], sources[:, 1]
        a = self.base_loss(e0, s0) + self.base_loss(e1, s1)
        b = self.base_loss(e0, s1) + self.base_loss(e1, s0)
        return torch.min(a, b).mean()
