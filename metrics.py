"""Scale-aware SDR (scale solved per-sample) for monitoring separation quality."""
import torch

EPS = 1e-8


def calc_sdr_torch(estimation: torch.Tensor, origin: torch.Tensor,
                   mask: torch.Tensor = None) -> torch.Tensor:
    """Per-sample SDR in dB. estimation, origin: (B, T). Returns (B,)."""
    if mask is not None:
        origin = origin * mask
        estimation = estimation * mask

    origin_power = origin.pow(2).sum(1, keepdim=True) + EPS            # (B, 1)
    scale = (origin * estimation).sum(1, keepdim=True) / origin_power  # (B, 1)

    est_true = scale * origin            # projection onto target
    est_res = estimation - est_true      # residual

    true_power = est_true.pow(2).sum(1)
    res_power = est_res.pow(2).sum(1)
    return 10 * torch.log10(true_power) - 10 * torch.log10(res_power)
