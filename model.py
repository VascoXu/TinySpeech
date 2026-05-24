"""Causal Conv-TasNet — single-channel post-filter for the MVDR-beamformed front end.

Mirrors the ClearBuds reference (clearbuds_waveform/src/conv_tasnet.py) exactly:
  encoder    : strided 1D-conv with stride=L (no overlap), ReLU.   (B, 1, T) -> (B, N, K)
  separator  : channelwise LayerNorm + 1×1 bottleneck + stacked dilated TCN blocks
               + 1×1 mask projection + ReLU mask nonlinearity.     (B, N, K) -> (B, C, N, K)
  decoder    : per-frame linear basis (N -> L), then concatenate.  (B, N, K), mask -> (B, C, K·L)

The model emits C streams (default C=1: just the target voice). C=2 is supported but rarely
useful for our beamformed-input setup — see the README and conversation history for rationale.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelwiseLayerNorm(nn.Module):
    """Per-frame normalization across channels. Causal by construction (no time mixing)."""

    def __init__(self, channel_size: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel_size, 1))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (B, C, K)
        mean = y.mean(dim=1, keepdim=True)
        var = ((y - mean) ** 2).mean(dim=1, keepdim=True)
        return self.gamma * (y - mean) / (var + self.eps).sqrt() + self.beta


class DepthwiseSeparableConv(nn.Module):
    """Depthwise dilated conv -> ReLU -> cLN -> pointwise 1×1. Causal via left-pad + chomp."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, causal: bool = True):
        super().__init__()
        self.causal = causal
        self.padding = (kernel - 1) * dilation if causal else (kernel - 1) * dilation // 2
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel, dilation=dilation,
                                   padding=self.padding, groups=in_ch, bias=False)
        self.relu = nn.ReLU()
        self.norm = ChannelwiseLayerNorm(in_ch)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        if self.causal:
            out = out[:, :, :-self.padding]
        return self.pointwise(self.norm(self.relu(out)))


class TemporalBlock(nn.Module):
    """1×1 (B->H) -> ReLU -> cLN -> DepthwiseSeparableConv (H->B) with residual add."""

    def __init__(self, B: int, H: int, kernel: int, dilation: int, causal: bool = True):
        super().__init__()
        self.conv1x1 = nn.Conv1d(B, H, 1, bias=False)
        self.relu = nn.ReLU()
        self.norm = ChannelwiseLayerNorm(H)
        self.dsconv = DepthwiseSeparableConv(H, B, kernel, dilation, causal=causal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dsconv(self.norm(self.relu(self.conv1x1(x))))
        return out + x[:, :, -out.size(-1):]


class TemporalConvNet(nn.Module):
    """cLN -> 1×1 bottleneck -> R repeats of X dilated blocks -> 1×1 mask projection -> ReLU."""

    def __init__(self, N: int, B: int, H: int, P: int, X: int, R: int, C: int,
                 causal: bool = True):
        super().__init__()
        self.C = C
        self.layer_norm = ChannelwiseLayerNorm(N)
        self.bottleneck = nn.Conv1d(N, B, 1, bias=False)
        self.blocks = nn.ModuleList()
        self.receptive_field = 0
        for r in range(R):
            for x in range(X):
                dilation = 2 ** x
                self.blocks.append(TemporalBlock(B, H, P, dilation, causal=causal))
                if r == 0 and x == 0:
                    self.receptive_field += P
                else:
                    self.receptive_field += (P - 1) * dilation
        self.mask_conv = nn.Conv1d(B, C * N, 1, bias=False)

    def forward(self, mixture_w: torch.Tensor) -> torch.Tensor:
        # mixture_w: (M, N, K) -> (M, C, N, K)
        M, _, K = mixture_w.size()
        x = self.bottleneck(self.layer_norm(mixture_w))
        for block in self.blocks:
            x = block(x)
        score = self.mask_conv(x)                       # (M, C·N, K)
        return F.relu(score).view(M, self.C, -1, K)     # (M, C, N, K)


class Encoder(nn.Module):
    """Non-overlapping windowed Conv1d with ReLU. (B, 1, T) -> (B, N, K=T/L)."""

    def __init__(self, L: int, N: int, input_channels: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(input_channels, N, kernel_size=L, stride=L, bias=False)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(mixture))


class Decoder(nn.Module):
    """Per-frame basis Linear(N -> L), then concatenate L-sample windows. Inverts the encoder."""

    def __init__(self, N: int, L: int):
        super().__init__()
        self.basis = nn.Linear(N, L, bias=False)

    def forward(self, mixture_w: torch.Tensor, est_mask: torch.Tensor) -> torch.Tensor:
        # mixture_w: (M, N, K)   est_mask: (M, C, N, K)
        M, C = est_mask.size(0), est_mask.size(1)
        source_w = mixture_w.unsqueeze(1) * est_mask                # (M, C, N, K)
        source_w = source_w.transpose(2, 3)                         # (M, C, K, N)
        est_source = self.basis(source_w)                           # (M, C, K, L)
        return est_source.reshape(M, C, -1)                         # (M, C, K·L)


class TasNet(nn.Module):
    """Causal Conv-TasNet, ClearBuds-shape defaults (N=256, L=40, B=256, H=512, P=3, X=8, R=4); C=1 by default."""

    def __init__(self, N: int = 256, L: int = 40, B: int = 256, H: int = 512,
                 P: int = 3, X: int = 8, R: int = 4, C: int = 1, causal: bool = True,
                 sr: int = 16000):
        super().__init__()
        self.N, self.L, self.B, self.H = N, L, B, H
        self.P, self.X, self.R, self.C = P, X, R, C
        self.causal = causal
        self.sr = sr
        self.stride = L  # exposed for train.py diagnostics; ClearBuds uses non-overlapping windows

        self.encoder = Encoder(L, N)
        self.separator = TemporalConvNet(N, B, H, P, X, R, C, causal=causal)
        self.decoder = Decoder(N, L)
        self.receptive_field = self.separator.receptive_field

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def _pad(self, x: torch.Tensor):
        # (B, T) or (B, 1, T) -> (B, 1, T_pad) where T_pad is a multiple of L.
        if x.dim() == 2:
            x = x.unsqueeze(1)
        T = x.size(-1)
        rest = (-T) % self.L
        if rest > 0:
            x = F.pad(x, (0, rest))
        return x, T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T) or (B, 1, T) -> (B, C, T)
        x, T = self._pad(x)
        enc = self.encoder(x)                  # (B, N, K)
        mask = self.separator(enc)             # (B, C, N, K)
        out = self.decoder(enc, mask)          # (B, C, K·L)
        return out[:, :, :T]
