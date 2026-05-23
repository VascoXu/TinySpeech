"""Conv-TasNet for single-channel speech separation.

  encoder   : strided 1D conv, waveform -> N-dim "frames"     (B, T) -> (B, N, L)
  separator : stacked dilated depth-wise TCN blocks emit one mask per source
  decoder   : overlap-add convT back to time domain           (B, C, N, L) -> (B, C, T)
"""
import torch
import torch.nn as nn


class cLN(nn.Module):
    """Cumulative layer norm (causal): normalize by stats from t' <= t over all channels."""

    def __init__(self, dimension, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(1, dimension, 1))
        self.bias = nn.Parameter(torch.zeros(1, dimension, 1))

    def forward(self, input):
        # input: (B, C, T)
        channel = input.size(1)
        time_step = input.size(2)

        step_sum = input.sum(1)                   # (B, T)
        step_pow_sum = input.pow(2).sum(1)        # (B, T)
        cum_sum = torch.cumsum(step_sum, dim=1)
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)

        entry_cnt = torch.arange(1, time_step + 1, device=input.device, dtype=input.dtype) * channel
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()

        cum_mean = cum_mean.unsqueeze(1)          # (B, 1, T)
        cum_std = cum_std.unsqueeze(1)            # (B, 1, T)

        return (input - cum_mean) / cum_std * self.gain + self.bias


class DepthConv1d(nn.Module):
    """One TCN residual block: 1x1 conv -> depth-wise dilated conv -> 1x1 residual+skip."""

    def __init__(self, input_channel, hidden_channel, kernel, padding,
                 dilation=1, skip=True, causal=False):
        super().__init__()
        self.causal = causal
        self.skip = skip

        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        # Causal: pad only on the left (size = (kernel-1)*dilation), then slice off the right tail
        # so the receptive field stays in the past.
        self.padding = (kernel - 1) * dilation if causal else padding
        self.dconv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel,
                                 dilation=dilation, groups=hidden_channel,
                                 padding=self.padding)
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        if causal:
            self.reg1 = cLN(hidden_channel)
            self.reg2 = cLN(hidden_channel)
        else:
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        if skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        # input: (B, C_in, L)
        output = self.reg1(self.nonlinearity1(self.conv1d(input)))
        if self.causal:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)[:, :, :-self.padding]))
        else:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
        residual = self.res_out(output)
        if self.skip:
            return residual, self.skip_out(output)
        return residual


class TCN(nn.Module):
    """Stacked dilated TCN: `stack` repeats of `layer` blocks with dilation 1,2,4,...,2^(layer-1)."""

    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim,
                 layer, stack, kernel=3, skip=True, causal=False, dilated=True):
        super().__init__()
        # input: (B, N=input_dim, L)
        self.LN = cLN(input_dim) if causal else nn.GroupNorm(1, input_dim, eps=1e-8)
        self.BN = nn.Conv1d(input_dim, BN_dim, 1)

        self.dilated = dilated
        self.skip = skip
        self.receptive_field = 0
        self.TCN = nn.ModuleList()
        for s in range(stack):
            for i in range(layer):
                dilation = 2 ** i if dilated else 1
                padding = 2 ** i if dilated else 1
                self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel,
                                            dilation=dilation, padding=padding,
                                            skip=skip, causal=causal))
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    self.receptive_field += (kernel - 1) * (2 ** i if dilated else 1)

        # output is nn.Sequential — keys are .output.0 (PReLU) / .output.1 (Conv1d); keep as-is.
        self.output = nn.Sequential(nn.PReLU(), nn.Conv1d(BN_dim, output_dim, 1))

    def forward(self, input):
        # input: (B, N, L) -> output: (B, N*C, L)
        output = self.BN(self.LN(input))
        if self.skip:
            skip_connection = 0.
            for block in self.TCN:
                residual, skip = block(output)
                output = output + residual
                skip_connection = skip_connection + skip
            return self.output(skip_connection)
        for block in self.TCN:
            output = output + block(output)
        return self.output(output)


class TasNet(nn.Module):
    # train.py uses: TasNet(num_spk=2, causal=True, sr=16000)
    def __init__(self, enc_dim=512, feature_dim=128, sr=16000, win=2,
                 layer=8, stack=3, kernel=3, num_spk=2, causal=False):
        super().__init__()
        self.num_spk = num_spk
        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.win = int(sr * win / 1000)   # samples per encoder frame
        self.stride = self.win // 2       # 50% overlap
        self.layer = layer
        self.stack = stack
        self.kernel = kernel
        self.causal = causal

        self.encoder = nn.Conv1d(1, enc_dim, self.win, bias=False, stride=self.stride)
        self.TCN = TCN(enc_dim, enc_dim * num_spk, feature_dim, feature_dim * 2,
                       layer, stack, kernel, causal=causal)
        self.receptive_field = self.TCN.receptive_field
        self.decoder = nn.ConvTranspose1d(enc_dim, 1, self.win, bias=False, stride=self.stride)

    def pad_signal(self, input):
        # (B, T) or (B, 1, T) -> (B, 1, T_pad) so encoder convs stride cleanly.
        if input.dim() not in (2, 3):
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size, _, nsample = input.size()
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = torch.zeros(batch_size, 1, rest, device=input.device, dtype=input.dtype)
            input = torch.cat([input, pad], 2)
        pad_aux = torch.zeros(batch_size, 1, self.stride, device=input.device, dtype=input.dtype)
        input = torch.cat([pad_aux, input, pad_aux], 2)
        return input, rest

    def forward(self, input):
        # (B, T) -> (B, C, T) where C = num_spk
        output, rest = self.pad_signal(input)          # (B, 1, T_pad)
        batch_size = output.size(0)

        enc_output = self.encoder(output)              # (B, N, L)
        masks = torch.sigmoid(self.TCN(enc_output)).view(
            batch_size, self.num_spk, self.enc_dim, -1)  # (B, C, N, L)
        masked_output = enc_output.unsqueeze(1) * masks  # (B, C, N, L)

        output = self.decoder(masked_output.view(batch_size * self.num_spk, self.enc_dim, -1))
        output = output[:, :, self.stride:-(rest + self.stride)].contiguous()  # trim padding
        return output.view(batch_size, self.num_spk, -1)  # (B, C, T)
