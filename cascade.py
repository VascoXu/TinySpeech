"""Run Conv-TasNet → ClearBuds-UNet TF-mask cascade on a wav.

The UNet's binary TF-bin mask is applied to the linear STFT of the Conv-TasNet output:
bins it flags as 'not target voice' are zeroed before ISTFT. Cleans up most of
Conv-TasNet's residual smearing/musical-noise artifacts. Saves both the Conv-TasNet
baseline and the cascade output for A/B comparison.

Sample-rate note: the ClearBuds UNet was trained at 15625 Hz, so we resample for the
UNet stage and back to 16 kHz at the end.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torchaudio.functional as AF
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

# UNet checkpoint is a pickled nn.Module → its source has to be importable.
sys.path.insert(0, str(Path(__file__).parent / "clearbuds/clearbuds_spectrogram"))

from dataset import SR
from model import TasNet

UNET_SR  = 15625     # ClearBuds UNet inference rate
N_FFT    = 1024
HOP      = 256       # librosa.stft default for n_fft=1024
N_MELS   = 128
FMIN     = 20
TIME_DIM = 64        # causal context frames per UNet pass
STEP     = 2         # frames emitted per UNet pass
AMIN     = 1e-10     # power_to_db floor


def power_to_db(power: torch.Tensor) -> torch.Tensor:
    """librosa.power_to_db(power, ref=1.0, amin=1e-10, top_db=80)."""
    db = 10.0 * torch.log10(power.clamp_min(AMIN))
    return db.clamp_min(db.max() - 80.0)


def run_conv_tasnet(checkpoint: Path, wav: torch.Tensor, device: torch.device) -> torch.Tensor:
    """(T,) at SR → (T,) target estimate at SR."""
    model = TasNet(causal=True, sr=SR).to(device).eval()
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"  Conv-TasNet: epoch {ckpt.get('epoch', '?')}, "
          f"val si-sdr {ckpt.get('val_sisdr', float('nan')):+.2f} dB")
    with torch.no_grad():
        return model(wav.unsqueeze(0))[0, 0].cpu()


def apply_unet_mask(checkpoint: Path, wav: torch.Tensor, cutoff: float, device: torch.device) -> torch.Tensor:
    """Gate `wav` (at SR) through the UNet's TF-bin mask. Returns gated wav at SR."""
    unet = torch.load(checkpoint, map_location=device, weights_only=False).to(device).eval()
    print(f"  UNet: {sum(p.numel() for p in unet.parameters())/1e6:.2f}M params, cutoff={cutoff}")

    wav_unet = AF.resample(wav, orig_freq=SR, new_freq=UNET_SR)

    window = torch.hann_window(N_FFT)
    linear_stft = torch.stft(wav_unet, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                             window=window, center=True, return_complex=True)   # (F, T)
    mel_fb = AF.melscale_fbanks(n_freqs=N_FFT // 2 + 1, f_min=float(FMIN),
                                f_max=UNET_SR / 2.0, n_mels=N_MELS, sample_rate=UNET_SR,
                                norm="slaney", mel_scale="slaney")               # (F, M)
    mel_db = power_to_db(mel_fb.T @ linear_stft.abs() ** 2)                      # (M, T)

    # Stream UNet in causal 64-frame chunks, emit STEP frames per pass.
    mel_padded = torch.nn.functional.pad(mel_db, (TIME_DIM - STEP, 0))
    chunks, idx = [], 0
    with torch.no_grad():
        while idx + TIME_DIM <= mel_padded.shape[1]:
            x = mel_padded[:, idx:idx + TIME_DIM]
            x = (x - x.mean()) / (x.std() + 1e-8)
            out = unet(x.unsqueeze(0).unsqueeze(0).to(device))
            chunks.append(out[0, 0, :, -STEP:].cpu())
            idx += STEP
    mask_mel = torch.cat(chunks, dim=-1)

    # Mel mask → linear mask via filter-bank transpose, threshold, apply to STFT.
    T = min(mask_mel.shape[-1], linear_stft.shape[-1])
    linear_mask = (mel_fb @ mask_mel[:, :T]) > cutoff
    gated_stft = torch.where(linear_mask, linear_stft[:, :T], torch.zeros_like(linear_stft[:, :T]))
    print(f"  TF bins kept: {linear_mask.float().mean().item()*100:.1f}%")

    gated_unet = torch.istft(gated_stft, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                             window=window, center=True)
    return AF.resample(gated_unet, orig_freq=UNET_SR, new_freq=SR)


def save_wav(path: Path, x: torch.Tensor) -> None:
    peak = max(x.abs().max().item(), 1e-9)
    norm = 0.99 / peak if peak > 0.99 else 1.0
    AudioEncoder((x * norm).unsqueeze(0), sample_rate=SR).to_file(str(path))


def main(args):
    device = torch.device("cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    samples = AudioDecoder(str(args.input), sample_rate=SR).get_all_samples().data
    noisy = samples[0] if samples.shape[0] == 1 else samples.mean(dim=0)
    print(f"input: {samples.shape[0]} ch, {noisy.numel() / SR:.1f}s")

    t0 = time.perf_counter()
    estimate = run_conv_tasnet(args.checkpoint, noisy, device)
    gated = apply_unet_mask(args.unet_checkpoint, estimate, args.cutoff, device)
    print(f"  cascade forward: {(time.perf_counter() - t0) * 1000:.0f} ms")

    save_wav(args.out_dir / "baseline.wav", estimate)
    save_wav(args.out_dir / "cascade.wav",  gated)
    print(f"saved baseline + cascade -> {args.out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Input .wav (any sample rate)")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Writes baseline.wav (Conv-TasNet only) and cascade.wav (+ UNet gate)")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"),
                   help="Conv-TasNet checkpoint")
    p.add_argument("--unet-checkpoint", type=Path,
                   default=Path("clearbuds/clearbuds_spectrogram/checkpoints/model_causal.pt"),
                   help="ClearBuds UNet checkpoint (pickled nn.Module)")
    p.add_argument("--cutoff", type=float, default=0.003,
                   help="Binary mask threshold on the mel→linear-upsampled mask")
    main(p.parse_args())
