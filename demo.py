"""Run a trained TinySpeech checkpoint on a .wav and save the denoised output."""
import argparse
from pathlib import Path

import torch
import torchaudio.functional as AF
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from conv_tasnet import TasNet
from dataset import SR, HPF_HZ


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = AudioDecoder(str(args.input), sample_rate=SR).get_all_samples()
    noisy = AF.highpass_biquad(samples.data.mean(dim=0), SR, HPF_HZ)

    model = TasNet(num_spk=1, causal=True, sr=SR).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.checkpoint}  (epoch {ckpt.get('epoch', '?')}, "
          f"val si-sdr {ckpt.get('val_sisdr', float('nan')):+.2f} dB)")

    with torch.no_grad():
        estimate = model(noisy.unsqueeze(0).to(device)).squeeze(1).squeeze(0).cpu()

    peak = estimate.abs().max().item()
    if peak > 0.99:
        estimate = estimate * (0.99 / peak)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    AudioEncoder(estimate.unsqueeze(0), sample_rate=SR).to_file(str(args.output))
    print(f"saved denoised audio -> {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Noisy input .wav")
    p.add_argument("--output", type=Path, required=True, help="Where to save denoised .wav")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    main(p.parse_args())
