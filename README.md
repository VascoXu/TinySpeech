# TinySpeech

Causal Conv-TasNet for single-channel babble denoising at 16 kHz.

## Setup

Expects [VCTK-0.92](https://datashare.ed.ac.uk/handle/10283/3443) speech and [WHAM!](http://wham.whisper.ai/) noise under `datasets/`.

## Train

```bash
# 1. Render a fixed val set (one-time)
python prepare.py \
  --speech-root datasets/VCTK-Corpus-0.92/wav48_silence_trimmed \
  --wham-root datasets/wham_noise

# 2. Train
python train.py \
  --speech-root datasets/VCTK-Corpus-0.92/wav48_silence_trimmed \
  --wham-root datasets/wham_noise \
  --val-pt datasets/val_vctk.pt \
  --epochs 100
```

Resume: add `--resume checkpoints/last.pt`.

## Demo

```bash
# Denoise a real .wav
python demo.py --input some.wav --out-dir out/ --checkpoint checkpoints/best.pt

# Listen to the model on random training-distribution mixes
python sample.py --speech-root datasets/VCTK-Corpus-0.92/wav48_silence_trimmed \
                 --wham-root datasets/wham_noise --out-dir samples/
```
