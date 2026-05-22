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

## Train with reverb (fine-tune)

Convolves target + babble through random rooms (`pyroomacoustics`). Resume from the dry baseline:

```bash
# 1. Bake a bank of 2000 random-room RIRs (one-time, ~10 s)
python rir.py --n-rirs 2000 --out-pt datasets/rir_bank.pt

# 2. Render a matching reverberant val set
python prepare.py \
  --speech-root datasets/VCTK-Corpus-0.92/wav48_silence_trimmed \
  --wham-root datasets/wham_noise \
  --rir-bank datasets/rir_bank.pt \
  --out-pt datasets/val_vctk_reverb.pt

# 3. Fine-tune from the dry checkpoint
python train.py \
  --speech-root datasets/VCTK-Corpus-0.92/wav48_silence_trimmed \
  --wham-root datasets/wham_noise \
  --val-pt datasets/val_vctk_reverb.pt \
  --rir-bank datasets/rir_bank.pt \
  --resume checkpoints/best.pt \
  --ckpt-dir checkpoints_reverb \
  --epochs 20
```

## Demo

```bash
# Denoise a real .wav
python demo.py --input some.wav --out-dir out/ --checkpoint checkpoints/best.pt

# Listen to the model on random training-distribution mixes
python sample.py --speech-root datasets/VCTK-Corpus-0.92/wav48_silence_trimmed \
                 --wham-root datasets/wham_noise --out-dir samples/
```
