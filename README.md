# TinySpeech

Causal Conv-TasNet for single-channel babble denoising at 16 kHz.

## Setup

Expects [VCTK-0.92](https://datashare.ed.ac.uk/handle/10283/3443) and [LibriSpeech](https://www.openslr.org/12/) (train-clean-360, test-clean) speech and [WHAM!](http://wham.whisper.ai/) noise under `datasets/`.

## Train

```bash
# 1. Render a fixed val set (one-time)
python prepare.py \
  --speech-root datasets/VCTK-Corpus-0.92/wav48_silence_trimmed \
  --wham-root datasets/wham_noise


# 2. Train (multi-corpus: VCTK + LibriSpeech train-clean-360 for OOD robustness)
python train.py \
  --dataset vctk librispeech \
  --wham-root datasets/wham_noise \
  --val-pt datasets/val_vctk.pt \
  --epochs 100
```

Dataset name → path mapping lives in `dataset.py:DATASETS`. Add new datasets there.

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
  --dataset vctk librispeech \
  --wham-root datasets/wham_noise \
  --val-pt datasets/val_vctk_reverb.pt \
  --rir-bank datasets/rir_bank.pt \
  --resume checkpoints/best.pt \
  --ckpt-dir checkpoints_reverb \
  --epochs 20
```

## OOD test set

For cross-corpus speaker generalization eval — both target and babble are unseen LibriSpeech speakers.

```bash
python prepare.py \
  --speech-root datasets/LibriSpeech/test-clean \
  --wham-root datasets/wham_noise \
  --out-pt datasets/test_libri.pt \
  --n-examples 1000 \
  --seed 1
```

## Evaluate

Run a checkpoint against a rendered test set. Reports SI-SDR, SI-SDRi, PESQ, STOI.

```bash
python eval.py \
  --checkpoint checkpoints/best.pt \
  --test-pt datasets/test_libri.pt
```

## Beamformer preview

Renders multi-mic 3D-room scenes (target + co-directional babble + env noise), applies delay-and-sum beamforming, and saves per-component wavs for listening. Used to validate geometry + SIR/SNR conventions for the eventual beamform + post-filter training pipeline (where the post-filter sees beamformed audio, not raw mixtures).

```bash
python beam.py --dataset vctk librispeech --wham-root datasets/wham_noise
```

Outputs to `preview_beam/example_NN/` with 6 wavs per example: dry anechoic target (label), mic 0 raw, beamformed mix, plus the beamformed target / babble / env decompositions. Prints per-scene SI-SDR(mic0 vs beam) so you can verify the beamformer is helping.

## Demo

```bash
# Denoise a real .wav
python demo.py --input some.wav --out-dir out/ --checkpoint checkpoints/best.pt

# Listen to the model on random training-distribution mixes
python sample.py --speech-root datasets/VCTK-Corpus-0.92/wav48_silence_trimmed \
                 --wham-root datasets/wham_noise --out-dir samples/
```
