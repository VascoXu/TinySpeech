# TinySpeech

Causal Conv-TasNet for single-channel babble denoising at 16 kHz.

## Setup

Expects [VCTK-0.92](https://datashare.ed.ac.uk/handle/10283/3443) and [LibriSpeech](https://www.openslr.org/12/) (train-clean-360, test-clean) speech and [WHAM!](http://wham.whisper.ai/) noise under `datasets/`.

## Train

Training inputs are *beamformed* single-channel signals: ClearBuds-style multi-mic 2D-polygon scenes from a pre-rendered RIR bank → MVDR (oracle noise covariance, all-ones steering for target at array origin) → mono. The post-filter is causal Conv-TasNet trained with `L1MultiResSTFTLoss`.

```bash
# 1. Render the multi-mic RIR bank (one-time; see § Multi-mic RIR bank)
python beam_bank.py --n-scenes 100 --out-pt datasets/rir_bank_beam.pt

# 2. Render a fixed beam-rendered val set (one-time, matches training distribution)
python beam_prepare.py \
  --dataset vctk \
  --wham-root datasets/wham_noise \
  --beam-bank datasets/rir_bank_beam.pt \
  --out-pt datasets/val_beam.pt \
  --target-type reverberant

# 3. Train
python train.py \
  --dataset vctk librispeech \
  --wham-root datasets/wham_noise \
  --beam-bank datasets/rir_bank_beam.pt \
  --val-pt datasets/val_beam.pt \
  --target-type reverberant \
  --epochs 100
```

`--target-type` controls what the post-filter is asked to predict. `reverberant` (the target as seen at the array, after MVDR) is time-aligned with the beamformed input and turns the task into denoise-only — preferred for hearing-aid use where early reflections aid intelligibility. `anechoic` is the dry voice and asks the post-filter to denoise + dereverb in one shot. The val set must use the same `--target-type` as training.

Dataset name → path mapping lives in `dataset.py:DATASETS`. Add new datasets there.

Resume: add `--resume checkpoints/last.pt`.

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

Renders ClearBuds-style scenes (2D polygon room, 4-mic circular array centered at origin, target at origin, K=3 babble talkers at random angles + radii, env noise in a separate larger room), applies MVDR (all-ones steering, oracle noise covariance), and saves per-component wavs for listening.

```bash
python beam.py --dataset vctk librispeech --wham-root datasets/wham_noise
```

Outputs to `preview_beam/example_NN/` with 6 wavs per example: dry anechoic target (label), mic 0 raw, beamformed mix, plus the beamformed target / babble / env decompositions. Prints per-scene SI-SDR(mic0 vs beam) so you can verify the beamformer is helping.

## Multi-mic RIR bank

Pre-render the RIRs that `SpatialAudioDataset` looks up at training time. Each scene contains two paired rooms — same circular mic array (centered at origin), different acoustics:

- **FG room** (walls ±[15, 20] m): one target RIR at origin + a polar grid of babble RIRs at (angle, radius).
- **BG room** (walls ±[20, 40] m): a polar grid of env-noise RIRs at (angle, radius).

```bash
python beam_bank.py --n-scenes 100 --out-pt datasets/rir_bank_beam.pt
```

Defaults: 16 FG angles × 5 FG radii (babble), 8 BG angles × 3 BG radii (env), 4 mics, 1.5 s RIRs → ~25 MB/scene. Bank format documented in `beam_bank.py`.

## Demo

```bash
# Denoise a real .wav
python demo.py --input some.wav --out-dir out/ --checkpoint checkpoints/best.pt

# Listen to the model on random training-distribution mixes
python sample.py --speech-root datasets/VCTK-Corpus-0.92/wav48_silence_trimmed \
                 --wham-root datasets/wham_noise --out-dir samples/
```
