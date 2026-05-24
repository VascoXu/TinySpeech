# TinySpeech

Causal Conv-TasNet post-filter on top of MVDR beamforming, at 16 kHz. The target use case is on-device AR-glasses speech enhancement for hard-of-hearing users.

## Setup

Expects [VCTK-0.92](https://datashare.ed.ac.uk/handle/10283/3443), [LibriSpeech](https://www.openslr.org/12/) (train-clean-360, test-clean), and [WHAM!](http://wham.whisper.ai/) noise under `datasets/`. Dataset name → path mapping lives in [`dataset.py:DATASETS`](dataset.py).

## Repo layout

Library modules (importable, no `__main__`):

| File | Contents |
|---|---|
| [`dataset.py`](dataset.py) | `SpatialAudioDataset` (live training mixtures), `ProcessedSpeechDataset` (frozen val/test), `DATASETS`, audio helpers |
| [`beamforming.py`](beamforming.py) | Room sampling, mic-array placement, RIR rendering, MVDR weights + beamforming, peak normalize, `fft_convolve` |
| [`model.py`](model.py) | `TasNet` — causal Conv-TasNet (N=256, L=40, B=256, H=512, X=8, R=4, C=1) |
| [`losses.py`](losses.py) | `MultiResSTFTLoss` — spectral convergence + log-mag at 3 resolutions |
| [`metrics.py`](metrics.py) | `calc_sdr_torch` (SI-SDR) |

Scripts (one entry point each):

| File | Purpose |
|---|---|
| [`render_bank.py`](render_bank.py) | Pre-render multi-mic RIR bank (FG + BG rooms, polar grid of sources) |
| [`render_val.py`](render_val.py) | Pre-render frozen val/test set from the live distribution |
| [`preview.py`](preview.py) | Listen to a few scenes and the beamformer output — sanity-check the data recipe |
| [`train.py`](train.py) | Training loop |
| [`eval.py`](eval.py) | Eval on a frozen test set: SI-SDR, SI-SDRi, PESQ, STOI |
| [`demo.py`](demo.py) | Run a checkpoint on a real `.wav` |
| [`cascade.py`](cascade.py) | Conv-TasNet → ClearBuds-UNet TF-mask cascade on a real `.wav` |
| [`spectrogram.py`](spectrogram.py) | Side-by-side log-mag spectrograms of (noisy, target, estimate) |

## Train

Training input is MVDR-beamformed mono audio. Each example is a live multi-mic scene (2D polygon room, 4-mic circular array at the origin, target voice at origin, K~U[1,3] babble talkers + 1 env source in a separate larger room) → MVDR with all-ones steering + oracle noise covariance → mono. Loss is 5·L1 + 0.1·SC + 0.1·log-mag (waveform L1 plus multi-resolution STFT).

```bash
# 1. Render the multi-mic RIR bank (one-time)
python render_bank.py --n-scenes 1000 --out-pt datasets/rir_bank.pt

# 2. Render a fixed val set (matches training distribution; same --target-type)
python render_val.py \
  --dataset vctk \
  --wham-root datasets/wham_noise \
  --beam-bank datasets/rir_bank.pt \
  --out-pt datasets/val.pt \
  --target-type reverberant

# 3. Train
python train.py \
  --dataset vctk librispeech \
  --wham-root datasets/wham_noise \
  --beam-bank datasets/rir_bank.pt \
  --val-pt datasets/val.pt \
  --target-type reverberant \
  --epochs 30
```

`--target-type reverberant` predicts the target as seen at the array (post-MVDR) — time-aligned with the beamformed input, denoise-only task, preferred for hearing aids where early reflections aid intelligibility. `--target-type anechoic` predicts the dry voice (denoise + dereverb in one shot). Val set must match training.

Resume: add `--resume checkpoints/last.pt`.

## Preview the data recipe

Renders a few scenes end-to-end and saves each component as a wav so you can listen to what the model is being asked to denoise.

```bash
python preview.py --dataset vctk librispeech --wham-root datasets/wham_noise
```

Outputs `preview_beam/example_NN/` with: dry anechoic target (label), mic 0 raw, beamformed mix, and the beamformed target / babble / env decompositions. Prints per-scene SI-SDR(mic0 vs beam) so you can verify the beamformer is helping.

## Multi-mic RIR bank

Each scene contains two paired rooms — same circular mic array (centered at origin), different acoustics:

- **FG room** (walls ±[15, 20] m): one target RIR at origin + a polar grid of babble RIRs at (angle, radius).
- **BG room** (walls ±[20, 40] m): a polar grid of env-noise RIRs at (angle, radius).

Defaults: 16 FG angles × 5 FG radii (babble), 8 BG angles × 3 BG radii (env), 4 mics, 1.5 s RIRs → ~25 MB/scene. Bank format documented in [`render_bank.py`](render_bank.py).

## Evaluate

```bash
python eval.py --checkpoint checkpoints/best.pt --test-pt datasets/val.pt
```

Reports SI-SDR, SI-SDRi, PESQ, STOI (mean / median / std / min / max).

## Demo

```bash
python demo.py --input some.wav --out-dir out/ --checkpoint checkpoints/best.pt
```

Multi-channel input is averaged (delay-and-sum at boresight); for true MVDR on a real recording you'd need known mic positions and an oracle/estimate of the noise covariance — out of scope for the demo.

## Cascade (Conv-TasNet → UNet TF-mask)

Reference baseline: feed the Conv-TasNet output through ClearBuds' pretrained spectrogram UNet, which predicts a TF-bin "voice vs artifact" mask. Bins flagged as artifact are zeroed in the linear STFT before ISTFT. Substantially reduces Conv-TasNet's residual smearing and musical-noise artifacts.

```bash
python cascade.py --input some.wav --out-dir out/
```

Writes both `baseline.wav` (Conv-TasNet only) and `cascade.wav` (+ UNet gate) for A/B. Tunable `--cutoff` controls how aggressive the gate is (default 0.003; lower = gentler, higher = cleaner silence but risks chopping consonants). Requires the ClearBuds reference checkout at `clearbuds/clearbuds_spectrogram/` for the UNet checkpoint.
