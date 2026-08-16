# Semiconductor Image Restoration - KLA AI Hackathon

Restoration of degraded semiconductor inspection images: joint speckle/Gaussian
denoising and 2x super-resolution (128->256 and 256->512) using a modified
NAFNet. Ships a standalone evaluation script, ONNX export for benchmarking,
and per-pixel uncertainty estimation.

## Problem

Microscope inspection images are degraded by multiplicative speckle noise
(which can push pixel values outside the valid range), additive Gaussian
noise, and spatial-resolution reduction by downsampling. The task is to
recover the clean, full-resolution image from the degraded low-resolution
input, generalize to out-of-distribution image sources, and run fast enough
for end-to-end benchmarking on an H100.

Training data: 3,200 paired samples (GT 256x256, NoisyLR 128x128, float32
`.npy`, filenames matched).

## Results

Held-out validation split (320 images, same deterministic split as training,
seed 42), produced by `scripts/compute_metrics.py`:

| Method | PSNR (dB) | SSIM | LPIPS (lower is better) |
|----------------------|-----------|--------|---------|
| Bicubic upsampling | 23.36 | 0.5572 | 0.4309 |
| DiagNAFNet (ours) | 27.96 | 0.7651 | 0.2825 |

The model improves PSNR by +4.6 dB and LPIPS by 0.15 over the bicubic
baseline. Side-by-side comparisons (input / bicubic / model / ground truth)
are in `results/examples/`.

Checkpoint: `weights/best.pt` (epoch 82, val loss 0.0659, 65.3M parameters).

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements_freeze.txt` contains the full `pip freeze` of the Kaggle
environment the submitted model was trained in.

## Data layout

```
data_root/
    GT/         clean high-res images (.npy, float32, 0-1)
    NoisyLR/    noisy low-res images (.npy, values may exceed [0,1] due to speckle)
```

A deterministic 90/10 train/val split is created automatically.

## Training

```bash
python scripts/train.py --data_root <path> --output_dir weights
```

Configuration lives in `configs/train_config.yaml`. Key settings:

- `model_type: diag_nafnet` - NAFNet backbone + degradation encoder + uncertainty head
- `synth_ratio: 0.25` - mixes procedurally generated OOD pairs into training
- `ema_decay: 0.999` - exponential moving average of weights; `best.pt` stores
  whichever of raw/EMA weights validates better
- Loss: L1 + 0.1 VGG-perceptual + 0.05 SSIM + 0.1 Sobel-edge + 0.05 heteroscedastic NLL

Resume with `--resume weights/latest.pt`. Optional wandb logging via `--wandb`.

## Evaluation

Standalone script (this is the benchmarking entry point):

```bash
python scripts/evaluate.py \
    --input_dir <dir with degraded .npy files> \
    --output_dir outputs \
    --weights weights/best.pt
```

Options:

- `--tile_size 256 --overlap 32` - tiled inference for large inputs
- `--batch_size 8` - batched fast path (default) with threaded image writing
- `--fp16` - half-precision inference on GPU
- `--tta` - 8-view geometric self-ensemble (better quality, 8x compute)
- `--weights2 <big_model.pt>` - uncertainty-gated cascade: a small model
  screens every image, hard ones escalate to the big model
- `--save_uncertainty` - writes JET heatmaps of per-pixel predicted error

Quality metrics against ground truth (validation split):

```bash
python scripts/compute_metrics.py --data_root <path> --weights weights/best.pt
```

## Model

`models/diag_nafnet.py` - DiagNAFNet, a NAFNet (Chen et al., ECCV 2022)
backbone with three additions:

1. **Degradation encoder.** A small CNN inspects the input and estimates
   degradation severity; FiLM layers condition the restorer accordingly, so
   behavior adapts per image instead of one-size-fits-all.
2. **Uncertainty head.** Trained with a heteroscedastic NLL loss to predict
   per-pixel error magnitude. Used for QC heatmaps and cascade routing.
3. **Cascade-ready interface.** `forward()` returns only the restored image
   (ONNX-export compatible); `forward_with_aux()` also returns the uncertainty
   map and sigma estimates.

Training-time augmentation: flips/rotations/transpose, plus photometric
degradation on the LR input (speckle, Gaussian noise, intensity scale/shift,
JPEG artifacts, blur). `data/ood_synth.py` additionally generates unlimited
procedural textures (fractal noise, particle fields, gratings, circuit-like
geometry) run through the physics degradation pipeline at randomized
severities, for out-of-distribution robustness.

## Repository structure

```
configs/train_config.yaml     training configuration
models/nafnet.py              NAFNet backbone
models/diag_nafnet.py         DiagNAFNet (backbone + encoder + uncertainty)
models/loss.py                composite loss (L1/VGG/SSIM/edge/NLL)
data/dataset.py               paired dataset, augmentations, split logic
data/degrade.py               synthetic degradation pipeline
data/ood_synth.py             procedural OOD texture generator
scripts/train.py              training entry point
scripts/evaluate.py           standalone inference (benchmark entry point)
scripts/compute_metrics.py    PSNR/SSIM/LPIPS evaluation vs baselines
scripts/export_onnx.py        ONNX export + speed benchmark
weights/best.pt               trained checkpoint (Git LFS)
outputs/                      restored test images (Git LFS)
```

## Team

- Team: [add team name if required by the submission form]
- Members: Mohit Sai Vinnakota, Sushant Kumar, Nisarg Amit
- College: Vellore Institute of Technology
- Contact: vinnakotamohitsai@gmail.com / +91 9900076111

## License

MIT - see LICENSE.
