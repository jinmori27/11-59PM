# KLA Hackathon: AI-Based Restoration of Degraded Semiconductor Images

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## ��� Problem Statement

Semiconductor manufacturing relies on microscopic inspection images to detect defects at every production stage. These images are degraded by:
- **Speckle Noise**: Multiplicative grainy noise pushing pixel values beyond true range
- **Gaussian Noise**: Additive noise softening edges and fine structures
- **Spatial Resolution Reduction**: Downsampling (512→256 or 256→128) losing critical detail

**Goal**: Train a single AI model that simultaneously denoises (speckle + Gaussian) and super-resolves degraded images back to ground truth quality.

## ��� Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/KLA-Hackathon.git
cd KLA-Hackathon

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

Place paired training data in this structure (`.npy` float32 arrays):
```
data_root/                 # any path passed via --data_root
├── GT/                    # Clean high-res images (256x256 or 512x512), 0-1 range
└── NoisyLR/               # Noisy low-res images (128x128 or 256x256),
                           # values may exceed [0,1] due to speckle
```

**Note**: Filenames must match between `GT/` and `NoisyLR/`. A random 90/10 train/val split is created automatically (seeded).

### 3. Train Model

```bash
# Basic training (uses configs/train_config.yaml)
python scripts/train.py --data_root data/train --output_dir weights

# With custom config and wandb logging
python scripts/train.py \
    --data_root data/train \
    --output_dir weights \
    --config configs/train_config.yaml \
    --wandb \
    --wandb_project kla-restoration \
    --wandb_run_name nafnet-2x

# Resume from checkpoint
python scripts/train.py --data_root data/train --output_dir weights --resume weights/latest.pt
```

**Key training features:**
- Mixed precision (FP16) for 2x speedup
- Cosine annealing LR schedule
- Composite loss: L1 + Perceptual (VGG) + SSIM + Edge
- Heavy augmentations for OOD generalization
- Automatic checkpointing (best + latest + periodic)

### 4. Evaluate / Inference (Critical for Benchmarking)

```bash
# Basic inference on test images
python scripts/evaluate.py \
    --input_dir test_degraded \
    --output_dir outputs \
    --weights weights/best.pt

# With tiling for large images (>256x256)
python scripts/evaluate.py \
    --input_dir test_degraded \
    --output_dir outputs \
    --weights weights/best.pt \
    --tile_size 256 \
    --overlap 32

# FP16 inference (faster on GPU)
python scripts/evaluate.py \
    --input_dir test_degraded \
    --output_dir outputs \
    --weights weights/best.pt \
    --fp16

# CPU inference
python scripts/evaluate.py \
    --input_dir test_degraded \
    --output_dir outputs \
    --weights weights/best.pt \
    --device cpu
```

**Evaluation script requirements met:**
- �� Standalone `.py` file (not notebook)
- �� Accepts `--input_dir` and `--output_dir`
- �� Loads model, runs inference on all images, saves outputs
- �� Handles 128→256 and 256→512 inputs automatically (both are 2× SR)
- �� Progress bar + timing stats
- �� No manual edits needed

### 5. Export to ONNX (for H100 Benchmarking)

```bash
# Export with dynamic axes (variable input size)
python scripts/export_onnx.py \
    --weights weights/best.pt \
    --output weights/model.onnx

# Export FP16 for faster inference
python scripts/export_onnx.py \
    --weights weights/best.pt \
    --output weights/model_fp16.onnx \
    --fp16

# Export fixed input size (256x256)
python scripts/export_onnx.py \
    --weights weights/best.pt \
    --output weights/model_fixed.onnx \
    --static \
    --benchmark
```

## ��� Repository Structure

```
KLA-Hackathon/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── configs/
│   └── train_config.yaml     # Training hyperparameters
├── models/
│   ├── __init__.py
│   ├── nafnet.py             # NAFNet architecture
│   └── loss.py               # Composite loss (L1+Perceptual+SSIM+Edge)
├── data/
│   ├── __init__.py
│   ├── dataset.py            # Paired dataset + augmentations
│   └── degrade.py            # Synthetic degradation for dev
├── scripts/
│   ├── train.py              # Training entry point
│   ├── evaluate.py           # *** CRITICAL: Standalone inference ***
│   └── export_onnx.py        # ONNX export for benchmarking
├── weights/                  # Trained models (Git LFS)
│   ├── best.pt              # Best validation model
│   └── latest.pt            # Latest checkpoint
├── outputs/                  # Restored test images (submission)
��── notebooks/
    └── explore_data.ipynb    # Data exploration
```

## Model Architecture: DiagNAFNet (Self-Diagnosing NAFNet)

**Why NAFNet?**
- Simple baseline for image restoration (CVPR 2022)
- No complex attention -> fast inference
- Handles denoising + deblurring + super-resolution in one model
- ~65M params (width 48 config) / ~0.11M params (local variant, width 32)

**Architecture Details:**
```
Input (1, H, W) -> Intro Conv -> Encoder (4 levels) -> Middle (12 blocks)
                                                    -> Decoder (4 levels + skips)
                                                    -> Ending Conv -> Output (1, H x scale, W x scale)
```
- **Upscaling**: PixelShuffle-based learned 2x upsampling (both dataset variants, 128->256 and 256->512, are 2x)
- **SimpleGate**: Channel-wise gating instead of attention
- **LayerNorm2d**: Channel-wise normalization

### Out-of-the-Box Features

**1. Self-Diagnosing Restoration (degradation conditioning)**
A small degradation encoder looks at the input first and estimates how corrupted
it is (speckle/Gaussian severity proxies), then conditions the restorer through
FiLM layers. The model adapts its behaviour per image instead of one-size-fits-all.

**2. Per-Pixel Uncertainty (heteroscedastic NLL loss)**
The model outputs a per-pixel confidence map alongside the restoration - flag
*where* the result is unreliable. Enable at inference with `--save_uncertainty`
to write JET heatmaps next to outputs. Semiconductor QC use case: prioritize
human review of high-uncertainty regions.

**3. Uncertainty-Gated Cascade Inference**
Run a small fast model on everything; escalate only the images it is unsure
about to the big model. Quality of the big model, speed close to the small one:
```bash
python scripts/evaluate.py --input_dir test --output_dir out     --weights weights/fast_diag.pt --weights2 weights/best.pt
```

**4. Infinite Synthetic OOD Curriculum**
`data/ood_synth.py` procedurally generates never-repeating textures (fractal
fBm, particle fields, gratings, circuit-like geometry, speckle fields) and
runs them through the physics degradation pipeline at randomized severities.
Mix into training with `synth_ratio` (default 0.25) - targets the
out-of-distribution half of the test set.

**5. Fast Batched Inference Pipeline + Geometric Self-Ensemble**
The evaluation script's default path batches uniform-size inputs, keeps FP16
on GPU, and writes images on a thread pool (the benchmark counts I/O time).
`--tta` averages 8 geometric views for a free quality bump when speed is not
the priority.

**6. EMA Weights**
Training maintains an exponential moving average of weights, validated
separately each epoch; `best.pt` stores whichever (raw vs EMA) validates better.

## ��� Loss Function

```
L_total = 1.0 × L1 + 0.1 × Perceptual(VGG19) + 0.05 × SSIM + 0.1 × Edge(Sobel)
```

| Component | Weight | Purpose |
|-----------|--------|---------|
| L1 (Charbonnier) | 1.0 | Pixel fidelity |
| Perceptual (VGG19 relu1-5) | 0.1 | Texture/structure |
| SSIM | 0.05 | Structural similarity |
| Edge (Sobel gradients) | 0.1 | Sharpness preservation |

## ��� Data Augmentations (Critical for OOD)

**Geometric (both degraded & GT):**
- Horizontal/Vertical flip (50%)
- 90° rotation (50%)
- Transpose (30%)
- Random crop to 256×256

**Photometric (degraded only):**
- Speckle noise (σ=0.05-0.2, multiplicative)
- Gaussian noise (σ=0-25/255, additive)
- Intensity scaling (0.8-1.2×) + shift (±10/255)
- JPEG compression (Q=70-95)
- Gaussian blur (σ=0.5-1.5)

## ��� Expected Results

| Metric | Target (In-Dist) | Target (OOD) |
|--------|------------------|--------------|
| PSNR | >32 dB | >28 dB |
| SSIM | >0.92 | >0.85 |
| LPIPS | <0.08 | <0.15 |
| Inference (H100) | <15ms | <15ms |

## ��� Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 8GB VRAM (RTX 3070) | 24GB+ (H100/A100) |
| RAM | 16GB | 32GB+ |
| Storage | 50GB | 100GB+ |

**Cloud Training**: Tested on H100 (80GB) - ~2 hours for 50 epochs

## ��� Submission Checklist

- [x] **README.md** - Complete setup & usage instructions
- [x] **requirements.txt** - Full dependency list
- [x] **scripts/evaluate.py** - Standalone inference (CRITICAL)
- [x] **scripts/train.py** - Reproducible training
- [x] **scripts/export_onnx.py** - ONNX export for H100
- [x] **models/nafnet.py** - Core architecture
- [x] **models/loss.py** - Composite loss
- [x] **data/dataset.py** - Data loading + augmentations
- [x] **configs/train_config.yaml** - Hyperparameters
- [ ] **weights/best.pt** - Trained model (Git LFS)
- [ ] **outputs/** - Restored test images

## ��� Reproducing Results

```bash
# 1. Setup
git clone https://github.com/YOUR_USERNAME/KLA-Hackathon.git
cd KLA-Hackathon
pip install -r requirements.txt

# 2. Download data (KLA provides paired dataset)
# Place in data/train/{degraded,ground_truth}/

# 3. Train
python scripts/train.py --data_root data/train --output_dir weights --wandb

# 4. Evaluate on test set
python scripts/evaluate.py \
    --input_dir test_degraded \
    --output_dir outputs \
    --weights weights/best.pt

# 5. Export for benchmarking
python scripts/export_onnx.py \
    --weights weights/best.pt \
    --output weights/model.onnx \
    --fp16 --benchmark
```

## ��� References

1. **NAFNet**: Chen et al., "Simple Baselines for Image Restoration", ECCV 2022
2. **Perceptual Loss**: Johnson et al., "Perceptual Losses for Real-Time Style Transfer", ECCV 2016
3. **SSIM**: Wang et al., "Image Quality Assessment: SSIM", TIP 2004
4. **Speckle Noise**: Goodman, "Statistical Properties of Laser Speckle Patterns", 1976

## ��� Team

- **Team Name**: [YOUR_TEAM_NAME]
- **Members**: [Member names & roles]
- **College**: [College name]
- **Contact**: [Email/Phone]

## ��� License

MIT License - See LICENSE file for details.

---

**For KLA Benchmarking Team**: The evaluation script (`scripts/evaluate.py`) is ready to run as-is. Just provide `--input_dir`, `--output_dir`, and `--weights`. It handles 2× super-resolution (128→256 and 256→512 inputs), supports tiled inference for large images, and outputs timing statistics.