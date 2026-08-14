# KLA Hackathon: AI-Based Restoration of Degraded Semiconductor Images

## Problem Summary
Restore degraded semiconductor inspection images (speckle noise + Gaussian noise + super-resolution) to match ground truth. Must handle all 3 degradations simultaneously, generalize to out-of-distribution test data, and run fast on H100.

## Constraints
- **Timeline**: 2 days
- **Training**: Cloud GPU (H100 target)
- **Data**: Will download later (paired degraded/ground truth, grayscale, 512→256 or 256→128)
- **Submission**: PPT/PDF + GitHub repo with evaluation script, training script, weights, outputs

## Recommended Approach

### Model Architecture: NAFNet (Simple Baseline) → NAFNet-Local (Lightweight)
**Why NAFNet?**
- State-of-the-art for image restoration (denoising, deblurring, super-resolution)
- Simple architecture: no complex attention, just simple gate mechanism
- Fast inference, fewer parameters than Restormer/SwinIR
- Handles all degradation types with single model
- Proven on real-world denoising + SR benchmarks

**Architecture Details:**
- NAFNet (4-block, 48-channel) ≈ 678K params → ~10ms inference on H100
- Input: degraded (256×256 or 128×128) → Output: restored (512×512 or 256×256)
- Use pixel shuffle for upsampling (learnt upsampling)
- Single model handles both scale factors (2× and 4×) via scale conditioning

### Loss Function: Composite Loss
```
L_total = L1 + 0.1 * Perceptual (VGG) + 0.05 * SSIM + 0.1 * Edge (Sobel)
```
- L1: Pixel fidelity
- Perceptual: Texture/structure preservation
- SSIM: Structural similarity
- Edge: Sharpness preservation for semiconductor edges

### Data Augmentation (Critical for OOD Generalization)
- Random rotation (0°, 90°, 180°, 270°)
- Random flip (H/V)
- Gaussian noise injection (σ=0-25)
- Speckle noise simulation (multiplicative)
- Random downsampling (bicubic, bilinear, nearest)
- JPEG compression artifacts (quality 70-95)
- Intensity scaling (0.8-1.2) + shift (-10 to +10) → handles range exceedance

### Training Strategy
- **Phase 1** (fast): 50 epochs, batch=16, lr=2e-4, cosine decay
- **Phase 2** (fine-tune): 20 epochs, batch=8, lr=5e-5, only on harder samples
- Mixed precision (FP16) for speed
- Gradient accumulation if VRAM limited

### Inference Optimization
- ONNX export for H100 benchmarking
- FP16 inference
- Tiled inference for large images (overlap=32)
- Batch processing

## Repository Structure
```
KLA-Hackathon/
├── README.md
├── requirements.txt
├── configs/
│   └── train_config.yaml
├── models/
│   ├── __init__.py
│   ├── nafnet.py          # Core architecture
│   └── loss.py            # Composite loss
├── data/
│   ├── __init__.py
│   ├── dataset.py         # Paired dataset + augmentations
│   └── degrade.py         # Synthetic degradation for dev
├── scripts/
│   ├── train.py           # Training entry point
│   ├── evaluate.py        # **Critical**: standalone evaluation script
│   └── export_onnx.py     # ONNX export for benchmarking
├── weights/               # Trained weights (Git LFS)
├── outputs/               # Restored test images
└── notebooks/
    └── explore_data.ipynb # Data exploration
```

## Key Files to Implement (Priority Order)
1. `models/nafnet.py` - Core model
2. `data/dataset.py` - Data loading + augmentations
3. `models/loss.py` - Composite loss
4. `scripts/train.py` - Training loop
5. `scripts/evaluate.py` - **MUST WORK standalone** (benchmark entry point)
6. `scripts/export_onnx.py` - ONNX export
7. `configs/train_config.yaml` - Hyperparameters
8. `requirements.txt` - Dependencies
9. `README.md` - Setup + usage instructions

## Evaluation Script Requirements (Critical)
```bash
python scripts/evaluate.py \
  --input_dir /path/to/test_degraded \
  --output_dir /path/to/output_restored \
  --weights weights/best.pt \
  --device cuda
```
- No manual edits needed
- Loads model, runs inference on all images, saves outputs
- Handles both 256→512 and 128→256 automatically
- Progress bar, timing stats

## PPT Slides Outline (8-9 slides)
1. Team Details
2. Problem Statement: Why semiconductor inspection needs AI restoration
3. Idea: NAFNet-based unified restoration (denoise + SR)
4. Solution: Architecture, loss, augmentation, training pipeline
5. Innovation: Scale-conditioning, edge-aware loss, OOD augmentations
6. Results: SSIM/PSNR/LPIPS + visual comparisons
7. Tech Stack: PyTorch, H100, ONNX, model size, inference time
8. GitHub + Video Link
9. References

## Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| OOD generalization fails | Heavy augmentation + perceptual loss |
| Slow inference | NAFNet lightweight + ONNX + FP16 |
| Range exceedance (speckle) | Intensity normalization in dataset |
| 2-day deadline | Start with synthetic data, iterate fast |

## Next Steps
1. ✅ Create project structure
2. Implement NAFNet model (models/nafnet.py)
3. Implement dataset with augmentations (data/dataset.py)
4. Implement composite loss (models/loss.py)
5. Implement training script (scripts/train.py)
6. Implement **evaluation script** (scripts/evaluate.py) - PRIORITY
7. Test with synthetic data
8. Train on real data when available
9. Export ONNX, benchmark
10. Generate PPT + finalize repo