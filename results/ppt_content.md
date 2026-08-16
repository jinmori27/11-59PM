# PPT Content - KLA Hackathon Submission

Slide-by-slide content with the real measured numbers. Speaker notes under
each slide. Comparison images referenced live in `results/examples/` and
`outputs/`.

---

## Slide 1 - Title

**AI-Based Restoration of Degraded Semiconductor Images**
Joint despeckling, denoising, and 2x super-resolution

Mohit Sai Vinnakota | Sushant Kumar | Nisarg Amit
Vellore Institute of Technology
vinnakotamohitsai@gmail.com | +91 9900076111

*Notes: 15 seconds. One line: "One model that undoes speckle, noise, and
downsampling together, and tells you where it might be wrong."*

---

## Slide 2 - Problem

- Inspection images arrive degraded three ways at once:
  - Speckle noise (multiplicative; pushes pixels outside the valid range)
  - Additive Gaussian noise (softens edges and fine structures)
  - Resolution loss (256->128 downsampling destroys defect detail)
- Requirements: restore full resolution, generalize to unseen image
  distributions (test set is partly out-of-distribution), and stay fast -
  end-to-end runtime is measured on an H100
- Data: 3,200 paired samples (clean 256x256 GT / degraded 128x128 NoisyLR)

*Notes: Emphasize "simultaneously" - this is not three chained models, it is
one restoration pass. Point at the histogram claim from the problem
statement: speckle widens the intensity distribution beyond GT range.*

---

## Slide 3 - Approach: DiagNAFNet

Base: NAFNet (Chen et al., ECCV 2022) - simple, fast, restoration-proven
backbone. 65.3M parameters.

Three additions of ours:

1. **Degradation encoder (self-diagnosis)** - a small CNN estimates how
   corrupted each input is; FiLM conditioning layers adapt the restorer to
   that estimate. The model diagnoses the degradation before treating it.
2. **Per-pixel uncertainty head** - trained with a heteroscedastic NLL loss;
   outputs a map of predicted error magnitude alongside the restoration.
3. **Cascade-ready interface** - the same uncertainty signal can route easy
   images to a small fast model and escalate only hard images to the big one.

*Notes: This is the innovation slide. The pitch: "our model knows how bad the
input is, and knows where it is unsure."*

---

## Slide 4 - Training Strategy

- Composite loss: L1 + VGG-perceptual (0.1) + SSIM (0.05) + Sobel edge (0.1)
  + heteroscedastic NLL (0.05)
- Paired geometric augmentation (flips, 90-degree rotations, transpose)
- Photometric degradation on inputs: extra speckle, Gaussian noise,
  intensity scale/shift, JPEG artifacts, blur - covering more of the
  degradation space than the training data alone
- **Synthetic OOD curriculum**: unlimited procedurally generated textures
  (fractal noise, particle fields, gratings, circuit-like geometry) run
  through the physics degradation pipeline at randomized severities - 25% of
  every epoch
- EMA of weights, validated separately; best checkpoint kept automatically
- Trained on Kaggle T4: 100 epochs, ~6 hours, best val loss 0.0659

*Notes: The OOD curriculum is the direct answer to the problem statement's
"test set includes out-of-distribution samples."*

---

## Slide 5 - Results

Validation split (320 held-out images):

| Method | PSNR (dB) | SSIM | LPIPS |
|---|---|---|---|
| Bicubic upsampling | 23.36 | 0.557 | 0.431 |
| **DiagNAFNet (ours)** | **27.96** | **0.765** | **0.283** |

- +4.6 dB PSNR over bicubic baseline
- LPIPS (perceptual) improved 0.431 -> 0.283
- Visual comparisons: results/examples/compare_*.png
  (input | bicubic | ours | ground truth)

*Notes: Lead with the delta. "Almost 5 dB is the difference between a blurry
guess and usable inspection imagery." Show 2-3 comparison strips side by side.*

---

## Slide 6 - Uncertainty: Knowing Where Restoration Is Unreliable

- Same model, same forward pass: restored image + per-pixel confidence map
- Heatmaps (JET): blue = confident, red = high predicted error
- Use case for inspection QC: prioritize human review on high-uncertainty
  regions instead of reviewing everything
- 400 test outputs + heatmaps generated: outputs/XXXXXX.png and
  outputs/XXXXXX_unc.png
- Bonus: this signal drives the fast/big model cascade at inference

*Notes: This is the demo moment - put one image next to its heatmap. This
feature does not exist in vanilla restoration baselines.*

---

## Slide 7 - Speed and Deployment

- Standalone evaluation script (required for benchmarking):
  `evaluate.py --input_dir <in> --output_dir <out> --weights best.pt`
- Batched inference pipeline: uniform-size batches, FP16 on GPU, threaded
  image writing - the benchmark counts I/O, so the whole path is optimized
- Measured on CPU (laptop, no GPU): 268 ms/image, 3.7 img/s for the full
  400-image test set - GPU (T4/H100) runtime is far lower
- ONNX export with dynamic axes ready (`scripts/export_onnx.py`), including
  an FP16 variant for H100 benchmarking
- Optional: --tta geometric self-ensemble (+quality, 8x compute) and
  uncertainty-gated cascade (big-model quality at near-small-model cost)

*Notes: Honest number discipline: quote the CPU measurement as measured,
GPU as expected-faster, and mention the ONNX path for the official H100 run.*

---

## Slide 8 - Reproducibility and Repo

- Complete pipeline on GitHub: training script, config, standalone
  evaluation script, metrics script, ONNX export
- `weights/best.pt` (Git LFS), 400 restored test outputs, environment
  freeze (`requirements_freeze.txt` = pip freeze of the training
  environment)
- Deterministic train/val split (seed 42) - reported metrics reproduce
  exactly via `scripts/compute_metrics.py`

---

## Slide 9 - References

1. Chen, L. et al., "Simple Baselines for Image Restoration", ECCV 2022 (NAFNet)
2. Perez, E. et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018
3. Nix, D. & Weigend, A., "Estimating the mean and variance of the target
   probability distribution", NeurIPS 1994 (heteroscedastic uncertainty)
4. Wang, Z. et al., "Image Quality Assessment: From Error Visibility to
   Structural Similarity", TIP 2004 (SSIM)
5. Zhang, R. et al., "The Unreasonable Effectiveness of Deep Features as a
   Perceptual Metric", CVPR 2018 (LPIPS)
