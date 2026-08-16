#!/usr/bin/env python3
"""
Compute restoration metrics (PSNR / SSIM / LPIPS) for a trained model on the
validation split, with a bicubic-upsampling baseline for reference.

Reproduces the exact training val split (same seed), so numbers are directly
comparable to the val loss logged during training.

Usage:
    python scripts/compute_metrics.py \
        --data_root /kaggle/input/datasets/<you>/kla-dataset/train \
        --weights /kaggle/working/weights/best.pt \
        --out results/metrics.md

    # Skip LPIPS (avoids the AlexNet weight download):
    ... --skip_lpips
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import create_model_from_config
from data import create_dataloaders


def psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    mse = float(np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2))
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def _ssim_numpy(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    """Gaussian-window SSIM fallback when scikit-image is unavailable
    (equivalent to skimage gaussian_weights=True, sigma=1.5)."""
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a ** 2
    var_b = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / \
               ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return float(ssim_map.mean())


def load_model(weights_path: str, device: str) -> torch.nn.Module:
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    if 'config' in checkpoint:
        model = create_model_from_config(checkpoint['config'])
    else:
        model = create_model_from_config({})
    state = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state)
    return model.to(device).eval()


def make_strip(lr: np.ndarray, panels: list, labels: list, out_path: Path):
    """Save a side-by-side comparison strip (all panels already 0-255 uint8)."""
    gap = 4
    h, w = panels[0].shape[:2]
    strip = np.full((h + 24, (w + gap) * len(panels) - gap, 3), 255, dtype=np.uint8)
    for i, (panel, label) in enumerate(zip(panels, labels)):
        x = i * (w + gap)
        strip[24:h + 24, x:x + w, :] = panel
        cv2.putText(strip, label, (x + 4, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), strip)


def main():
    parser = argparse.ArgumentParser(description='Compute PSNR/SSIM/LPIPS vs bicubic baseline')
    parser.add_argument('--data_root', type=str, required=True, help='Root containing GT/ and NoisyLR/')
    parser.add_argument('--weights', type=str, required=True, help='Trained checkpoint (.pt)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--val_split', type=float, default=0.1, help='Must match training to reproduce the split')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--skip_lpips', action='store_true', help='Skip LPIPS (no AlexNet download)')
    parser.add_argument('--examples', type=int, default=8, help='Number of comparison strips to save (0 disables)')
    parser.add_argument('--examples_dir', type=str, default='results/examples')
    parser.add_argument('--out', type=str, default='results/metrics.md')
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    # Same seed/split logic as training -> identical val set
    _, val_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=256,
        scale=2,
        val_split=args.val_split,
    )

    model = load_model(args.weights, device)
    print(f"Model loaded from {args.weights}")

    lpips_fn = None
    if not args.skip_lpips:
        try:
            import lpips as lpips_lib
            lpips_fn = lpips_lib.LPIPS(net='alex').to(device).eval()
            print("LPIPS (alexnet) loaded")
        except Exception as e:
            print(f"LPIPS unavailable ({e}); continuing without it")
            lpips_fn = None

    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except ImportError:
        ssim_fn = _ssim_numpy
        print("scikit-image not found; using numpy SSIM fallback")

    stats = {
        'bicubic': {'psnr': [], 'ssim': [], 'lpips': []},
        'model': {'psnr': [], 'ssim': [], 'lpips': []},
    }
    examples_dir = Path(args.examples_dir)
    if args.examples > 0:
        examples_dir.mkdir(parents=True, exist_ok=True)

    n_done, n_saved, t_start = 0, 0, time.time()
    with torch.no_grad():
        for batch in val_loader:
            degraded = batch['degraded'].to(device)
            gt = batch['ground_truth'].to(device)
            restored = model(degraded)

            for i in range(degraded.shape[0]):
                lr01 = np.clip(((degraded[i, 0].cpu().numpy() + 1) / 2), 0, 1)
                gt01 = np.clip(((gt[i, 0].cpu().numpy() + 1) / 2), 0, 1)
                out01 = np.clip(((restored[i, 0].cpu().numpy() + 1) / 2), 0, 1)
                h, w = lr01.shape
                bic01 = np.clip(cv2.resize(lr01, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC), 0, 1)

                for name, img in (('bicubic', bic01), ('model', out01)):
                    stats[name]['psnr'].append(psnr(gt01, img))
                    stats[name]['ssim'].append(
                        ssim_fn(gt01, img, data_range=1.0))

                if lpips_fn is not None:
                    def to_lpips(a):
                        return torch.from_numpy(a[None, None]).float().to(device) * 2 - 1
                    with torch.no_grad():
                        gt_t = to_lpips(gt01)
                        stats['bicubic']['lpips'].append(
                            float(lpips_fn(to_lpips(bic01), gt_t)))
                        stats['model']['lpips'].append(
                            float(lpips_fn(to_lpips(out01), gt_t)))

                if n_saved < args.examples:
                    lr_disp = cv2.resize(lr01, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
                    make_strip(
                        lr01,
                        [cv2.cvtColor((p * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                         for p in (lr_disp, bic01, out01, gt01)],
                        ['input (nearest 2x)', 'bicubic', 'model', 'ground truth'],
                        examples_dir / f'compare_{n_saved:03d}.png',
                    )
                    n_saved += 1
                n_done += 1

    wall = time.time() - t_start

    def row(name):
        s = stats[name]
        lp = f"{np.mean(s['lpips']):.4f}" if s['lpips'] else 'n/a'
        return (f"| {name} | {np.mean(s['psnr']):.2f} | {np.mean(s['ssim']):.4f} | {lp} |")

    lines = [
        "# Restoration Metrics (validation split)",
        "",
        f"- Checkpoint: `{args.weights}`",
        f"- Images: {n_done} (val_split={args.val_split}, seed 42 - identical to training)",
        f"- Evaluated on: {device} in {wall:.1f}s",
        "",
        "| Method | PSNR (dB) | SSIM | LPIPS |",
        "|--------|-----------|------|-------|",
        row('bicubic'),
        row('model'),
    ]
    report = "\n".join(lines)
    print("\n" + report + "\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding='utf-8')
    print(f"Saved: {out_path}")
    if args.examples > 0:
        print(f"Comparison strips: {examples_dir}/compare_*.png")


if __name__ == '__main__':
    main()
