# Restoration Metrics (validation split)

- Checkpoint: `weights/best.pt`
- Images: 320 (val_split=0.1, seed 42 - identical to training)
- Evaluated on: cpu in 101.8s

| Method | PSNR (dB) | SSIM | LPIPS |
|--------|-----------|------|-------|
| bicubic | 23.36 | 0.5572 | 0.4309 |
| model | 27.96 | 0.7651 | 0.2825 |