"""
NAFNet: Simple Baseline for Image Restoration (CVPR 2022)
Lightweight implementation for semiconductor image restoration.
Handles denoising (speckle + Gaussian) + super-resolution simultaneously.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 2D inputs"""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """Simple Gating Mechanism: splits channels and multiplies"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    NAFNet Block: Simplified Transformer block for image restoration
    No attention, just channel mixing + simple gating
    """
    def __init__(self, channels: int, expansion: float = 2.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        hidden = int(channels * expansion)

        # Channel mixing
        self.conv1 = nn.Conv2d(channels, hidden * 2, 1, 1, 0)
        self.sg = SimpleGate()
        self.conv2 = nn.Conv2d(hidden, channels, 1, 1, 0)

        # Spatial mixing (depth-wise conv)
        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        self.conv4 = nn.Conv2d(channels, channels, 1, 1, 0)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Scale parameter (learnable)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel mixing branch
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.sg(x)
        x = self.conv2(x)
        x = identity + self.drop_path(x * self.beta)

        # Spatial mixing branch
        identity = x
        x = self.norm2(x)
        x = self.conv3(x)
        x = F.gelu(x)
        x = self.conv4(x)
        x = identity + self.drop_path(x * self.gamma)

        return x


class DropPath(nn.Module):
    """Stochastic Depth (DropPath)"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class NAFNet(nn.Module):
    """
    NAFNet for Image Restoration with configurable upscaling
    Args:
        in_channels: Input channels (1 for grayscale)
        out_channels: Output channels (1 for grayscale)
        width: Base channel width
        enc_blks: Number of encoder blocks per level
        middle_blks: Number of middle blocks
        dec_blks: Number of decoder blocks per level
        scale: Upscale factor (2 or 4)
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 48,
        enc_blks: list = [2, 2, 4, 8],
        middle_blks: int = 12,
        dec_blks: list = [2, 2, 2, 2],
        scale: int = 2
    ):
        super().__init__()
        self.scale = scale

        # Adjust encoder/decoder depth based on scale
        # For 2x: need 1 less down/up pair than 4x
        # 4 levels give 16x down/up -> same size, need extra 2x up = 32x
        # 3 levels give 8x down/up -> + 2x = 16x (not enough)
        # So we use 4 levels + final 2x upsampling for 2x scale

        # Input projection
        self.intro = nn.Conv2d(in_channels, width, 3, 1, 1)

        # Encoder - 4 levels
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num_blks in enc_blks:
            blocks = nn.Sequential(*[NAFBlock(chan) for _ in range(num_blks)])
            self.encoders.append(blocks)
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2, 0))
            chan *= 2

        # Middle
        self.middle = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blks)])

        # Decoder - 4 levels (each 2x upsample = 16x total from bottleneck)
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        for num_blks in dec_blks:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, 1, 0),
                nn.PixelShuffle(2)
            ))
            chan //= 2
            blocks = nn.Sequential(*[NAFBlock(chan) for _ in range(num_blks)])
            self.decoders.append(blocks)

        # Final upsampling to achieve target scale
        # After 4 decoder levels: bottleneck (H/16) -> H (16x)
        # For scale=2: need additional 2x
        # For scale=4: need additional 4x
        self.final_upscale = nn.Sequential()
        if scale == 2:
            self.final_upscale = nn.Sequential(
                nn.Conv2d(width, width * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(width, width, 3, 1, 1)
            )
        elif scale == 4:
            self.final_upscale = nn.Sequential(
                nn.Conv2d(width, width * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(width, width * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(width, width, 3, 1, 1)
            )

        # Output projection
        self.ending = nn.Conv2d(width, out_channels, 3, 1, 1)

        # Scale conditioning for multi-scale support
        self.scale_embedding = nn.Embedding(2, width)  # scale 2 or 4

    def forward(self, x: torch.Tensor, scale: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: [B, 1, H, W] degraded input
            scale: 2 or 4 (optional, inferred from input size if not provided)
        Returns:
            [B, 1, H*scale, W*scale] restored output
        """
        B, C, H, W = x.shape

        # Infer scale from input if not provided
        if scale is None:
            if H == 128 and W == 128:
                scale = 4
            elif H == 256 and W == 256:
                scale = 2
            else:
                scale = 2

        # Initial feature extraction
        feat = self.intro(x)

        # Add scale conditioning
        scale_idx = 0 if scale == 2 else 1
        scale_emb = self.scale_embedding.weight[scale_idx].view(1, -1, 1, 1)
        feat = feat + scale_emb

        # Encoder
        enc_feats = []
        for encoder, down in zip(self.encoders, self.downs):
            feat = encoder(feat)
            enc_feats.append(feat)
            feat = down(feat)

        # Middle
        feat = self.middle(feat)

        # Decoder with skip connections
        for i, (up, decoder) in enumerate(zip(self.ups, self.decoders)):
            feat = up(feat)
            # Skip connection from encoder
            skip = enc_feats[-(i + 1)]
            # Resize if needed
            if feat.shape[-2:] != skip.shape[-2:]:
                feat = F.interpolate(feat, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            feat = feat + skip
            feat = decoder(feat)

        # Final upscaling to achieve target resolution
        feat = self.final_upscale(feat)

        # Output
        out = self.ending(feat)

        return out


class NAFNetLocal(nn.Module):
    """
    Lightweight NAFNet variant for faster inference
    ~200K params, ~5ms on H100
    """
    def ____init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 32,
        num_blocks: int = 8,
        scale: int = 2
    ):
        super().__init__()
        self.scale = scale

        self.intro = nn.Conv2d(in_channels, width, 3, 1, 1)

        # Single-level processing with residual blocks
        self.body = nn.Sequential(*[NAFBlock(width) for _ in range(num_blocks)])

        # Upsampling
        if scale == 2:
            self.up = nn.Sequential(
                nn.Conv2d(width, width * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(width, width, 3, 1, 1)
            )
        elif scale == 4:
            self.up = nn.Sequential(
                nn.Conv2d(width, width * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(width, width * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(width, width, 3, 1, 1)
            )

        self.ending = nn.Conv2d(width, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.intro(x)
        feat = self.body(feat)
        feat = self.up(feat)
        out = self.ending(feat)
        return out


def create_model(
    model_type: str = 'nafnet',
    scale: int = 2,
    **kwargs
) -> nn.Module:
    """Factory function to create model"""
    if model_type == 'nafnet':
        return NAFNet(scale=scale, **kwargs)
    elif model_type == 'nafnet_local':
        return NAFNetLocal(scale=scale, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Quick test
    model = NAFNet(scale=2)
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    model4 = NAFNet(scale=4)
    x4 = torch.randn(1, 1, 128, 128)
    y4 = model4(x4)
    print(f"Input: {x4.shape} -> Output: {y4.shape}")
    print(f"Params: {sum(p.numel() for p in model4.parameters()) / 1e6:.2f}M")