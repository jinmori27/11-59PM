"""
DiagNAFNet: Self-Diagnosing NAFNet
==================================
NAFNet backbone extended with three out-of-the-box capabilities:

1. Degradation Encoder  - a small CNN that estimates how corrupted the input is
   (speckle/gaussian sigma proxies) and conditions the restorer through FiLM
   (Feature-wise Linear Modulation) layers. The model "diagnoses before it treats".

2. Uncertainty Head     - predicts a per-pixel log-variance map trained with a
   heteroscedastic NLL loss. At inference this doubles as a confidence map
   (defect-inspection QC use case) and drives the cascade escalation policy.

3. Cascade-ready        - `forward_with_aux()` exposes (restored, logvar, sigmas)
   so evaluate.py can route easy images to a small model and escalate only the
   hard ones to the big model.

The plain `forward()` returns a tensor only, so ONNX export and existing
evaluation code keep working unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .nafnet import NAFNet


class DegradationEncoder(nn.Module):
    """Estimates degradation statistics from the LR input.

    Returns a conditioning embedding plus two softplus-activated sigma proxies
    (interpreted as speckle/gaussian severity; trained end-to-end, unsupervised).
    """
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.GELU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, embed_dim),
            nn.GELU(),
        )
        self.sigma_head = nn.Sequential(nn.Linear(embed_dim, 2), nn.Softplus())
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.net(x)
        return emb, self.sigma_head(emb)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: x * (1 + scale) + shift.

    Final linear layers are zero-initialised so the modulation starts as an
    identity and is learned only if it helps.
    """
    def __init__(self, embed_dim: int, channels: int):
        super().__init__()
        self.scale = nn.Linear(embed_dim, channels)
        self.shift = nn.Linear(embed_dim, channels)
        for layer in (self.scale, self.shift):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        s = self.scale(emb).view(x.shape[0], -1, 1, 1)
        t = self.shift(emb).view(x.shape[0], -1, 1, 1)
        return x * (1 + s) + t


class DiagNAFNet(NAFNet):
    """NAFNet + degradation conditioning + uncertainty estimation.

    Accepts the same constructor arguments as NAFNet (width, enc_blks,
    middle_blks, dec_blks, scale, ...) plus:
        use_film: enable degradation-conditioned FiLM layers
        use_uncertainty: enable the per-pixel log-variance head
        embed_dim: size of the degradation embedding
    """
    def __init__(
        self,
        use_film: bool = True,
        use_uncertainty: bool = True,
        embed_dim: int = 64,
        **nafnet_kwargs
    ):
        super().__init__(**nafnet_kwargs)
        width = self.ending.in_channels  # width actually used by the backbone
        self.use_film = use_film
        self.use_uncertainty = use_uncertainty

        self.deg_encoder = DegradationEncoder(embed_dim)
        if use_film:
            self.film_in = FiLM(embed_dim, width)
            self.film_out = FiLM(embed_dim, width)
        if use_uncertainty:
            # Zero-init weights, bias -> log(0.05^2): initial sigma ~ 0.05
            self.unc_head = nn.Conv2d(width, 1, 1)
            nn.init.zeros_(self.unc_head.weight)
            self.unc_head.bias.data.fill_(-6.0)

    def _extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared encoder/decoder trunk; returns (features, embedding, sigma_proxies)."""
        emb, sigmas = self.deg_encoder(x)

        feat = self.intro(x)
        if self.use_film:
            feat = self.film_in(feat, emb)

        # Scale conditioning (fixed by constructor, matching the 2x datasets)
        scale_idx = 0 if self.scale == 2 else 1
        feat = feat + self.scale_embedding.weight[scale_idx].view(1, -1, 1, 1)

        enc_feats = []
        for encoder, down in zip(self.encoders, self.downs):
            feat = encoder(feat)
            enc_feats.append(feat)
            feat = down(feat)

        feat = self.middle(feat)

        for i, (up, decoder) in enumerate(zip(self.ups, self.decoders)):
            feat = up(feat)
            skip = enc_feats[-(i + 1)]
            if feat.shape[-2:] != skip.shape[-2:]:
                feat = F.interpolate(feat, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            feat = feat + skip
            feat = decoder(feat)

        if self.use_film:
            feat = self.film_out(feat, emb)

        feat = self.final_upscale(feat)
        return feat, emb, sigmas

    def forward(self, x: torch.Tensor, scale: Optional[int] = None) -> torch.Tensor:
        feat, _, _ = self._extract_features(x)
        return self.ending(feat)

    def forward_with_aux(
        self, x: torch.Tensor, scale: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Returns (restored, logvar_map, sigma_proxies).

        logvar_map is None when use_uncertainty=False.
        """
        feat, _, sigmas = self._extract_features(x)
        out = self.ending(feat)
        logvar = self.unc_head(feat).clamp(-12, 4) if self.use_uncertainty else None
        return out, logvar, sigmas
