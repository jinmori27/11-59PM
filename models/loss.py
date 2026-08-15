"""
Composite Loss Function for Semiconductor Image Restoration
Combines L1, Perceptual (VGG), SSIM, and Edge losses
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
import kornia.losses as kornia_losses


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss"""
    def __init__(self, layer_weights: dict = None, use_l1: bool = True):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False

        # Extract feature maps from specific layers
        # relu1_2, relu2_2, relu3_2, relu4_2, relu5_2
        self.layers = nn.ModuleList()
        current = nn.Sequential()
        layer_names = ['relu1_2', 'relu2_2', 'relu3_2', 'relu4_2', 'relu5_2']
        layer_indices = [3, 8, 17, 26, 35]  # VGG19 layer indices for these layers

        idx = 0
        for i, module in enumerate(vgg):
            current.add_module(str(i), module)
            if i in layer_indices:
                self.layers.append(current)
                current = nn.Sequential()

        self.layer_weights = layer_weights or {
            'relu1_2': 1.0/32,
            'relu2_2': 1.0/16,
            'relu3_2': 1.0/8,
            'relu4_2': 1.0/4,
            'relu5_2': 1.0
        }
        self.use_l1 = use_l1
        self.criterion = nn.L1Loss() if use_l1 else nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Normalize to ImageNet stats for VGG
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)

        # Convert grayscale to 3-channel for VGG
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std

        loss = 0.0
        pred_feats = pred_norm
        target_feats = target_norm

        for i, layer in enumerate(self.layers):
            pred_feats = layer(pred_feats)
            target_feats = layer(target_feats)
            layer_name = list(self.layer_weights.keys())[i]
            weight = self.layer_weights[layer_name]
            loss += weight * self.criterion(pred_feats, target_feats.detach())

        return loss


class SSIMLoss(nn.Module):
    """SSIM Loss using Kornia"""
    def __init__(self, window_size: int = 11, reduction: str = 'mean'):
        super().__init__()
        self.loss_fn = kornia_losses.SSIMLoss(window_size=window_size, reduction=reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)


class EdgeLoss(nn.Module):
    """Edge-aware loss using Sobel gradients"""
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == 'l1':
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.MSELoss()

        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def get_gradients(self, x: torch.Tensor) -> tuple:
        """Compute Sobel gradients"""
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return gx, gy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_gx, pred_gy = self.get_gradients(pred)
        target_gx, target_gy = self.get_gradients(target)

        loss = self.criterion(pred_gx, target_gx) + self.criterion(pred_gy, target_gy)
        return loss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 variant with epsilon)"""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss


class HeteroscedasticNLL(nn.Module):
    """Pixel-wise NLL under a Laplace likelihood: |err|/sigma + 0.5*log(sigma^2).

    Trains the model's uncertainty head: sigma grows where errors are
    systematically large, giving a calibrated per-pixel confidence map.
    """
    def __init__(self, clamp_min: float = -12.0, clamp_max: float = 4.0):
        super().__init__()
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, pred: torch.Tensor, target: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = logvar.clamp(self.clamp_min, self.clamp_max)
        err = (pred - target).abs()
        return (err / torch.exp(0.5 * logvar) + 0.5 * logvar).mean()


class CompositeLoss(nn.Module):
    """
    Composite Loss for Semiconductor Image Restoration
    L_total = w1*L1 + w2*Perceptual + w3*SSIM + w4*Edge + w5*Charbonnier + w6*NLL
    """
    def __init__(
        self,
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        ssim_weight: float = 0.05,
        edge_weight: float = 0.1,
        charbonnier_weight: float = 0.0,
        uncertainty_weight: float = 0.05,
        use_perceptual: bool = True,
        use_ssim: bool = True,
        use_edge: bool = True
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.charbonnier_weight = charbonnier_weight
        self.uncertainty_weight = uncertainty_weight

        self.l1 = nn.L1Loss()
        self.charbonnier = CharbonnierLoss()
        self.nll = HeteroscedasticNLL()

        self.use_perceptual = use_perceptual
        self.use_ssim = use_ssim
        self.use_edge = use_edge

        if use_perceptual:
            self.perceptual = PerceptualLoss()
        if use_ssim:
            self.ssim = SSIMLoss()
        if use_edge:
            self.edge = EdgeLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, logvar: torch.Tensor = None) -> dict:
        losses = {}

        # L1 Loss
        l1_loss = self.l1(pred, target)
        losses['l1'] = l1_loss
        total = self.l1_weight * l1_loss

        # Charbonnier Loss (optional)
        if self.charbonnier_weight > 0:
            charb_loss = self.charbonnier(pred, target)
            losses['charbonnier'] = charb_loss
            total += self.charbonnier_weight * charb_loss

        # Perceptual Loss
        if self.use_perceptual:
            perc_loss = self.perceptual(pred, target)
            losses['perceptual'] = perc_loss
            total += self.perceptual_weight * perc_loss

        # SSIM Loss
        if self.use_ssim:
            ssim_loss = self.ssim(pred, target)
            losses['ssim'] = ssim_loss
            total += self.ssim_weight * ssim_loss

        # Edge Loss
        if self.use_edge:
            edge_loss = self.edge(pred, target)
            losses['edge'] = edge_loss
            total += self.edge_weight * edge_loss

        # Heteroscedastic NLL (requires the model's uncertainty head)
        if logvar is not None and self.uncertainty_weight > 0:
            nll_loss = self.nll(pred, target, logvar)
            losses['nll'] = nll_loss
            total += self.uncertainty_weight * nll_loss

        losses['total'] = total
        return losses


def create_loss(config: dict) -> CompositeLoss:
    """Factory function to create loss from config"""
    return CompositeLoss(**config)


if __name__ == "__main__":
    # Quick test
    loss_fn = CompositeLoss()
    pred = torch.randn(2, 1, 256, 256)
    target = torch.randn(2, 1, 256, 256)
    losses = loss_fn(pred, target)
    for k, v in losses.items():
        print(f"{k}: {v.item():.4f}")