"""
Training Script for KLA Semiconductor Image Restoration
Supports mixed precision, cosine annealing, wandb logging, checkpointing.
"""
import os
import argparse
import yaml
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import NAFNet, CompositeLoss, create_model, create_model_from_config, create_loss
from data import create_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='Train NAFNet for Semiconductor Image Restoration')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to config file')
    parser.add_argument('--data_root', type=str, required=True, help='Path to training data root')
    parser.add_argument('--output_dir', type=str, default='weights', help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='kla-restoration', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EMA:
    """Exponential moving average of model weights (evaluated separately;
    often a free quality bump at test time)."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    best_val_loss: float,
    config: Dict,
    path: Path,
    ema: 'EMA' = None,
    use_ema: bool = False
):
    """Save training checkpoint. When use_ema=True and an EMA is provided,
    the saved model_state_dict is the EMA (averaged) weights."""
    state_dict = ema.shadow if (use_ema and ema is not None) else model.state_dict()
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'best_val_loss': best_val_loss,
        'uses_ema': use_ema and ema is not None,
        'config': config
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer = None,
    scheduler = None,
    scaler: GradScaler = None,
    device: str = 'cpu'
) -> tuple:
    """Load training checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return start_epoch, best_val_loss


def _forward_with_aux(model: nn.Module, x: torch.Tensor):
    """Runs the model; returns (restored, logvar) where logvar is None for
    models without an uncertainty head (plain NAFNet etc.)."""
    if hasattr(model, 'forward_with_aux'):
        restored, logvar, _sigmas = model.forward_with_aux(x)
        return restored, logvar
    return model(x), None


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: str,
    epoch: int,
    log_interval: int = 50
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    loss_components = {}

    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for i, batch in enumerate(pbar):
        degraded = batch['degraded'].to(device, non_blocking=True)
        ground_truth = batch['ground_truth'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=scaler is not None):
            restored, logvar = _forward_with_aux(model, degraded)
            losses = criterion(restored, ground_truth, logvar=logvar)
            loss = losses['total']

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if ema is not None:
            ema.update(model)

        # Accumulate losses
        batch_size = degraded.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        for k, v in losses.items():
            if k != 'total':
                loss_components[k] = loss_components.get(k, 0.0) + v.item() * batch_size

        if i % log_interval == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Average losses
    avg_loss = total_loss / total_samples
    avg_components = {k: v / total_samples for k, v in loss_components.items()}
    avg_components['total'] = avg_loss

    return avg_components


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Dict[str, float]:
    """Validate model"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    loss_components = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc='Validation'):
            degraded = batch['degraded'].to(device, non_blocking=True)
            ground_truth = batch['ground_truth'].to(device, non_blocking=True)

            restored, logvar = _forward_with_aux(model, degraded)
            losses = criterion(restored, ground_truth, logvar=logvar)

            batch_size = degraded.size(0)
            total_loss += losses['total'].item() * batch_size
            total_samples += batch_size

            for k, v in losses.items():
                if k != 'total':
                    loss_components[k] = loss_components.get(k, 0.0) + v.item() * batch_size

    avg_loss = total_loss / total_samples
    avg_components = {k: v / total_samples for k, v in loss_components.items()}
    avg_components['total'] = avg_loss

    return avg_components


def main():
    args = parse_args()
    config = load_config(args.config)

    # Merge CLI args into config
    config['data_root'] = args.data_root
    config['output_dir'] = args.output_dir
    config['device'] = args.device
    config['use_wandb'] = args.wandb
    config['wandb_project'] = args.wandb_project
    config['wandb_run_name'] = args.wandb_run_name
    config['seed'] = args.seed

    # Setup
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Wandb
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=config
        )

    # Data
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=config.get('batch_size', 16),
        num_workers=config.get('num_workers', 4),
        patch_size=config.get('patch_size', 256),
        scale=config.get('scale', 2),
        val_split=config.get('val_split', 0.1),
        cache=config.get('cache', False),
        synth_ratio=config.get('synth_ratio', 0.0)
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    print("Creating model...")
    model = create_model_from_config(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M")

    # Loss
    criterion = create_loss(config.get('loss', {})).to(device)

    # Optimizer (float() guards: YAML needs a decimal point for scientific
    # notation, otherwise values like 2e-4 arrive as strings)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config.get('lr', 2e-4)),
        weight_decay=float(config.get('weight_decay', 1e-4)),
        betas=(0.9, 0.999)
    )

    # Scheduler
    epochs = int(config.get('epochs', 50))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(config.get('min_lr', 1e-6))
    )

    # Mixed precision
    scaler = GradScaler() if config.get('mixed_precision', True) and device.type == 'cuda' else None

    # EMA of weights (validated separately; best checkpoint may use it)
    ema_decay = config.get('ema_decay', 0.999)
    ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        print(f"Resuming from {args.resume}")
        start_epoch, best_val_loss = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, scaler, device
        )
        print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")

    # Training loop
    print(f"Starting training from epoch {start_epoch} to {epochs}")
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        # Train
        train_losses = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch,
            log_interval=config.get('log_interval', 50)
        )

        # Validate
        val_losses = validate(model, val_loader, criterion, device)

        # Also validate the EMA weights; best checkpoint = better of the two
        use_ema_for_best = False
        if ema is not None:
            raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema.shadow)
            ema_val_losses = validate(model, val_loader, criterion, device)
            model.load_state_dict(raw_state)
            if ema_val_losses['total'] < val_losses['total']:
                val_losses = ema_val_losses
                use_ema_for_best = True

        # Scheduler step
        scheduler.step()

        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time

        log_msg = (f"Epoch {epoch}/{epochs} | "
                   f"Train: {train_losses['total']:.4f} | "
                   f"Val: {val_losses['total']:.4f} | "
                   f"LR: {current_lr:.2e} | "
                   f"Time: {epoch_time:.1f}s")
        print(log_msg)

        if args.wandb:
            wandb.log({
                'epoch': epoch,
                'train/total_loss': train_losses['total'],
                'val/total_loss': val_losses['total'],
                'lr': current_lr,
                **{f'train/{k}': v for k, v in train_losses.items() if k != 'total'},
                **{f'val/{k}': v for k, v in val_losses.items() if k != 'total'}
            })

        # Save checkpoint
        is_best = val_losses['total'] < best_val_loss
        if is_best:
            best_val_loss = val_losses['total']
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_val_loss, config,
                output_dir / 'best.pt', ema=ema, use_ema=use_ema_for_best
            )
            ema_tag = " (EMA weights)" if use_ema_for_best else ""
            print(f"  -> New best model saved!{ema_tag} Val loss: {best_val_loss:.4f}")

        # Save latest
        save_checkpoint(
            model, optimizer, scheduler, scaler, epoch, best_val_loss, config,
            output_dir / 'latest.pt'
        )

        # Periodic save
        if (epoch + 1) % config.get('save_interval', 10) == 0:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, best_val_loss, config,
                output_dir / f'epoch_{epoch}.pt'
            )

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()