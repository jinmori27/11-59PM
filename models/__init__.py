from .nafnet import NAFNet, NAFNetLocal
from .diag_nafnet import DiagNAFNet
from .loss import CompositeLoss, HeteroscedasticNLL, create_loss
import torch.nn as nn


def create_model(model_type: str = 'nafnet', scale: int = 2, **kwargs) -> nn.Module:
    """Factory function to create model"""
    if model_type == 'nafnet':
        return NAFNet(scale=scale, **kwargs)
    elif model_type == 'nafnet_local':
        return NAFNetLocal(scale=scale, **kwargs)
    elif model_type == 'diag_nafnet':
        return DiagNAFNet(scale=scale, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_model_from_config(config: dict, model_type_default: str = 'nafnet',
                             scale_default: int = 2) -> nn.Module:
    """Build a model from a train_config.yaml section or a checkpoint's config dict."""
    model_type = config.get('model_type', model_type_default)
    kwargs = dict(
        scale=config.get('scale', scale_default),
        width=config.get('width', 48),
        enc_blks=config.get('enc_blks', [2, 2, 4, 8]),
        middle_blks=config.get('middle_blks', 12),
        dec_blks=config.get('dec_blks', [2, 2, 2, 2]),
    )
    if model_type == 'diag_nafnet':
        for key in ('use_film', 'use_uncertainty', 'embed_dim'):
            if key in config:
                kwargs[key] = config[key]
    return create_model(model_type, **kwargs)


__all__ = [
    'NAFNet', 'NAFNetLocal', 'DiagNAFNet', 'CompositeLoss', 'HeteroscedasticNLL',
    'create_model', 'create_model_from_config', 'create_loss',
]
