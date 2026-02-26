"""
segmentation_model.py
---------------------
Segmentation models using segmentation-models-pytorch.

Supported architectures:
  - Unet        : classic encoder-decoder, great baseline
  - UnetPlusPlus: nested skip connections → sharper boundaries
  - FPN         : Feature Pyramid Network → multi-scale awareness
  - DeepLabV3Plus: atrous convolutions → wide field-of-view without losing resolution

WHY multiple architectures?
  No single model wins on every satellite image:
  - UNet++ excels at small, irregular dump patches
  - FPN handles dumps of wildly varying sizes
  - DeepLabV3+ captures global context (surrounding land use)
  Ensemble at inference time combines all three strengths.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from src.utils.config import DEVICE, SEG_ARCH, SEG_ENCODER, SEG_WEIGHTS


_ARCH_MAP = {
    "Unet":          smp.Unet,
    "UnetPlusPlus":  smp.UnetPlusPlus,
    "FPN":           smp.FPN,
    "DeepLabV3Plus": smp.DeepLabV3Plus,
}


def get_segmentation_model(
    arch: str = SEG_ARCH,
    encoder: str = SEG_ENCODER,
    encoder_weights: str = SEG_WEIGHTS,
) -> nn.Module:
    """
    Build and return a segmentation model.

    Args:
        arch            : 'Unet', 'UnetPlusPlus', 'FPN', 'DeepLabV3Plus'
        encoder         : backbone name ('resnet34', 'resnet50', etc.)
        encoder_weights : pretrained weights ('imagenet' or None)

    Returns:
        model on DEVICE with single-channel binary output
    """
    if arch not in _ARCH_MAP:
        raise ValueError(f"Unknown arch '{arch}'. Choose from: {list(_ARCH_MAP)}")

    model_cls = _ARCH_MAP[arch]
    model = model_cls(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,        # binary mask
        activation=None,  # raw logits
    )

    model = model.to(DEVICE)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Segmentation Model : {arch} + {encoder} ({encoder_weights})")
    print(f"  Device             : {DEVICE}")
    print(f"  Params             : {total:,} total | {trainable:,} trainable")
    return model
