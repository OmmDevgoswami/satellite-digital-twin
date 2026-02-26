"""
transforms.py
-------------
Augmentation pipelines for satellite imagery (Albumentations + TTA).

WHY satellite-specific additions?
  - CLAHE   : Contrast Limited Adaptive Histogram Equalization — improves
    contrast in hazy / low-light satellite scenes without over-amplifying noise.
  - GridDistortion / ElasticTransform : simulate geometric distortions from
    satellite sensor perspective and atmospheric lensing.
  - RandomShadow : some dump sites are partially shadowed by buildings/trees.
  - Sharpen / Emboss : can help the model learn edge features for dump borders.

TTA (Test-Time Augmentation):
  At inference, run 5 slightly different views of the same image through the
  model and average the outputs. Reduces prediction variance significantly.
  WHY? A classifier confident on the original image AND its flipped / rotated
  versions is much less likely to be fooling itself on noise artefacts.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.utils.config import IMAGE_SIZE, NORM_MEAN, NORM_STD

H, W = IMAGE_SIZE


def get_train_transforms() -> A.Compose:
    """
    Strong satellite-specific augmentations for training.
    """
    return A.Compose([
        A.Resize(H, W),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.3),

        # ─ Photometric ──────────────────────────────────────────────────────
        A.RandomBrightnessContrast(brightness_limit=0.25,
                                   contrast_limit=0.25, p=0.5),
        A.HueSaturationValue(hue_shift_limit=12,
                             sat_shift_limit=25,
                             val_shift_limit=12, p=0.35),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.35),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),

        # ─ Noise / blur ─────────────────────────────────────────────────────
        A.GaussNoise(std_range=(0.04, 0.20), p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.MotionBlur(blur_limit=5, p=0.15),

        # ─ Geometric distortion ─────────────────────────────────────────────
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.25),
        A.ElasticTransform(alpha=60, sigma=6, p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15, border_mode=0, p=0.3),

        # ─ Occlusion ────────────────────────────────────────────────────────
        A.CoarseDropout(num_holes_range=(1, 8),
                        hole_height_range=(16, 32),
                        hole_width_range=(16, 32),
                        fill=0, p=0.25),

        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    """
    Minimal transforms for validation and test sets (no randomness).
    """
    return A.Compose([
        A.Resize(H, W),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])


# Alias — test uses the same pipeline as val
get_test_transforms = get_val_transforms


def get_tta_transforms() -> list:
    """
    Test-Time Augmentation: returns 5 deterministic transform pipelines
    to run on the same image, results averaged for final prediction.

    TTA variants:
      0 — Original (no flip)
      1 — Horizontal flip
      2 — Vertical flip
      3 — 90° rotation
      4 — Horizontal + Vertical flip
    """
    base = [A.Resize(H, W), A.Normalize(mean=NORM_MEAN, std=NORM_STD), ToTensorV2()]
    return [
        A.Compose(base),
        A.Compose([A.Resize(H, W), A.HorizontalFlip(p=1.0),
                   A.Normalize(mean=NORM_MEAN, std=NORM_STD), ToTensorV2()]),
        A.Compose([A.Resize(H, W), A.VerticalFlip(p=1.0),
                   A.Normalize(mean=NORM_MEAN, std=NORM_STD), ToTensorV2()]),
        A.Compose([A.Resize(H, W), A.RandomRotate90(p=1.0),
                   A.Normalize(mean=NORM_MEAN, std=NORM_STD), ToTensorV2()]),
        A.Compose([A.Resize(H, W), A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0),
                   A.Normalize(mean=NORM_MEAN, std=NORM_STD), ToTensorV2()]),
    ]
