"""
config.py
---------
Central configuration file for the Garbage Dump Detection project.
WHY: Having one place for all project settings avoids 'magic numbers'
     scattered across files and makes experiments easy to reproduce.

AerialWaste Dataset Notes:
  - training.json: image-level metadata only (no polygon masks)
  - testing.json:  166 images have polygon segmentation masks
  - ALL images are from dump sites; labeling uses 'categories' field:
      len(categories) > 0  →  CONFIRMED DUMP   (label = 1)
      len(categories) == 0 →  CANDIDATE / NONE (label = 0)
  - Images live in: data/raw/AerialWaste/images/images0/
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT_DIR        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW_DIR    = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROC_DIR   = os.path.join(ROOT_DIR, "data", "processed")
OUTPUT_DIR      = os.path.join(ROOT_DIR, "outputs")
PRED_DIR        = os.path.join(OUTPUT_DIR, "predictions")
CHECKPOINT_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")

# ── AerialWaste specific ──────────────────────────────────────────────────
AERIAL_ROOT     = os.path.join(DATA_RAW_DIR, "AerialWaste")
# Images are nested inside images/images0/ after extraction
# Support multiple zip folders: images0, images1, images2, etc.
IMAGES_DIR      = os.path.join(AERIAL_ROOT, "images", "images0")
TRAIN_JSON      = os.path.join(AERIAL_ROOT, "training.json")
TEST_JSON       = os.path.join(AERIAL_ROOT, "testing.json")

# ── Multi-dataset support ─────────────────────────────────────────────────
# List of all image directories to search (AerialWaste + any other datasets)
# Automatically discovers all images*/ subfolders in AerialWaste/images/
def get_all_image_dirs():
    """Return list of all image directories to search for images."""
    dirs = []
    aerial_images_root = os.path.join(AERIAL_ROOT, "images")
    if os.path.exists(aerial_images_root):
        # Find all images0, images1, images2, etc. folders
        for item in os.listdir(aerial_images_root):
            item_path = os.path.join(aerial_images_root, item)
            if os.path.isdir(item_path) and item.startswith("images"):
                dirs.append(item_path)
    # If no subfolders found, fall back to default
    if not dirs:
        dirs = [IMAGES_DIR] if os.path.exists(IMAGES_DIR) else []
    return dirs

ALL_IMAGE_DIRS = get_all_image_dirs()

# ── Dataset Settings ──────────────────────────────────────────────────────
IMAGE_SIZE      = (256, 256)       # All images resized to this (H, W)
CLASSES         = ["no_dump", "dump"]   # Binary classification labels
NUM_CLASSES     = 1                # For segmentation (binary mask)

# ── Training Hyperparameters ─────────────────────────────────────────────
BATCH_SIZE      = 16
NUM_EPOCHS      = 30
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-5
TRAIN_SPLIT     = 0.70
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.15
RANDOM_SEED     = 42

# ── Model Settings ────────────────────────────────────────────────────────
CLF_MODEL_NAME  = "resnet34"
SEG_ARCH        = "Unet"
SEG_ENCODER     = "resnet34"
SEG_WEIGHTS     = "imagenet"

# ── Normalization ─────────────────────────────────────────────────────────
NORM_MEAN       = [0.485, 0.456, 0.406]
NORM_STD        = [0.229, 0.224, 0.225]

# ── Device ────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def print_config():
    print("=" * 50)
    print("  Project Configuration")
    print("=" * 50)
    print(f"  ROOT DIR    : {ROOT_DIR}")
    print(f"  IMAGES DIR  : {IMAGES_DIR}")
    print(f"  ALL IMAGE DIRS ({len(ALL_IMAGE_DIRS)}):")
    for d in ALL_IMAGE_DIRS:
        print(f"    - {d}")
    print(f"  IMAGE SIZE  : {IMAGE_SIZE}")
    print(f"  BATCH SIZE  : {BATCH_SIZE}")
    print(f"  EPOCHS      : {NUM_EPOCHS}")
    print(f"  LR          : {LEARNING_RATE}")
    print(f"  DEVICE      : {DEVICE}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
