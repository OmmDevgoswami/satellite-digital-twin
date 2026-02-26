"""
02_verify_dataloader.py
-----------------------
Phase 3: Verify the Dataset and DataLoader work correctly.

This script confirms:
  1. Records load correctly from training.json
  2. Train/Val/Test split is stratified (proportional class balance)
  3. Images and labels come out in the right format
  4. Augmentations are visible (saved as a PNG grid)
  5. WeightedSampler makes dumps appear roughly 50% of the time

Run from project root:
    conda activate dump_detect
    python notebooks/02_verify_dataloader.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset import load_records, split_records, AerialWasteDataset, make_weighted_sampler
from src.data.transforms import get_train_transforms, get_val_transforms
from src.utils.config import TRAIN_JSON, NORM_MEAN, NORM_STD, BATCH_SIZE

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eda_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NORM_MEAN = np.array(NORM_MEAN)
NORM_STD  = np.array(NORM_STD)


def denormalize(tensor):
    """Reverse ImageNet normalization for display."""
    img = tensor.permute(1, 2, 0).numpy()   # CHW → HWC
    img = img * NORM_STD + NORM_MEAN
    return np.clip(img, 0, 1)


print("\n" + "█" * 55)
print("  Phase 3: DataLoader Verification")
print("█" * 55)

# ── Step 1: Load records ───────────────────────────────────────────────────
print("\n[1] Loading records ...")
records = load_records(TRAIN_JSON)
print(f"    Total on-disk records : {len(records)}")
print(f"    Dump (label=1)        : {sum(r['label'] for r in records)}")
print(f"    No-dump (label=0)     : {sum(1 for r in records if r['label'] == 0)}")

# ── Step 2: Stratified split ───────────────────────────────────────────────
print("\n[2] Splitting records ...")
train_recs, val_recs, test_recs = split_records(records)

# ── Step 3: Create datasets ────────────────────────────────────────────────
print("\n[3] Creating datasets ...")
train_ds = AerialWasteDataset(train_recs, mode="clf", transform=get_train_transforms())
val_ds   = AerialWasteDataset(val_recs,   mode="clf", transform=get_val_transforms())
test_ds  = AerialWasteDataset(test_recs,  mode="clf", transform=get_val_transforms())
print(f"    Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

# ── Step 4: Inspect one item ───────────────────────────────────────────────
print("\n[4] Inspecting a single sample ...")
img_tensor, label = train_ds[0]
print(f"    Image tensor shape  : {img_tensor.shape}")   # expect [3, 256, 256]
print(f"    Image dtype         : {img_tensor.dtype}")   # expect torch.float32
print(f"    Image value range   : [{img_tensor.min():.2f}, {img_tensor.max():.2f}]")
print(f"    Label               : {label.item()} ({'dump' if label.item()==1 else 'no-dump'})")

# ── Step 5: Visualise augmented batch ─────────────────────────────────────
print("\n[5] Visualising 8 augmented training samples ...")
n_show = 8
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

indices = list(range(min(n_show, len(train_ds))))
for i, idx in enumerate(indices):
    img_t, lbl = train_ds[idx]
    img_np     = denormalize(img_t)
    axes[i].imshow(img_np)
    axes[i].set_title(f"{'DUMP' if lbl==1 else 'No-Dump'}", fontsize=10,
                      color="red" if lbl==1 else "green", fontweight="bold")
    axes[i].axis("off")

plt.suptitle("Augmented Training Samples (each re-run looks different!)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "augmented_samples.png")
plt.savefig(out, dpi=90, bbox_inches="tight")
plt.close()
print(f"    [SAVED] {out}")

# ── Step 6: Check weighted sampler balance ─────────────────────────────────
print("\n[6] Checking WeightedSampler class balance ...")
from torch.utils.data import DataLoader
sampler     = make_weighted_sampler(train_recs)
loader      = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
batch_imgs, batch_labels = next(iter(loader))
dump_frac   = batch_labels.float().mean().item()
print(f"    Batch size          : {len(batch_labels)}")
print(f"    Dump fraction       : {dump_frac:.2f}  (ideal ~0.50)")
print(f"    Image batch shape   : {batch_imgs.shape}")  # [B, 3, 256, 256]

print("\n" + "█" * 55)
print("  ✅ Phase 3 Verification Complete!")
print("  If dump fraction is near 0.5, the sampler is working.")
print("█" * 55)
