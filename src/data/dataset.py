"""
dataset.py
----------
PyTorch Dataset class for the AerialWaste project.

WHY a custom Dataset?
  PyTorch's DataLoader works with any class that implements:
    __len__()         → total number of samples
    __getitem__(idx)  → one (image, label) pair

  We write our own because the AerialWaste JSON format is unique —
  image-level labels come from the 'categories' metadata field,
  not from a standard folder structure.

  For Phase 4 (classification): returns (image_tensor, label)
  For Phase 5 (segmentation):   returns (image_tensor, mask_tensor)
"""

import os
import json
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.utils.config import (
    IMAGES_DIR, TRAIN_JSON, TEST_JSON, ALL_IMAGE_DIRS,
    TRAIN_SPLIT, VAL_SPLIT, RANDOM_SEED, BATCH_SIZE,
)
from src.data.transforms import get_train_transforms, get_val_transforms


# ── Helper: safe int conversion ────────────────────────────────────────────
def _safe_int(val, default=0):
    """Convert to int safely — handles 'n/a', None, empty string."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


# ── Helper: polygon → binary mask ─────────────────────────────────────────
def _polygon_to_mask(polygons, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    for pts in polygons:
        pts_int = np.array(pts, dtype=np.float32).reshape(-1, 2)
        pts_int = pts_int.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts_int], color=1)
    return mask


# ══════════════════════════════════════════════════════════════════════════
# Core Dataset class
# ══════════════════════════════════════════════════════════════════════════
class AerialWasteDataset(Dataset):
    """
    Dataset for both classification and segmentation modes.

    Args:
        records   : list of dicts (from load_records() below)
        mode      : 'clf' → returns (image, label)
                    'seg' → returns (image, mask)
        transform : Albumentations Compose pipeline
    """

    def __init__(self, records: list, mode: str = "clf", transform=None):
        assert mode in ("clf", "seg"), "mode must be 'clf' or 'seg'"
        self.records   = records
        self.mode      = mode
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # ── Load image (search across all image directories) ──────────────
        img_path = None
        for img_dir in ALL_IMAGE_DIRS:
            candidate = os.path.join(img_dir, rec["file_name"])
            if os.path.exists(candidate):
                img_path = candidate
                break
        
        if img_path is None:
            # Try fallback to default IMAGES_DIR
            img_path = os.path.join(IMAGES_DIR, rec["file_name"])
        
        img = cv2.imread(img_path)
        if img is None:
            # Fallback: return a black image if file somehow missing
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ── Classification mode ───────────────────────────────────────────
        if self.mode == "clf":
            label = rec["label"]    # 0 or 1
            if self.transform:
                result = self.transform(image=img)
                img    = result["image"]          # torch.Tensor CHW
            else:
                img = torch.from_numpy(img.transpose(2, 0, 1)).float()
            return img, torch.tensor(label, dtype=torch.long)

        # ── Segmentation mode ─────────────────────────────────────────────
        else:
            h, w = img.shape[:2]
            mask = _polygon_to_mask(rec["polygons"], h, w)

            if self.transform:
                result = self.transform(image=img, mask=mask)
                img    = result["image"]           # CHW tensor
                mask   = result["mask"]            # HW tensor (uint8)
            else:
                img  = torch.from_numpy(img.transpose(2, 0, 1)).float()
                mask = torch.from_numpy(mask)

            # Convert mask to float32 for BCEWithLogitsLoss
            mask = mask.float().unsqueeze(0)       # 1 x H x W
            return img, mask


# ══════════════════════════════════════════════════════════════════════════
# Record loading — reads JSON + filters to on-disk images only
# ══════════════════════════════════════════════════════════════════════════
def load_records(json_path: str, require_ondisk: bool = True) -> list:
    """
    Parse AerialWaste JSON and return list of record dicts.
    If require_ondisk=True, only include images present in IMAGES_DIR.

    WHY filter to on-disk?
      The JSON references all 7 zip files, but we only downloaded images0.
      Filtering ensures __getitem__ never fails with a missing file.
      
      Now supports multiple image directories: searches all folders in
      ALL_IMAGE_DIRS (images0, images1, images2, etc.) to find each image.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Build polygon lookup (used for segmentation mode)
    ann_lookup = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        ann_lookup.setdefault(img_id, [])
        for seg in ann.get("segmentation", []):
            ann_lookup[img_id].append(seg)

    records = []
    for img in data["images"]:
        fname = os.path.basename(img["file_name"])

        if require_ondisk:
            # Search across all image directories
            found = False
            for img_dir in ALL_IMAGE_DIRS:
                if os.path.exists(os.path.join(img_dir, fname)):
                    found = True
                    break
            if not found:
                continue        # skip images we haven't downloaded

        cats  = img.get("categories", [])
        label = 1 if len(cats) > 0 else 0

        records.append({
            "file_name": fname,
            "label":     label,
            "severity":  _safe_int(img.get("severity", 0)),
            "evidence":  _safe_int(img.get("evidence", 0)),
            "polygons":  ann_lookup.get(img["id"], []),
        })

    return records


# ══════════════════════════════════════════════════════════════════════════
# Train / Val / Test split
# ══════════════════════════════════════════════════════════════════════════
def split_records(records: list, seed: int = RANDOM_SEED):
    """
    Split records into train / val / test lists.

    WHY stratified split?
      If we split randomly, the rare 'dump' class might end up mostly in
      one split. We ensure each split has a proportional representation of
      both classes (stratified sampling).
    """
    random.seed(seed)

    dump_recs   = [r for r in records if r["label"] == 1]
    nodump_recs = [r for r in records if r["label"] == 0]

    def _split(lst):
        n      = len(lst)
        n_tr   = int(n * TRAIN_SPLIT)
        n_val  = int(n * VAL_SPLIT)
        random.shuffle(lst)
        return lst[:n_tr], lst[n_tr:n_tr + n_val], lst[n_tr + n_val:]

    tr_d, va_d, te_d = _split(dump_recs)
    tr_n, va_n, te_n = _split(nodump_recs)

    train = tr_d + tr_n;  random.shuffle(train)
    val   = va_d + va_n;  random.shuffle(val)
    test  = te_d + te_n;  random.shuffle(test)

    print(f"  Split — Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"          Train dump: {sum(r['label'] for r in train)} | "
          f"Val dump: {sum(r['label'] for r in val)} | "
          f"Test dump: {sum(r['label'] for r in test)}")
    return train, val, test


# ══════════════════════════════════════════════════════════════════════════
# Weighted sampler — fix class imbalance
# ══════════════════════════════════════════════════════════════════════════
def make_weighted_sampler(records: list) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler so the DataLoader draws dump and
    no-dump samples with equal probability per batch.

    WHY not just oversample?
      Rather than duplicating minority-class images, the sampler assigns
      higher probability to drawing dump images on each iteration.
      This is more memory-efficient and avoids exact duplicates.
    """
    labels    = [r["label"] for r in records]
    n_dump    = sum(labels)
    n_nodump  = len(labels) - n_dump
    # Weight inversely proportional to class frequency
    w_dump    = 1.0 / n_dump    if n_dump   > 0 else 0.0
    w_nodump  = 1.0 / n_nodump  if n_nodump > 0 else 0.0
    weights   = [w_dump if lbl == 1 else w_nodump for lbl in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ══════════════════════════════════════════════════════════════════════════
# Convenience function: build all three DataLoaders at once
# ══════════════════════════════════════════════════════════════════════════
def get_clf_dataloaders(batch_size: int = BATCH_SIZE, num_workers: int = 0):
    """
    Build classification DataLoaders for train / val / test splits.

    Returns:
        (train_loader, val_loader, test_loader)

    WHY num_workers=0 by default?
      Windows has issues with multiprocessing in DataLoader when using
      conda. Workers=0 means the main process loads data (safe on Windows).
      Set to 2-4 on Linux/Mac for speed.
    """
    print("  Loading records from training.json ...")
    records            = load_records(TRAIN_JSON)
    train_recs, val_recs, test_recs = split_records(records)

    train_ds = AerialWasteDataset(train_recs, mode="clf",
                                  transform=get_train_transforms())
    val_ds   = AerialWasteDataset(val_recs,   mode="clf",
                                  transform=get_val_transforms())
    test_ds  = AerialWasteDataset(test_recs,  mode="clf",
                                  transform=get_val_transforms())

    sampler  = make_weighted_sampler(train_recs)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers,
                              pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
