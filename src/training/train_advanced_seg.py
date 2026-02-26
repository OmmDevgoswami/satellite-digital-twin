"""
train_advanced_seg.py
---------------------
Train 3 advanced segmentation architectures beyond the baseline U-Net:
  1. UNet++   (UnetPlusPlus) — nested skip connections for sharp masks
  2. FPN      (Feature Pyramid Network) — multi-scale dump detection
  3. DeepLabV3+ — atrous convolutions for wide context

WHY train all three?
  Each architecture captures different spatial patterns:
  - UNet++ excels at precise boundary delineation for irregular dumps
  - FPN handles dumps at varied scales (tiny roadside vs. massive landfill)
  - DeepLabV3+ uses dilated convolutions to see surrounding land-use context

The best model (by val IoU) is saved as best_advanced_seg.pt for ensemble use.

Run from project root:
    conda activate dump_detect
    python src/training/train_advanced_seg.py
"""

import os, sys, time, csv, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.dataset            import load_records, AerialWasteDataset
from src.data.transforms         import get_train_transforms, get_val_transforms
from src.models.segmentation_model import get_segmentation_model
from src.training.metrics        import batch_iou, batch_dice
from src.training.train_segmentation import get_seg_splits, DiceBCELoss
from src.utils.config            import (DEVICE, CHECKPOINT_DIR, RANDOM_SEED)

SEG_BATCH = 4     # smaller batch — DeepLabV3+ needs more GPU/CPU memory


# ── Combined Tversky + BCE Loss (better than Dice for small objects) ──────────
class TverskyBCELoss(nn.Module):
    """
    Tversky loss penalises false negatives more than false positives.
    Critical for tiny dump patches we must not miss (high recall).

    Tversky index: T = TP / (TP + alpha*FP + beta*FN)
    With alpha=0.3, beta=0.7  →  recall-focused (FN costs more).
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 bce_weight: float = 0.4):
        super().__init__()
        self.alpha   = alpha
        self.beta    = beta
        self.bce_w   = bce_weight
        self.tversky_w = 1 - bce_weight
        self.bce     = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs  = torch.sigmoid(logits)
        smooth = 1e-6
        tp = (probs * targets).sum(dim=(2, 3))
        fp = (probs * (1 - targets)).sum(dim=(2, 3))
        fn = ((1 - probs) * targets).sum(dim=(2, 3))
        tversky = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)
        tversky_loss = (1 - tversky).mean()

        return self.tversky_w * tversky_loss + self.bce_w * bce_loss


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = total_iou = total_dice = 0.0
    for images, masks in loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_iou  += batch_iou(logits, masks)  * images.size(0)
        total_dice += batch_dice(logits, masks) * images.size(0)
    n = len(loader.dataset)
    return {"loss": total_loss/n, "iou": total_iou/n, "dice": total_dice/n}


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = total_iou = total_dice = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            logits = model(images)
            total_loss += criterion(logits, masks).item() * images.size(0)
            total_iou  += batch_iou(logits, masks)  * images.size(0)
            total_dice += batch_dice(logits, masks) * images.size(0)
    n = len(loader.dataset)
    return {"loss": total_loss/n, "iou": total_iou/n, "dice": total_dice/n}


def train_model(arch: str, train_loader, val_loader, test_loader,
                n_epochs: int = 20, patience: int = 6) -> dict:
    """Train one segmentation architecture and save best checkpoint."""
    print(f"\n{'='*55}")
    print(f"  Training: {arch}")
    print(f"{'='*55}")

    model    = get_segmentation_model(arch=arch)
    criterion = TverskyBCELoss(alpha=0.3, beta=0.7, bce_weight=0.4)
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-7)

    save_path     = os.path.join(CHECKPOINT_DIR, f"best_{arch.lower()}.pt")
    best_val_iou  = 0.0
    patience_cnt  = 0
    history       = {"train_loss": [], "val_loss": [],
                     "train_iou":  [], "val_iou":  []}

    print(f"  {'Epoch':>5}  {'TrLoss':>8}  {'TrIoU':>7}  {'VaLoss':>8}  {'VaIoU':>7}")
    print("  " + "-" * 52)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, criterion)
        vl = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(vl["loss"])
        history["train_iou"].append(tr["iou"])
        history["val_iou"].append(vl["iou"])

        dt = time.time() - t0
        print(f"  {epoch:>5}  {tr['loss']:>8.4f}  {tr['iou']:>7.4f}  "
              f"{vl['loss']:>8.4f}  {vl['iou']:>7.4f}  ({dt:.0f}s)")

        if vl["iou"] > best_val_iou:
            best_val_iou = vl["iou"]
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
            print(f"          ✓ Saved (val_IoU={best_val_iou:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    test_m = evaluate(model, test_loader, criterion)
    print(f"\n  Test IoU  : {test_m['iou']:.4f}")
    print(f"  Test Dice : {test_m['dice']:.4f}")

    return {
        "arch":       arch,
        "save_path":  save_path,
        "best_val_iou": best_val_iou,
        "test_iou":   test_m["iou"],
        "test_dice":  test_m["dice"],
        "history":    history,
    }


def main():
    print("\n" + "█" * 60)
    print("  Advanced Segmentation Training (UNet++ / FPN / DeepLabV3+)")
    print("█" * 60)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Data (same splits as baseline seg) ───────────────────────────────────
    print("\n[1] Building segmentation splits ...")
    train_recs, val_recs, test_recs = get_seg_splits()
    train_ds = AerialWasteDataset(train_recs, mode="seg",
                                  transform=get_train_transforms())
    val_ds   = AerialWasteDataset(val_recs,   mode="seg",
                                  transform=get_val_transforms())
    test_ds  = AerialWasteDataset(test_recs,  mode="seg",
                                  transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=SEG_BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=SEG_BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=SEG_BATCH, shuffle=False, num_workers=0)

    # ── Train all architectures ─────────────────────────────────────────────
    archs   = ["UnetPlusPlus", "FPN", "DeepLabV3Plus"]
    results = []
    for arch in archs:
        res = train_model(arch, train_loader, val_loader, test_loader)
        results.append(res)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ADVANCED SEGMENTATION RESULTS SUMMARY")
    print("=" * 60)
    best_iou   = 0.0
    best_arch  = None
    best_path  = None
    for r in results:
        print(f"  {r['arch']:<15} | Test IoU: {r['test_iou']:.4f} "
              f"| Test Dice: {r['test_dice']:.4f} | → {r['save_path']}")
        if r["test_iou"] > best_iou:
            best_iou  = r["test_iou"]
            best_arch = r["arch"]
            best_path = r["save_path"]

    # Always copy best as best_advanced_seg.pt for easy loading
    import shutil
    if best_path:
        dest = os.path.join(CHECKPOINT_DIR, "best_advanced_seg.pt")
        shutil.copy2(best_path, dest)
        print(f"\n  ★ Best architecture: {best_arch} (IoU={best_iou:.4f})")
        print(f"    Copied to: {dest}")

    # Save summary JSON
    import json
    summary = {r["arch"]: {"test_iou": r["test_iou"], "test_dice": r["test_dice"]}
               for r in results}
    json_path = os.path.join(CHECKPOINT_DIR, "advanced_seg_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved → {json_path}")
    print("█" * 60)


if __name__ == "__main__":
    main()
