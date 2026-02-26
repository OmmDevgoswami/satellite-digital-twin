"""
train_segmentation.py
---------------------
Phase 5: Train the U-Net segmentation model on AerialWaste.

Dataset strategy:
  - testing.json contains 166 polygon-annotated images
  - We split these 166 into train/val/test for segmentation training
  - Masks are generated dynamically from polygon coordinates

Loss function: DiceLoss + BCEWithLogitsLoss (combined)
  WHY combined?
    - BCE alone doesn't penalise shape errors well
    - Dice alone is unstable early in training
    - Their combination converges faster and produces sharper masks

Metrics: IoU (Intersection over Union) + Dice coefficient

Run from project root:
    conda activate dump_detect
    python src/training/train_segmentation.py

Expected time: ~3-5 min/epoch on CPU × ~15 epochs ≈ 45-75 min total
"""

import os, sys, time, csv, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.dataset          import load_records, AerialWasteDataset
from src.data.transforms       import get_train_transforms, get_val_transforms
from src.models.segmentation_model import get_segmentation_model
from src.training.metrics      import batch_iou, batch_dice
from src.utils.config          import (DEVICE, CHECKPOINT_DIR, TEST_JSON,
                                       TRAIN_SPLIT, VAL_SPLIT, RANDOM_SEED,
                                       BATCH_SIZE)

SEG_BATCH = 8     # smaller batch for segmentation (higher resolution features)


# ── Build segmentation-specific splits from the 166 annotated images ──────
def get_seg_splits(seed=RANDOM_SEED):
    """
    Load and split the 166 polygon-annotated images from testing.json.

    WHY testing.json for segmentation TRAINING?
      training.json has NO polygon masks. testing.json has 166 annotated
      images. We split those 166 into:
        70% train (≈116) | 15% val (≈25) | 15% test (≈25)
    """
    records = load_records(TEST_JSON, require_ondisk=True)
    seg_records = [r for r in records if len(r["polygons"]) > 0]
    print(f"  Segmentation-eligible records : {len(seg_records)}")

    random.seed(seed)
    random.shuffle(seg_records)
    n     = len(seg_records)
    n_tr  = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)
    train = seg_records[:n_tr]
    val   = seg_records[n_tr:n_tr + n_val]
    test  = seg_records[n_tr + n_val:]
    print(f"  Seg split — Train:{len(train)} | Val:{len(val)} | Test:{len(test)}")
    return train, val, test


# ── Combined Dice + BCE Loss ──────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE loss for binary segmentation.

    WHY?
      BCE alone treats each pixel independently — it doesn't care about
      connected dump regions. Dice loss maximises overlap between predicted
      and true mask — it penalises fragmented or undersized predictions.
      Together they balance pixel-level accuracy and region-level coverage.
    """

    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_w = dice_weight
        self.bce_w  = bce_weight
        self.bce    = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs    = torch.sigmoid(logits)
        smooth   = 1e-6
        inter    = (probs * targets).sum(dim=(2, 3))
        dice_loss = 1 - (2 * inter + smooth) / (
            probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth)
        dice_loss = dice_loss.mean()

        return self.dice_w * dice_loss + self.bce_w * bce_loss


# ── Training loop ─────────────────────────────────────────────────────────
def train_one_epoch_seg(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_iou, total_dice = 0.0, 0.0, 0.0

    for images, masks in loader:
        images = images.to(DEVICE)
        masks  = masks.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_iou  += batch_iou(logits, masks)  * images.size(0)
        total_dice += batch_dice(logits, masks) * images.size(0)

    n = len(loader.dataset)
    return {"loss": total_loss/n, "iou": total_iou/n, "dice": total_dice/n}


def evaluate_seg(model, loader, criterion):
    model.eval()
    total_loss, total_iou, total_dice = 0.0, 0.0, 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks  = masks.to(DEVICE)
            logits = model(images)
            loss   = criterion(logits, masks)
            total_loss += loss.item() * images.size(0)
            total_iou  += batch_iou(logits, masks)  * images.size(0)
            total_dice += batch_dice(logits, masks) * images.size(0)

    n = len(loader.dataset)
    return {"loss": total_loss/n, "iou": total_iou/n, "dice": total_dice/n}


# ── Visualise predictions ─────────────────────────────────────────────────
def visualize_predictions(model, dataset, n=4, save_path=None):
    """
    Save a grid of image | ground-truth mask | predicted mask for n samples.
    WHY? Visual inspection is the most intuitive way to evaluate segmentation.
    """
    model.eval()
    MEAN = np.array([0.485, 0.456, 0.406])
    STD  = np.array([0.229, 0.224, 0.225])

    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    with torch.no_grad():
        for row, idx in enumerate(indices):
            img_t, mask_t = dataset[idx]
            img_input = img_t.unsqueeze(0).to(DEVICE)
            pred_logit = model(img_input)
            pred_mask  = torch.sigmoid(pred_logit).squeeze().cpu().numpy()

            img_np   = img_t.permute(1, 2, 0).numpy()
            img_np   = np.clip(img_np * STD + MEAN, 0, 1)
            mask_np  = mask_t.squeeze().numpy()

            iou = batch_iou(pred_logit, mask_t.unsqueeze(0).to(DEVICE))

            axes[row][0].imshow(img_np)
            axes[row][0].set_title("Satellite Image", fontsize=9)
            axes[row][0].axis("off")

            axes[row][1].imshow(mask_np, cmap="Reds", vmin=0, vmax=1)
            axes[row][1].set_title("Ground Truth Mask", fontsize=9)
            axes[row][1].axis("off")

            axes[row][2].imshow(pred_mask, cmap="Reds", vmin=0, vmax=1)
            axes[row][2].set_title(f"Predicted Mask (IoU={iou:.3f})", fontsize=9)
            axes[row][2].axis("off")

    plt.suptitle("U-Net Segmentation Predictions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=90, bbox_inches="tight")
        print(f"  [SAVED] Predictions → {save_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print("\n" + "█" * 60)
    print("  Phase 5: Training U-Net Segmentation Model")
    print("█" * 60)

    # ── Data ───────────────────────────────────────────────────────────────
    print("\n[1] Building segmentation datasets ...")
    train_recs, val_recs, test_recs = get_seg_splits()

    train_ds = AerialWasteDataset(train_recs, mode="seg",
                                  transform=get_train_transforms())
    val_ds   = AerialWasteDataset(val_recs,   mode="seg",
                                  transform=get_val_transforms())
    test_ds  = AerialWasteDataset(test_recs,  mode="seg",
                                  transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=SEG_BATCH,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=SEG_BATCH,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=SEG_BATCH,
                              shuffle=False, num_workers=0)

    print(f"  Batches — Train:{len(train_loader)} | Val:{len(val_loader)} | Test:{len(test_loader)}")

    # ── Model ──────────────────────────────────────────────────────────────
    print("\n[2] Building model ...")
    model = get_segmentation_model()

    # ── Training setup ─────────────────────────────────────────────────────
    criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    patience_cnt  = 0
    patience      = 7
    n_epochs      = 30
    history       = {"train_loss": [], "val_loss": [],
                     "train_iou": [],  "val_iou": [],
                     "train_dice": [], "val_dice": []}

    save_path = os.path.join(CHECKPOINT_DIR, "best_segmentation.pt")
    print(f"\n[3] Training for up to {n_epochs} epochs ...")
    print(f"\n  {'Epoch':>5}  {'Tr Loss':>8}  {'Tr IoU':>7}  "
          f"{'Va Loss':>8}  {'Va IoU':>7}  {'LR':>10}")
    print("  " + "-" * 60)

    for epoch in range(1, n_epochs + 1):
        t0  = time.time()
        tr  = train_one_epoch_seg(model, train_loader, optimizer, criterion)
        val = evaluate_seg(model, val_loader, criterion)
        scheduler.step(val["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        for k in ["train_loss", "val_loss", "train_iou", "val_iou",
                  "train_dice", "val_dice"]:
            src = k.replace("train_", "").replace("val_", "")
            prefix = "train" if "train" in k else "val"
            history[k].append(tr[src] if prefix == "train" else val[src])

        dt = time.time() - t0
        print(f"  {epoch:>5}  {tr['loss']:>8.4f}  {tr['iou']:>7.4f}  "
              f"{val['loss']:>8.4f}  {val['iou']:>7.4f}  "
              f"{current_lr:>10.1e}  ({dt:.0f}s)")

        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            patience_cnt  = 0
            torch.save(model.state_dict(), save_path)
            print(f"          ✓ Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    # ── Save history ───────────────────────────────────────────────────────
    csv_path = os.path.join(CHECKPOINT_DIR, "seg_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch"] + list(history.keys()))
        writer.writeheader()
        for i in range(len(history["train_loss"])):
            row = {"epoch": i + 1}
            row.update({k: history[k][i] for k in history})
            writer.writerow(row)

    # ── Learning curves ────────────────────────────────────────────────────
    print("\n[4] Plotting learning curves ...")
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train", markersize=4)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val",   markersize=4)
    axes[0].set_title("Loss (Dice + BCE)")
    axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_iou"],  "b-o", label="Train IoU", markersize=4)
    axes[1].plot(epochs, history["val_iou"],    "r-o", label="Val IoU",   markersize=4)
    axes[1].set_title("IoU Score")
    axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle("Segmentation Training — Learning Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    curve_path = os.path.join(CHECKPOINT_DIR, "seg_learning_curves.png")
    plt.savefig(curve_path, dpi=90, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {curve_path}")

    # ── Test evaluation ────────────────────────────────────────────────────
    print("\n[5] Test Set Evaluation ...")
    model.load_state_dict(torch.load(save_path, map_location=DEVICE,
                                     weights_only=True))
    test_metrics = evaluate_seg(model, test_loader, criterion)
    print("\n" + "=" * 55)
    print("  TEST SET RESULTS (Segmentation)")
    print("=" * 55)
    print(f"  Loss      : {test_metrics['loss']:.4f}")
    print(f"  IoU       : {test_metrics['iou']:.4f}")
    print(f"  Dice      : {test_metrics['dice']:.4f}")

    # ── Visual predictions ─────────────────────────────────────────────────
    print("\n[6] Saving visual predictions ...")
    pred_path = os.path.join(CHECKPOINT_DIR, "seg_predictions.png")
    visualize_predictions(model, test_ds, n=4, save_path=pred_path)

    print("\n" + "█" * 60)
    print("  Phase 5 Complete!")
    print(f"  Best model : {save_path}")
    print(f"  Curves     : {curve_path}")
    print(f"  Predictions: {pred_path}")
    print("█" * 60)


if __name__ == "__main__":
    main()
