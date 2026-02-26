"""
trainer.py
----------
Training and evaluation loops for the classification model.

WHY separate train/eval loops?
  During training: model is in train mode (dropout active, BatchNorm uses
  running stats). During evaluation: dropout is OFF, BatchNorm uses
  accumulated stats. Calling model.eval() / model.train() toggles this.

Key concepts used here:
  - BCEWithLogitsLoss: numerically stable binary cross-entropy.
    Internally applies sigmoid, so the model outputs raw logits.
  - pos_weight: penalises misclassifying dumps more heavily.
    With 9.3x imbalance → pos_weight=9.3 makes each dump error
    count 9.3× more than a no-dump error.
  - torch.no_grad(): skips building the computation graph during eval,
    saving memory and speeding up inference.
"""

import os
import time
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils.config import DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, CHECKPOINT_DIR
from src.training.metrics import clf_metrics


# ══════════════════════════════════════════════════════════════════════════
# Training loop — one epoch
# ══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, epoch: int) -> dict:
    """
    Run one full pass over the training DataLoader.

    Returns:
        dict with 'loss', 'accuracy', 'f1', 'precision', 'recall'
    """
    model.train()
    running_loss  = 0.0
    all_labels, all_preds = [], []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).float().unsqueeze(1)   # [B,1]

        optimizer.zero_grad()
        logits = model(images)                            # [B,1] raw logits
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Convert logits → predictions (threshold at 0.5)
        probs  = torch.sigmoid(logits).detach().cpu()
        preds  = (probs >= 0.5).int().squeeze(1).tolist()
        lbls   = labels.int().detach().cpu().squeeze(1).tolist()
        all_preds.extend(preds)
        all_labels.extend(lbls)

    avg_loss = running_loss / len(loader.dataset)
    metrics  = clf_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    return metrics


# ══════════════════════════════════════════════════════════════════════════
# Evaluation loop
# ══════════════════════════════════════════════════════════════════════════

def evaluate(model, loader, criterion) -> dict:
    """
    Evaluate model on a DataLoader (val or test).
    No gradient computation — faster and memory-efficient.

    Returns:
        dict with 'loss', 'accuracy', 'f1', 'precision', 'recall'
    """
    model.eval()
    running_loss  = 0.0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)

            logits = model(images)
            loss   = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).cpu()
            preds = (probs >= 0.5).int().squeeze(1).tolist()
            lbls  = labels.int().cpu().squeeze(1).tolist()
            all_preds.extend(preds)
            all_labels.extend(lbls)

    avg_loss = running_loss / len(loader.dataset)
    metrics  = clf_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    return metrics


# ══════════════════════════════════════════════════════════════════════════
# Full training run
# ══════════════════════════════════════════════════════════════════════════

def run_training(model, train_loader, val_loader,
                 n_epochs: int  = NUM_EPOCHS,
                 lr: float      = LEARNING_RATE,
                 pos_weight_val: float = None,
                 save_name: str = "best_classifier.pt") -> dict:
    """
    Full training loop with:
      - Adam optimiser
      - ReduceLROnPlateau scheduler (halves LR if val_loss stagnates)
      - Early stopping (patience=7)
      - Best model checkpoint saving

    Args:
        pos_weight_val : scalar weight for positive (dump) class.
                         None → compute automatically from dataset.

    Returns:
        history dict with lists of per-epoch metrics.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Loss with positive class weighting ────────────────────────────────
    if pos_weight_val is None:
        # Compute from training data ratio
        pos_weight_val = 9.3   # approximation based on EDA (1193/119)
    pos_w     = torch.tensor([pos_weight_val], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=3)

    history       = {"train_loss": [], "val_loss": [],
                     "train_f1": [],   "val_f1": []}
    best_val_loss = float("inf")
    patience_cnt  = 0
    patience      = 7

    save_path = os.path.join(CHECKPOINT_DIR, save_name)
    print(f"\n  Starting training for {n_epochs} epochs ...")
    print(f"  Checkpoint will be saved to: {save_path}\n")
    print(f"  {'Epoch':>5}  {'Tr Loss':>8}  {'Tr F1':>7}  "
          f"{'Va Loss':>8}  {'Va F1':>7}  {'LR':>10}")
    print("  " + "-" * 58)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        tr  = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        val = evaluate(model, val_loader, criterion)

        # ── Scheduler step ─────────────────────────────────────────────────
        scheduler.step(val["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        # ── History ────────────────────────────────────────────────────────
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(val["loss"])
        history["train_f1"].append(tr["f1"])
        history["val_f1"].append(val["f1"])

        dt = time.time() - t0
        print(f"  {epoch:>5}  {tr['loss']:>8.4f}  {tr['f1']:>7.4f}  "
              f"{val['loss']:>8.4f}  {val['f1']:>7.4f}  "
              f"{current_lr:>10.1e}  ({dt:.0f}s)")

        # ── Save best model ────────────────────────────────────────────────
        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            patience_cnt  = 0
            torch.save(model.state_dict(), save_path)
            print(f"          ✓ Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs).")
                break

    # ── Save training history as CSV ───────────────────────────────────────
    csv_path = os.path.join(CHECKPOINT_DIR, "clf_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss",
                                               "val_loss", "train_f1", "val_f1"])
        writer.writeheader()
        for i in range(len(history["train_loss"])):
            writer.writerow({
                "epoch":      i + 1,
                "train_loss": history["train_loss"][i],
                "val_loss":   history["val_loss"][i],
                "train_f1":   history["train_f1"][i],
                "val_f1":     history["val_f1"][i],
            })
    print(f"\n  History saved → {csv_path}")
    print(f"  Best Val Loss : {best_val_loss:.4f}")
    return history
