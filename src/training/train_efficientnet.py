"""
train_efficientnet.py
---------------------
Train EfficientNet-B4 as an advanced binary dump classifier.

WHY EfficientNet-B4?
  EfficientNet uses compound scaling (depth × width × resolution) to achieve
  better accuracy per parameter than ResNet. B4 has a 380-pixel receptive field
  and captures fine-grained dump textures that ResNet34 sometimes misses.

  Training strategy:
    - Stage 1 (warm-up): Only train the classifier head for 5 epochs
    - Stage 2 (fine-tune): Unfreeze all layers, train with cosine LR decay

Run from project root:
    conda activate dump_detect
    python src/training/train_efficientnet.py
"""

import os, sys, time, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.dataset        import get_clf_dataloaders
from src.models.classifier   import get_efficientnet_classifier
from src.training.metrics    import clf_metrics, print_clf_report
from src.utils.config        import DEVICE, CHECKPOINT_DIR


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced binary classification.
    Focuses learning on hard/misclassified examples by down-weighting
    easy negatives. Critical when dump:no-dump ratio is 1:9.3.

    FL(p) = -alpha * (1-p)^gamma * log(p)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce   = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs    = torch.sigmoid(logits)
        p_t      = probs * targets + (1 - probs) * (1 - targets)
        focal_w  = (1 - p_t) ** self.gamma
        alpha_t  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * focal_w * bce_loss).mean()


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    all_labels, all_preds = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).float().unsqueeze(1)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits).detach().cpu() >= 0.5).int().squeeze(1).tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.int().cpu().squeeze(1).tolist())
    metrics = clf_metrics(all_labels, all_preds)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)
            logits = model(images)
            running_loss += criterion(logits, labels).item() * images.size(0)
            preds = (torch.sigmoid(logits).cpu() >= 0.5).int().squeeze(1).tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.int().cpu().squeeze(1).tolist())
    metrics = clf_metrics(all_labels, all_preds)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics


def main():
    print("\n" + "█" * 60)
    print("  EfficientNet-B4 Dump Classifier Training")
    print("█" * 60)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n[1] Building DataLoaders ...")
    train_loader, val_loader, test_loader = get_clf_dataloaders()

    # ── Model ────────────────────────────────────────────────────────────────
    print("\n[2] Building EfficientNet-B4 model ...")
    model = get_efficientnet_classifier(pretrained=True, freeze_layers=4)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    save_path = os.path.join(CHECKPOINT_DIR, "best_efficientnet.pt")

    # ── Stage 1: Warm-up head only ───────────────────────────────────────────
    print("\n[3] Stage 1 — Warm-up (head only, 5 epochs) ...")
    for param in model.base.features.parameters():
        param.requires_grad = False
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=1e-3, weight_decay=1e-5)
    for epoch in range(1, 6):
        tr = train_one_epoch(model, train_loader, optimizer, criterion)
        vl = evaluate(model, val_loader, criterion)
        print(f"  Warm-up {epoch}/5 | Tr Loss:{tr['loss']:.4f} F1:{tr['f1']:.4f} "
              f"| Va Loss:{vl['loss']:.4f} F1:{vl['f1']:.4f}")

    # ── Stage 2: Fine-tune all layers ────────────────────────────────────────
    print("\n[4] Stage 2 — Fine-tune all layers ...")
    for param in model.parameters():
        param.requires_grad = True

    n_epochs    = 25
    optimizer   = Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler   = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-7)
    best_val_f1 = 0.0
    patience_cnt, patience = 0, 8
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    print(f"  {'Epoch':>5}  {'TrLoss':>8}  {'TrF1':>7}  {'VaLoss':>8}  {'VaF1':>7}")
    print("  " + "-" * 52)
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, criterion)
        vl = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(vl["loss"])
        history["train_f1"].append(tr["f1"])
        history["val_f1"].append(vl["f1"])

        dt = time.time() - t0
        print(f"  {epoch:>5}  {tr['loss']:>8.4f}  {tr['f1']:>7.4f}  "
              f"{vl['loss']:>8.4f}  {vl['f1']:>7.4f}  ({dt:.0f}s)")

        if vl["f1"] > best_val_f1:
            best_val_f1  = vl["f1"]
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
            print(f"          ✓ Saved best model (val_F1={best_val_f1:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    # ── Save history ─────────────────────────────────────────────────────────
    csv_path = os.path.join(CHECKPOINT_DIR, "efficientnet_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch"] + list(history.keys()))
        writer.writeheader()
        for i in range(len(history["train_loss"])):
            row = {"epoch": i + 1}
            row.update({k: history[k][i] for k in history})
            writer.writerow(row)

    # ── Plot curves ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(ep, history["train_loss"], "b-o", label="Train", markersize=4)
    axes[0].plot(ep, history["val_loss"],   "r-o", label="Val",   markersize=4)
    axes[0].set_title("Focal Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(ep, history["train_f1"],   "b-o", label="Train F1", markersize=4)
    axes[1].plot(ep, history["val_f1"],     "r-o", label="Val F1",   markersize=4)
    axes[1].set_title("F1 Score"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.suptitle("EfficientNet-B4 Training Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    curve_path = os.path.join(CHECKPOINT_DIR, "efficientnet_curves.png")
    plt.savefig(curve_path, dpi=90, bbox_inches="tight")
    plt.close()

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n[5] Test Set Evaluation ...")
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    test_m = evaluate(model, test_loader, criterion)
    all_labels, all_preds = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(DEVICE))
            preds  = (torch.sigmoid(logits).cpu() >= 0.5).int().squeeze(1).tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.int().tolist())

    print("\n" + "=" * 55)
    print("  TEST SET RESULTS — EfficientNet-B4")
    print("=" * 55)
    print(f"  Loss      : {test_m['loss']:.4f}")
    print(f"  Accuracy  : {test_m['accuracy']:.4f}")
    print(f"  F1 Score  : {test_m['f1']:.4f}")
    print(f"  Precision : {test_m['precision']:.4f}")
    print(f"  Recall    : {test_m['recall']:.4f}")
    print_clf_report(all_labels, all_preds)

    print("\n" + "█" * 60)
    print(f"  EfficientNet-B4 Training Complete!")
    print(f"  Best model : {save_path}")
    print(f"  Curves     : {curve_path}")
    print("█" * 60)


if __name__ == "__main__":
    main()
