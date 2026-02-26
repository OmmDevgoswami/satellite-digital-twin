"""
train_classifier.py
-------------------
Phase 4: Train the binary dump classification model.

This is the main entry-point. Run it from the project root:

    conda activate dump_detect
    python src/training/train_classifier.py

What happens:
  1. Loads training/validation DataLoaders (with WeightedSampler)
  2. Builds ResNet34 with pretrained ImageNet weights
  3. Trains for up to 30 epochs with early stopping
  4. Saves the best model to outputs/checkpoints/best_classifier.pt
  5. Plots learning curves to outputs/checkpoints/learning_curves.png
  6. Runs inference on the test set and prints the full classification report

Expected training time on CPU:
  ~3-6 minutes per epoch × ~10-15 epochs ≈ 30-90 minutes total.
  TIP: Let it run overnight if needed, or reduce NUM_EPOCHS in config.py.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.dataset   import get_clf_dataloaders
from src.models.classifier import get_classifier
from src.training.trainer  import run_training, evaluate
from src.training.metrics  import print_clf_report
from src.utils.config     import DEVICE, CHECKPOINT_DIR


def plot_learning_curves(history: dict, save_path: str):
    """Save loss and F1 learning curves to a PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss",   markersize=4)
    axes[0].set_title("Loss Curve", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_f1"], "b-o", label="Train F1", markersize=4)
    axes[1].plot(epochs, history["val_f1"],   "r-o", label="Val F1",   markersize=4)
    axes[1].set_title("F1 Score Curve", fontsize=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Classification Training — Learning Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=90, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] Learning curves → {save_path}")


def main():
    print("\n" + "█" * 60)
    print("  Phase 4: Training Dump Classifier (ResNet34)")
    print("█" * 60)

    # ── Step 1: DataLoaders ────────────────────────────────────────────────
    print("\n[1] Building DataLoaders ...")
    train_loader, val_loader, test_loader = get_clf_dataloaders()
    print(f"    Train batches : {len(train_loader)}")
    print(f"    Val   batches : {len(val_loader)}")
    print(f"    Test  batches : {len(test_loader)}")

    # ── Step 2: Model ───────────────────────────────────────────────────────
    print("\n[2] Building model ...")
    model = get_classifier(backbone="resnet34", pretrained=True, freeze_layers=6)

    # ── Step 3: Train ───────────────────────────────────────────────────────
    print("\n[3] Training ...")
    # WHY lr=5e-5? With most layers frozen, we only update ~5M params.
    # A smaller LR avoids overshooting the narrow loss valley.
    history = run_training(model, train_loader, val_loader, lr=5e-5)

    # ── Step 4: Learning curves ─────────────────────────────────────────────
    print("\n[4] Plotting learning curves ...")
    curve_path = os.path.join(CHECKPOINT_DIR, "learning_curves.png")
    plot_learning_curves(history, curve_path)

    # ── Step 5: Test evaluation ─────────────────────────────────────────────
    print("\n[5] Evaluating on Test Set ...")
    import torch.nn as nn

    # Reload best model
    best_path = os.path.join(CHECKPOINT_DIR, "best_classifier.pt")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE,
                                     weights_only=True))
    model.eval()

    pos_w     = torch.tensor([9.3]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    test_metrics = evaluate(model, test_loader, criterion)

    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to(DEVICE))
            probs  = torch.sigmoid(logits).cpu()
            preds  = (probs >= 0.5).int().squeeze(1).tolist()
            lbls   = labels.int().tolist()
            all_preds.extend(preds)
            all_labels.extend(lbls)

    print("\n" + "=" * 55)
    print("  TEST SET RESULTS")
    print("=" * 55)
    print(f"  Loss      : {test_metrics['loss']:.4f}")
    print(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"  F1 Score  : {test_metrics['f1']:.4f}")
    print(f"  Precision : {test_metrics['precision']:.4f}")
    print(f"  Recall    : {test_metrics['recall']:.4f}")
    print("\n  Detailed Report:")
    print_clf_report(all_labels, all_preds)

    print("\n" + "█" * 60)
    print("  Phase 4 Complete!")
    print(f"  Best model  : {best_path}")
    print(f"  Curves      : {curve_path}")
    print("█" * 60)


if __name__ == "__main__":
    main()
