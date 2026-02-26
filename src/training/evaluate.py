"""
evaluate.py
-----------
Phase 6: Comprehensive Evaluation Report — Advanced Version

Generates:
  1. Classification metrics + comparison for ResNet and EfficientNet
  2. ROC curves for both
  3. Segmentation metrics comparison (UNet, UNet++, FPN, DeepLabV3+)
  4. Global results_summary.json for the dashboard

Run from project root:
    conda activate dump_detect
    python src/training/evaluate.py
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             classification_report)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.dataset          import get_clf_dataloaders, load_records, AerialWasteDataset
from src.data.transforms       import get_val_transforms
from src.models.classifier     import get_classifier
from src.models.segmentation_model import get_segmentation_model
from src.training.train_segmentation import get_seg_splits
from src.training.metrics      import batch_iou, batch_dice
from src.utils.config          import DEVICE, CHECKPOINT_DIR

EVAL_DIR = os.path.join(CHECKPOINT_DIR, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)


def evaluate_clf_model(name: str, backbone: str, ckpt_name: str, loader):
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] {name} not found at {ckpt_path}")
        return None

    print(f"  [CLF] Evaluating {name} ({backbone}) ...")
    try:
        model = get_classifier(backbone=backbone, pretrained=False, freeze_layers=0)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        model.eval()
    except Exception as e:
        print(f"  [ERROR] {name} load fail: {e}")
        return None

    all_labels, all_probs, all_preds = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(DEVICE))
            probs  = torch.sigmoid(logits).cpu().squeeze(1)
            preds  = (probs >= 0.5).int().tolist()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    labels_arr = np.array(all_labels)
    probs_arr  = np.array(all_probs)
    preds_arr  = np.array(all_preds)

    cm = confusion_matrix(labels_arr, preds_arr)
    fpr, tpr, _ = roc_curve(labels_arr, probs_arr)
    roc_auc     = auc(fpr, tpr)
    acc         = (preds_arr == labels_arr).mean()
    
    tp = int(cm[1,1]); fp = int(cm[0,1])
    tn = int(cm[0,0]); fn = int(cm[1,0])
    recall = tp / max(tp + fn, 1)
    prec   = tp / max(tp + fp, 1)
    f1     = 2 * prec * recall / max(prec + recall, 1e-8)

    return {
        "name":      name,
        "accuracy":  float(acc),
        "precision": float(prec),
        "recall":    float(recall),
        "f1":        float(f1),
        "roc_auc":   float(roc_auc),
        "fpr":       fpr.tolist(),
        "tpr":       tpr.tolist(),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }



def evaluate_seg_model(name: str, arch: str, ckpt_name: str, dataset):
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
    if not os.path.exists(ckpt_path):
        return None

    try:
        model = get_segmentation_model(arch=arch)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        model.eval()
    except Exception as e:
        print(f"  [ERROR] {name} load fail: {e}")
        return None

    ious, dices = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            img_t, mask_t = dataset[i]
            logit = model(img_t.unsqueeze(0).to(DEVICE))
            ious.append(batch_iou(logit,  mask_t.unsqueeze(0).to(DEVICE)))
            dices.append(batch_dice(logit, mask_t.unsqueeze(0).to(DEVICE)))

    return {
        "name":      name,
        "mean_iou":  float(np.mean(ious)),
        "mean_dice": float(np.mean(dices))
    }



def main():
    print("\n" + "█" * 60)
    print("  Comprehensive Evaluation — Advanced Models")
    print("█" * 60)

    # 1. Classification
    _, _, test_loader = get_clf_dataloaders()
    clf_jobs = [
        ("ResNet34 Baseline", "resnet34", "best_classifier.pt"),
        ("EfficientNet-B4",   "efficientnet_b4", "best_efficientnet.pt")
    ]
    clf_results = []
    for name, backbone, ckpt in clf_jobs:
        res = evaluate_clf_model(name, backbone, ckpt, test_loader)
        if res: clf_results.append(res)

    # 2. Segmentation
    _, _, test_recs = get_seg_splits()
    test_ds = AerialWasteDataset(test_recs, mode="seg", transform=get_val_transforms())
    seg_jobs = [
        ("U-Net Baseline", "Unet", "best_segmentation.pt"),
        ("UNet++",         "UnetPlusPlus", "best_unetplusplus.pt"),
        ("FPN",            "FPN", "best_fpn.pt"),
        ("DeepLabV3+",     "DeepLabV3Plus", "best_deeplabv3plus.pt"),
        ("Current Best",   "UnetPlusPlus", "best_advanced_seg.pt")
    ]
    seg_results = []
    for name, arch, ckpt in seg_jobs:
        res = evaluate_seg_model(name, arch, ckpt, test_ds)
        if res: seg_results.append(res)

    # 3. Save Summary
    summary = {
        "classification": {r["name"]: r for r in clf_results},
        "segmentation":   {r["name"]: r for r in seg_results}
    }
    # For back-compatibility with app.py's main display
    if clf_results:
        summary["classification_best"] = clf_results[0] if len(clf_results)==1 else (
            clf_results[1] if clf_results[1]["f1"] > clf_results[0]["f1"] else clf_results[0]
        )
    if seg_results:
        summary["segmentation_best"] = max(seg_results, key=lambda x: x["mean_iou"])

    json_path = os.path.join(EVAL_DIR, "results_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  [SAVED] Comparison summary -> {json_path}")

    # 4. Plots
    if clf_results:
        fig, ax = plt.subplots(figsize=(6, 5))
        for r in clf_results:
            ax.plot(r["fpr"], r["tpr"], label=f"{r['name']} (AUC={r['roc_auc']:.3f})")
        ax.plot([0,1],[0,1], "k--")
        ax.set_title("ROC Curve Comparison")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(EVAL_DIR, "roc_comparison.png"), dpi=90)
        plt.close()

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)
    if clf_results:
        print("\n  Classification Comparison:")
        for r in clf_results:
            print(f"    {r['name']:<20} | F1: {r['f1']:.4f} | AUC: {r['roc_auc']:.4f}")
    if seg_results:
        print("\n  Segmentation Comparison:")
        for r in seg_results:
            print(f"    {r['name']:<20} | IoU: {r['mean_iou']:.4f} | Dice: {r['mean_dice']:.4f}")
    print("\n" + "█" * 60)


if __name__ == "__main__":
    main()
