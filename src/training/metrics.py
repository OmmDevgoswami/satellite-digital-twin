"""
metrics.py
----------
Evaluation metrics for both classification and segmentation tasks.

WHY these specific metrics?
  - Accuracy alone is misleading with imbalanced data (9.3x imbalance).
    A model that always predicts 'no-dump' gets 92% accuracy but is useless.
  - F1-Score balances Precision and Recall — it's the standard metric
    for imbalanced binary classification.
  - IoU (Intersection over Union) is the standard metric for segmentation.
    It measures the overlap between predicted and ground truth regions.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)


# ══════════════════════════════════════════════════════════════════════════
# Classification Metrics
# ══════════════════════════════════════════════════════════════════════════

def clf_metrics(all_labels: list, all_preds: list) -> dict:
    """
    Compute classification metrics from lists of labels and predictions.

    Args:
        all_labels : list of ground-truth int labels (0 or 1)
        all_preds  : list of predicted int labels (0 or 1)

    Returns:
        dict with accuracy, f1, precision, recall
    """
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}


def print_clf_report(all_labels: list, all_preds: list,
                     class_names: list = None):
    """Print a full sklearn classification report."""
    target_names = class_names or ["No Dump", "Dump"]
    print(classification_report(all_labels, all_preds,
                                 target_names=target_names,
                                 zero_division=0))
    cm = confusion_matrix(all_labels, all_preds)
    print("  Confusion Matrix:")
    print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")


# ══════════════════════════════════════════════════════════════════════════
# Segmentation Metrics
# ══════════════════════════════════════════════════════════════════════════

def iou_score(pred_mask: np.ndarray, true_mask: np.ndarray,
              threshold: float = 0.5) -> float:
    """
    Intersection over Union (IoU) for binary segmentation.

    IoU = |Intersection| / |Union|
        = TP / (TP + FP + FN)

    Args:
        pred_mask : H×W float array of predicted probabilities or logits
        true_mask : H×W binary int array (0 or 1)
        threshold : sigmoid threshold to convert probabilities to binary
    """
    pred_bin = (pred_mask >= threshold).astype(np.int32)
    true_bin = true_mask.astype(np.int32)
    intersection = (pred_bin & true_bin).sum()
    union        = (pred_bin | true_bin).sum()
    if union == 0:
        return 1.0   # both masks are empty — perfect match
    return intersection / union


def dice_score(pred_mask: np.ndarray, true_mask: np.ndarray,
               threshold: float = 0.5) -> float:
    """
    Dice coefficient (F1 for segmentation masks).

    Dice = 2 * |Intersection| / (|Pred| + |True|)
         = 2TP / (2TP + FP + FN)
    """
    pred_bin = (pred_mask >= threshold).astype(np.int32)
    true_bin = true_mask.astype(np.int32)
    intersection = (pred_bin * true_bin).sum()
    denom = pred_bin.sum() + true_bin.sum()
    if denom == 0:
        return 1.0
    return 2.0 * intersection / denom


def batch_iou(pred_batch: torch.Tensor,
              mask_batch: torch.Tensor,
              threshold: float = 0.5) -> float:
    """
    Average IoU over an entire batch.

    Args:
        pred_batch : [B, 1, H, W] raw logits from model
        mask_batch : [B, 1, H, W] ground truth masks (0/1 float)
    """
    probs   = torch.sigmoid(pred_batch).detach().cpu().numpy()
    masks   = mask_batch.detach().cpu().numpy()
    ious    = []
    for p, m in zip(probs, masks):
        ious.append(iou_score(p.squeeze(), m.squeeze(), threshold))
    return float(np.mean(ious))


def batch_dice(pred_batch: torch.Tensor,
               mask_batch: torch.Tensor,
               threshold: float = 0.5) -> float:
    """Average Dice over an entire batch."""
    probs = torch.sigmoid(pred_batch).detach().cpu().numpy()
    masks = mask_batch.detach().cpu().numpy()
    dices = []
    for p, m in zip(probs, masks):
        dices.append(dice_score(p.squeeze(), m.squeeze(), threshold))
    return float(np.mean(dices))
