"""
folder_stream.py
----------------

Plain-Python fallback for local testing on Windows where the real Pathway
package is not available.

This script watches `data/stream_incoming/` for new images, runs the existing
classifier + segmentation models, and continuously appends results to
`outputs/pathway/live_events.csv`.

The Streamlit app's "Live Stream (Pathway)" tab only needs this CSV, so you
can demonstrate end-to-end behaviour locally even without a full Pathway
installation. For hackathon submission, judges can use
`src/streaming/pathway_pipeline.py` in a supported Linux environment with
Pathway installed.
"""

import csv
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.data.transforms import get_val_transforms
from src.models.classifier import get_classifier
from src.models.segmentation_model import get_segmentation_model
from src.utils.config import CHECKPOINT_DIR, DEVICE


STREAM_INPUT_DIR = os.path.join("data", "stream_incoming")
STREAM_OUTPUT_CSV = os.path.join("outputs", "pathway", "live_events.csv")


_CLF_MODEL: Optional[torch.nn.Module] = None
_SEG_MODEL: Optional[torch.nn.Module] = None


def _ensure_models() -> Tuple[torch.nn.Module, Optional[torch.nn.Module]]:
    global _CLF_MODEL, _SEG_MODEL

    if _CLF_MODEL is None:
        clf_path = os.path.join(CHECKPOINT_DIR, "best_classifier.pt")
        clf = get_classifier(pretrained=False, freeze_layers=0)
        clf.load_state_dict(
            torch.load(clf_path, map_location=DEVICE, weights_only=True)
        )
        clf.to(DEVICE).eval()
        _CLF_MODEL = clf

    if _SEG_MODEL is None:
        seg_path = os.path.join(CHECKPOINT_DIR, "best_segmentation.pt")
        if os.path.exists(seg_path):
            seg = get_segmentation_model()
            seg.load_state_dict(
                torch.load(seg_path, map_location=DEVICE, weights_only=True)
            )
            seg.to(DEVICE).eval()
            _SEG_MODEL = seg
        else:
            _SEG_MODEL = None

    return _CLF_MODEL, _SEG_MODEL


def _infer_on_image(path: str, threshold: float = 0.5) -> Tuple[float, bool, float]:
    clf, seg = _ensure_models()

    pil = Image.open(path).convert("RGB")
    img_np = np.array(pil)
    tensor = get_val_transforms()(image=img_np)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(clf(tensor)).item()
    is_dump = prob >= threshold

    coverage = 0.0
    if seg is not None:
        with torch.no_grad():
            mask = torch.sigmoid(seg(tensor)).squeeze().cpu().numpy()
        coverage = float((mask >= 0.3).mean() * 100.0)

    return float(prob), bool(is_dump), float(coverage)


def main() -> None:
    os.makedirs(STREAM_INPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(STREAM_OUTPUT_CSV), exist_ok=True)

    # Create CSV with header if it doesn't exist.
    if not os.path.exists(STREAM_OUTPUT_CSV):
        with open(STREAM_OUTPUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "id",
                    "filename",
                    "timestamp",
                    "dump_probability",
                    "dump_decision",
                    "dump_coverage_pct",
                ]
            )

    seen: set[str] = set()
    while True:
        files = [
            f
            for f in os.listdir(STREAM_INPUT_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]
        for fname in sorted(files):
            if fname in seen:
                continue
            seen.add(fname)
            fpath = os.path.join(STREAM_INPUT_DIR, fname)
            ts = time.time()
            prob, is_dump, coverage = _infer_on_image(fpath)

            with open(STREAM_OUTPUT_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        fname,
                        fname,
                        ts,
                        prob,
                        int(is_dump),
                        coverage,
                    ]
                )
        time.sleep(2.0)


if __name__ == "__main__":
    main()

