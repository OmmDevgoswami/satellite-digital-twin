import os
import time
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.data.transforms import get_val_transforms
from src.models.classifier import get_classifier
from src.models.segmentation_model import get_segmentation_model
from src.utils.config import CHECKPOINT_DIR, DEVICE, IMAGE_SIZE

try:
    import pathway as pw  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Pathway is not available in this environment. "
        "Install the official Pathway package from pathway.com "
        "in a supported Linux/Python environment to run this pipeline."
    ) from e

if not hasattr(pw, "Schema"):  # pragma: no cover
    raise RuntimeError(
        "The installed 'pathway' package is a placeholder stub, not the real Pathway.\n"
        "For local Windows testing, use the plain Python folder_stream script instead.\n"
        "For hackathon submission, run this pipeline in a Linux environment with the "
        "official Pathway package installed."
    )


# Incoming images dropped here will be treated as a live stream.
STREAM_INPUT_DIR = os.path.join("data", "stream_incoming")

# Pathway will maintain a continuously updated CSV of detections here.
STREAM_OUTPUT_CSV = os.path.join("outputs", "pathway", "live_events.csv")


class ImageEventSchema(pw.Schema):
    id: str = pw.column_definition(primary_key=True)
    filename: str
    timestamp: float


class DetectionSchema(pw.Schema):
    id: str = pw.column_definition(primary_key=True)
    filename: str
    timestamp: float
    dump_probability: float
    dump_decision: bool
    dump_coverage_pct: float


_CLF_MODEL: Optional[torch.nn.Module] = None
_SEG_MODEL: Optional[torch.nn.Module] = None
_INFER_CACHE: dict[str, Tuple[float, bool, float]] = {}


def _ensure_models() -> Tuple[torch.nn.Module, Optional[torch.nn.Module]]:
    """Lazily load classifier and segmentation models once per process."""
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


def _run_models_on_image(
    clf: torch.nn.Module,
    seg: Optional[torch.nn.Module],
    path: str,
    threshold: float = 0.5,
) -> Tuple[float, bool, float]:
    """Run classifier and segmentation on a single image path."""
    pil = Image.open(path).convert("RGB")
    img_np = np.array(pil)

    # Use the same validation transforms as the web app.
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


def _get_inference_for_filename(filename: str) -> Tuple[float, bool, float]:
    """Cached wrapper so we don't recompute inference multiple times per row."""
    if filename in _INFER_CACHE:
        return _INFER_CACHE[filename]

    clf, seg = _ensure_models()
    img_path = os.path.join(STREAM_INPUT_DIR, filename)
    prob, is_dump, coverage = _run_models_on_image(clf, seg, img_path)
    _INFER_CACHE[filename] = (prob, is_dump, coverage)
    return prob, is_dump, coverage


class FolderStreamSubject(pw.io.python.ConnectorSubject):
    """
    Simple custom connector that turns a folder of images into a stream.

    New files dropped into STREAM_INPUT_DIR are emitted as events with a timestamp.
    """

    def run(self) -> None:
        os.makedirs(STREAM_INPUT_DIR, exist_ok=True)
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
                ts = time.time()
                self.next(id=fname, filename=fname, timestamp=ts)
            time.sleep(2.0)


@pw.udf
def dump_probability(filename: str) -> float:
    prob, _, _ = _get_inference_for_filename(filename)
    return prob


@pw.udf
def dump_decision(filename: str) -> bool:
    _, is_dump, _ = _get_inference_for_filename(filename)
    return is_dump


@pw.udf
def dump_coverage_pct(filename: str) -> float:
    _, _, coverage = _get_inference_for_filename(filename)
    return coverage


def main() -> None:
    """
    Entry point to run the Pathway pipeline.

    Usage (from project root):
        conda activate dump_detect
        python -m src.streaming.pathway_pipeline

    Then drop images into data/stream_incoming/.
    Pathway will continuously update outputs/pathway/live_events.csv.
    """
    os.makedirs(os.path.dirname(STREAM_OUTPUT_CSV), exist_ok=True)

    src = pw.io.python.read(FolderStreamSubject(), schema=ImageEventSchema)

    detections = src.select(
        id=src.id,
        filename=src.filename,
        timestamp=src.timestamp,
        dump_probability=dump_probability(src.filename),
        dump_decision=dump_decision(src.filename),
        dump_coverage_pct=dump_coverage_pct(src.filename),
    )

    pw.io.csv.write(detections, STREAM_OUTPUT_CSV, schema=DetectionSchema)
    pw.run()


if __name__ == "__main__":
    main()

