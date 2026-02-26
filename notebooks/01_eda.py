"""
01_eda.py  (v3 — fixed paths + metadata-based labelling)
---------------------------------------------------------
Phase 2: Exploratory Data Analysis for AerialWaste dataset.

Key findings about this dataset:
  - Images live in: data/raw/AerialWaste/images/images0/
  - training.json has NO polygon masks — only image-level metadata
  - ALL images are from dump sites collected in the field
  - Binary classification label strategy:
      categories != [] → CONFIRMED DUMP   (label = 1)
      categories == [] → CANDIDATE / NONE (label = 0)
  - testing.json has 166 polygon-annotated images (for segmentation eval)

Run from project root:
    conda activate dump_detect
    python notebooks/01_eda.py
"""

import os
import sys
import json
import random
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import IMAGES_DIR, TRAIN_JSON, TEST_JSON, IMAGE_SIZE

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eda_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════
def load_metadata(json_path: str) -> list:
    """
    Load training/testing JSON and return a list of image dicts.
    Each dict has at minimum: id, file_name, severity, categories, polygons.
    severity/evidence are cast to int (JSON sometimes stores them as strings).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Build annotation lookup (image_id → polygons) if annotations exist
    ann_lookup = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        ann_lookup.setdefault(img_id, [])
        for seg in ann.get("segmentation", []):
            pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
            ann_lookup[img_id].append(pts)

    records = []
    for img in data["images"]:
        cat_ids  = img.get("categories", [])
        label    = 1 if len(cat_ids) > 0 else 0   # our binary label
        # Cast to int safely — JSON contains 'n/a', None, or int values
        def safe_int(val, default=0):
            try:
                return int(val)
            except (TypeError, ValueError):
                return default
        severity = safe_int(img.get("severity", 0))
        evidence = safe_int(img.get("evidence", 0))
        fname    = os.path.basename(img["file_name"])
        records.append({
            "id":         img["id"],
            "file_name":  fname,
            "label":      label,
            "severity":   severity,
            "evidence":   evidence,
            "categories": cat_ids,
            "polygons":   ann_lookup.get(img["id"], []),
        })
    return records


def get_ondisk_records(records: list) -> list:
    """Return only records whose image file exists in IMAGES_DIR.
    WHY: images0.zip is only 1 of 7 zips. Most JSON entries point to
         images in images1-6 which aren't downloaded. We filter those out
         so sampling functions always find a real file.
    """
    return [r for r in records if find_image(r["file_name"]) is not None]


def find_image(fname: str) -> str | None:
    """Return full path if image file exists in IMAGES_DIR, else None."""
    path = os.path.join(IMAGES_DIR, fname)
    return path if os.path.exists(path) else None


def polygon_to_mask(polygons, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    for pts in polygons:
        cv2.fillPoly(mask, [pts.astype(np.int32).reshape(-1, 1, 2)], color=1)
    return mask


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Scan the dataset
# ══════════════════════════════════════════════════════════════════════════
def step1_scan(train_records, test_records):
    print("\n" + "=" * 55)
    print("  STEP 1: Dataset Scan")
    print("=" * 55)

    disk = [f for f in os.listdir(IMAGES_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    n_dump   = sum(1 for r in train_records if r["label"] == 1)
    n_nodump = sum(1 for r in train_records if r["label"] == 0)

    print(f"\n  Images on disk (images0/)  : {len(disk)}")
    print(f"\n  TRAINING JSON  ({os.path.basename(TRAIN_JSON)})")
    print(f"    Total entries            : {len(train_records)}")
    print(f"    Confirmed dump  (label=1): {n_dump}  (categories not empty)")
    print(f"    Candidate/None  (label=0): {n_nodump}  (categories empty)")
    print(f"\n  TESTING JSON   ({os.path.basename(TEST_JSON)})")
    print(f"    Total entries            : {len(test_records)}")
    print(f"    With polygon masks       : {sum(1 for r in test_records if r['polygons'])}")

    print("\n  [WHY] We use 'categories' field as the binary label.")
    print("         Images with identified waste types → confirmed dump.")
    print("         This gives us a real research-grade labeling strategy.")


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Inspect image shapes
# ══════════════════════════════════════════════════════════════════════════
def step2_shapes(all_records, n_sample=20):
    print("\n" + "=" * 55)
    print("  STEP 2: Image Shape & Pixel Statistics")
    print("=" * 55)

    # Only sample from records that exist on disk (images0 subset)
    ondisk  = get_ondisk_records(all_records)
    sample  = random.sample(ondisk, min(n_sample, len(ondisk)))
    shapes, mins, maxs = [], [], []

    for rec in sample:
        img = cv2.imread(find_image(rec["file_name"]))
        if img is None:
            continue
        shapes.append(img.shape)
        mins.append(img.min())
        maxs.append(img.max())

    if not shapes:
        print("  [WARNING] No images loaded — check IMAGES_DIR path in config.py")
        return

    print(f"\n  On-disk images (images0)  : {len(ondisk)} / {len(all_records)} total")
    print(f"  Loaded sample size        : {len(shapes)}")
    print(f"  Unique image shapes       : {set(shapes)}")
    print(f"  Pixel value range         : min={min(mins)}, max={max(maxs)}")
    print(f"  Target training size      : {IMAGE_SIZE}")
    print("\n  [WHY] Uniform size lets PyTorch batch images efficiently.")


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Visualise sample images with labels
# ══════════════════════════════════════════════════════════════════════════
def step3_visualize(all_records, n=6):
    print("\n" + "=" * 55)
    print("  STEP 3: Visualising Sample Images")
    print("=" * 55)

    # Filter to on-disk images first
    ondisk      = get_ondisk_records(all_records)
    dump_recs   = [r for r in ondisk if r["label"] == 1]
    nodump_recs = [r for r in ondisk if r["label"] == 0]

    chosen = (random.sample(dump_recs,   min(n // 2, len(dump_recs)))
            + random.sample(nodump_recs, min(n // 2, len(nodump_recs))))
    random.shuffle(chosen)

    loaded = []
    for rec in chosen:
        path = find_image(rec["file_name"])
        if path:
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                loaded.append((img, rec))
        if len(loaded) >= n:
            break

    if not loaded:
        print("  [WARNING] No images found on disk — check path.")
        return

    cols = 3
    rows = len(loaded)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1:
        axes = [axes]

    for idx, (img, rec) in enumerate(loaded):
        resized = cv2.resize(img, IMAGE_SIZE)

        # Col 0: original
        axes[idx][0].imshow(img)
        axes[idx][0].set_title(f"Original  [{rec['file_name']}]", fontsize=8)
        axes[idx][0].axis("off")

        # Col 1: resized (what the model sees)
        axes[idx][1].imshow(resized)
        label_str = "[DUMP]" if rec["label"] == 1 else "[No Dump]"
        axes[idx][1].set_title(
            f"Resized {IMAGE_SIZE}\nLabel: {label_str} | Sev: {rec['severity']} | Ev: {rec['evidence']}",
            fontsize=8)
        axes[idx][1].axis("off")

        # Col 2: severity bar
        axes[idx][2].barh(
            ["Severity", "Evidence"],
            [rec["severity"], rec["evidence"]],
            color=["#e74c3c" if rec["label"] else "#2ecc71", "#3498db"]
        )
        axes[idx][2].set_xlim(0, 5)
        axes[idx][2].set_title("Metadata scores", fontsize=8)

    plt.suptitle("AerialWaste: Images + Labels + Metadata", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "sample_images.png")
    plt.savefig(out, dpi=90, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] → {out}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Class balance
# ══════════════════════════════════════════════════════════════════════════
def step4_balance(train_records, test_records):
    print("\n" + "=" * 55)
    print("  STEP 4: Class Balance Analysis")
    print("=" * 55)

    tr_dump   = sum(1 for r in train_records if r["label"] == 1)
    tr_nodump = len(train_records) - tr_dump
    ts_seg    = sum(1 for r in test_records  if r["polygons"])
    ts_noseg  = len(test_records) - ts_seg

    ratio = max(tr_dump, tr_nodump) / max(min(tr_dump, tr_nodump), 1)
    print(f"\n  Training  — Dump: {tr_dump} | No-Dump: {tr_nodump} | Ratio: {ratio:.1f}x")
    print(f"  Testing   — With mask: {ts_seg} | No mask: {ts_noseg}")
    print("\n  [WHY] Imbalanced data can bias the model.")
    print("         We'll use weighted loss in Phase 3 to compensate.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].pie([tr_dump, tr_nodump],
                labels=[f"Confirmed Dump\n({tr_dump})", f"Candidate/None\n({tr_nodump})"],
                colors=["#e74c3c", "#2ecc71"], autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 10})
    axes[0].set_title("Training: Class Balance", fontsize=12, fontweight="bold")

    axes[1].pie([ts_seg, ts_noseg],
                labels=[f"Has Polygon Mask\n({ts_seg})", f"No Mask\n({ts_noseg})"],
                colors=["#9b59b6", "#95a5a6"], autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 10})
    axes[1].set_title("Testing: Segmentation Coverage", fontsize=12, fontweight="bold")

    plt.suptitle("Dataset Balance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "class_balance.png")
    plt.savefig(out, dpi=90, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] → {out}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — Severity distribution
# ══════════════════════════════════════════════════════════════════════════
def step5_severity(train_records):
    print("\n" + "=" * 55)
    print("  STEP 5: Severity & Evidence Distribution")
    print("=" * 55)

    severities = [r["severity"] for r in train_records]
    evidences  = [r["evidence"]  for r in train_records]

    from collections import Counter
    sev_counts = Counter(severities)
    ev_counts  = Counter(evidences)
    print(f"\n  Severity levels  : {dict(sorted(sev_counts.items()))}")
    print(f"  Evidence levels  : {dict(sorted(ev_counts.items()))}")
    print("\n  [WHY] These metadata fields could be used as multi-class")
    print("         labels in future experiments beyond binary detection.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sev_keys = sorted(sev_counts.keys())
    axes[0].bar([str(k) for k in sev_keys], [sev_counts[k] for k in sev_keys],
                color="#e67e22", edgecolor="black")
    axes[0].set_title("Severity Distribution", fontsize=12)
    axes[0].set_xlabel("Severity Level")
    axes[0].set_ylabel("Count")

    ev_keys = sorted(ev_counts.keys())
    axes[1].bar([str(k) for k in ev_keys], [ev_counts[k] for k in ev_keys],
                color="#3498db", edgecolor="black")
    axes[1].set_title("Evidence Distribution", fontsize=12)
    axes[1].set_xlabel("Evidence Level")

    plt.suptitle("Dataset Metadata Statistics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "severity_distribution.png")
    plt.savefig(out, dpi=90, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] → {out}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "█" * 55)
    print("  Phase 2: EDA — AerialWaste Dataset")
    print("█" * 55)

    # Check paths
    missing = []
    if not os.path.isdir(IMAGES_DIR):
        missing.append(f"  Images folder: {IMAGES_DIR}")
    if not os.path.isfile(TRAIN_JSON):
        missing.append(f"  training.json: {TRAIN_JSON}")
    if not os.path.isfile(TEST_JSON):
        missing.append(f"  testing.json:  {TEST_JSON}")
    if missing:
        print("\n[ERROR] Missing paths:\n" + "\n".join(missing))
        sys.exit(1)

    print("\n  Loading training.json ...")
    train_records = load_metadata(TRAIN_JSON)
    print("  Loading testing.json  ...")
    test_records  = load_metadata(TEST_JSON)

    step1_scan(train_records, test_records)
    step2_shapes(train_records)
    step3_visualize(train_records, n=4)
    step4_balance(train_records, test_records)
    step5_severity(train_records)

    print("\n" + "█" * 55)
    print("  ✅ EDA Complete!")
    print("  Charts saved to: notebooks/eda_output/")
    print("    - sample_images.png")
    print("    - class_balance.png")
    print("    - severity_distribution.png")
    print("█" * 55)
