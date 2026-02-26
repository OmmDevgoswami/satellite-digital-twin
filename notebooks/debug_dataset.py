"""
debug_dataset.py
----------------
Run this to diagnose:
  1. Where your images actually are on disk
  2. What the training.json structure looks like

Run:
    conda activate dump_detect
    python notebooks/debug_dataset.py
"""

import os, sys, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import DATA_RAW_DIR

AERIAL_ROOT = os.path.join(DATA_RAW_DIR, "AerialWaste")
TRAIN_JSON  = os.path.join(AERIAL_ROOT, "training.json")

print("\n" + "="*60)
print("  DIAGNOSTIC 1: Finding Image Files")
print("="*60)

# Walk the entire AerialWaste folder and show everything
if not os.path.isdir(AERIAL_ROOT):
    print(f"[ERROR] Folder not found: {AERIAL_ROOT}")
    print("Please check the directory name matches exactly.")
    sys.exit(1)

print(f"\nScanning: {AERIAL_ROOT}\n")
image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
for root, dirs, files in os.walk(AERIAL_ROOT):
    # Summarise each subfolder
    imgs = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
    jsns = [f for f in files if f.endswith(".json")]
    rel  = os.path.relpath(root, AERIAL_ROOT)
    print(f"  [{rel}]")
    print(f"    Images : {len(imgs)}")
    print(f"    JSON   : {jsns}")
    if imgs:
        print(f"    Sample filenames: {imgs[:3]}")
    print()

print("\n" + "="*60)
print("  DIAGNOSTIC 2: training.json Structure")
print("="*60)

if not os.path.isfile(TRAIN_JSON):
    print(f"[ERROR] training.json not found at {TRAIN_JSON}")
    sys.exit(1)

with open(TRAIN_JSON, "r") as f:
    coco = json.load(f)

print(f"\nTop-level keys in training.json: {list(coco.keys())}")

# Images
imgs = coco.get("images", [])
print(f"\nTotal images listed : {len(imgs)}")
print("First image entry   :")
print(json.dumps(imgs[0], indent=4) if imgs else "  (none)")

# Annotations
anns = coco.get("annotations", [])
print(f"\nTotal annotations   : {len(anns)}")
if anns:
    print("First annotation entry:")
    print(json.dumps(anns[0], indent=4))
    print("\nSecond annotation entry:")
    print(json.dumps(anns[1], indent=4) if len(anns) > 1 else "  (only one)")
else:
    print("  [WARNING] annotations list is EMPTY in training.json")

# Categories
cats = coco.get("categories", [])
print(f"\nCategories: {cats}")

print("\n" + "="*60)
print("  Paste ALL of this output back to the assistant.")
print("="*60)
