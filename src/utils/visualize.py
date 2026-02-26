"""
visualize.py
------------
Utility functions for visualizing satellite images and their segmentation
masks throughout the project.

WHY: Visualization is critical in remote sensing projects. We always
     need to visually confirm that data is loading correctly, augmentations
     look right, and model predictions make sense.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2


def show_image(image: np.ndarray, title: str = "Image", ax=None):
    """Display a single image (RGB or grayscale)."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    if image.ndim == 2:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(image)
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def show_image_mask_pair(image: np.ndarray, mask: np.ndarray, title: str = ""):
    """
    Display a satellite image alongside its binary segmentation mask.

    Args:
        image : H x W x 3  RGB array (uint8)
        mask  : H x W      binary array (0 = no dump, 1 = dump)
        title : optional suptitle
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Satellite Image", fontsize=12)
    axes[0].axis("off")

    # Binary mask
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask\n(White = Dump)", fontsize=12)
    axes[1].axis("off")

    # Overlay: image with coloured mask on top
    overlay = image.copy()
    dump_pixels = mask > 0
    overlay[dump_pixels] = [255, 50, 50]   # Red highlight on dump areas
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay\n(Red = Dump zone)", fontsize=12)
    axes[2].axis("off")

    red_patch = mpatches.Patch(color="red", label="Illegal Dump")
    axes[2].legend(handles=[red_patch], loc="lower right", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()


def show_batch(images, masks, n: int = 4):
    """
    Display n image-mask pairs from a batch for quick sanity check.

    Args:
        images : list or batch tensor of RGB images
        masks  : list or batch tensor of binary masks
        n      : how many pairs to display (max)
    """
    n = min(n, len(images))
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    for i in range(n):
        img = images[i]
        msk = masks[i]

        # Handle torch tensors → numpy
        if hasattr(img, "numpy"):
            img = img.permute(1, 2, 0).numpy()
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img = np.clip(img, 0, 1)
        if hasattr(msk, "numpy"):
            msk = msk.squeeze().numpy()

        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Image {i+1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(msk, cmap="gray")
        axes[1, i].set_title(f"Mask {i+1}")
        axes[1, i].axis("off")

    plt.suptitle("Batch Preview", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_class_distribution(labels: list, class_names: list = None):
    """
    Bar chart of class distribution. Helps detect class imbalance early.

    Args:
        labels      : list of integer class labels
        class_names : list of class name strings
    """
    import collections
    counter = collections.Counter(labels)
    if class_names is None:
        class_names = [str(k) for k in sorted(counter.keys())]

    counts = [counter.get(i, 0) for i in range(len(class_names))]
    colors = ["#2ecc71", "#e74c3c"]  # green, red

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(class_names, counts, color=colors[:len(class_names)], edgecolor="black")
    ax.set_title("Class Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Samples")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(count), ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    plt.show()


def plot_mask_coverage(masks: list, threshold: float = 0.01):
    """
    Histogram of dump coverage percentage across all masks.
    Helps understand how small/large the dump regions are.

    Args:
        masks     : list of binary numpy masks
        threshold : minimum coverage to count as 'has dump'
    """
    coverages = []
    for m in masks:
        coverage = m.sum() / m.size
        coverages.append(coverage * 100)   # as percentage

    has_dump = sum(1 for c in coverages if c > threshold * 100)
    total    = len(coverages)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of coverage
    axes[0].hist(coverages, bins=30, color="#3498db", edgecolor="black")
    axes[0].set_title("Dump Coverage Distribution (%)")
    axes[0].set_xlabel("% of image covered by dump")
    axes[0].set_ylabel("Frequency")

    # Pie chart: has dump vs no dump
    axes[1].pie(
        [has_dump, total - has_dump],
        labels=["Has Dump", "No Dump"],
        colors=["#e74c3c", "#2ecc71"],
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[1].set_title("Class Balance")

    plt.suptitle("Dataset Statistics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
