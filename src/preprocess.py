"""Preprocess the CHROMIFY portrait dataset.

This script reads the raw images from `data/raw/`, crops them to a portrait ratio,
resizes them to 128 x 160, saves RGB copies, and derives the grayscale L channel
and LAB targets from the same CIELAB conversion.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Allow `python src/preprocess.py` when launched from the project root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    COLOR_DIR,
    GRAYSCALE_DIR,
    LAB_DIR,
    RAW_DATA_DIR,
    SUPPORTED_IMAGE_EXTENSIONS,
    TARGET_IMAGE_SIZE,
    ensure_project_directories,
)
from src.preprocessing.image_utils import (
    crop_to_4_5,
    load_rgb_image,
    resize_image,
    lab_l_channel,
    rgb_to_lab,
)


def find_image_files(raw_dir: Path) -> List[Path]:
    """Collect image files recursively from the raw dataset directory."""
    if not raw_dir.exists():
        return []

    files: List[Path] = []
    for file_path in sorted(raw_dir.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            files.append(file_path)
    return files


def save_rgb_image(image: np.ndarray, output_path: Path) -> None:
    """Save an RGB array losslessly while preserving the output filename."""
    image_pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    image_pil.save(output_path, format="PNG")


def save_grayscale_image(image: np.ndarray, output_path: Path) -> None:
    """Save a grayscale array losslessly while preserving the output filename."""
    image_pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    image_pil.save(output_path, format="PNG")


def process_single_image(
    image_path: Path,
    color_dir: Path,
    grayscale_dir: Path,
    lab_dir: Path,
) -> None:
    """Process one image and save RGB, grayscale, and LAB versions."""
    rgb_image = load_rgb_image(image_path)
    cropped = crop_to_4_5(rgb_image, target_width=None, target_height=None)
    resized_rgb = resize_image(cropped, TARGET_IMAGE_SIZE)

    lab_image = rgb_to_lab(resized_rgb)
    grayscale_image = lab_l_channel(lab_image)

    save_rgb_image(resized_rgb, color_dir / image_path.name)
    save_grayscale_image(grayscale_image, grayscale_dir / image_path.name)
    encoded_lab, encoded_bytes = cv2.imencode(".png", lab_image)
    if not encoded_lab:
        raise OSError(f"Could not save LAB image: {lab_dir / image_path.name}")
    encoded_bytes.tofile(str(lab_dir / image_path.name))


def preprocess_dataset(
    raw_dir: Path = RAW_DATA_DIR,
    color_dir: Path = COLOR_DIR,
    grayscale_dir: Path = GRAYSCALE_DIR,
    lab_dir: Path = LAB_DIR,
) -> Dict[str, int]:
    """Run preprocessing for all images in the raw dataset folder."""
    ensure_project_directories()
    color_dir.mkdir(parents=True, exist_ok=True)
    grayscale_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    image_files = find_image_files(raw_dir)
    if not image_files:
        print(f"No images were found under: {raw_dir}")
        return {"successful": 0, "failed": 0}

    successful = 0
    failed = 0

    for image_path in tqdm(image_files, desc="Preprocessing portrait images", unit="image"):
        try:
            process_single_image(image_path, color_dir, grayscale_dir, lab_dir)
            successful += 1
        except Exception as exc:
            print(f"Skipping unreadable or corrupted image: {image_path.name} ({exc})")
            failed += 1

    print(f"Preprocessing finished successfully: {successful} images processed.")
    print(f"Preprocessing finished with errors: {failed} images skipped.")

    return {"successful": successful, "failed": failed}


if __name__ == "__main__":
    preprocess_dataset()
