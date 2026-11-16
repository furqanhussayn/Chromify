"""Validate the approved and preprocessed CHROMIFY 5,000-image dataset."""

from __future__ import annotations

import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import COLOR_DIR, GRAYSCALE_DIR, LAB_DIR, RAW_DATA_DIR, SPLITS_DIR
from src.dataset import ColorizationDataset


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "expansion_preview" / "selection_manifest.csv"
EXPECTED = 5_000
EXPECTED_SHAPES = {"train": 4_000, "val": 500, "test": 500}


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest() -> List[Dict[str, str]]:
    """Read the approved manifest."""
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def image_files(directory: Path) -> Dict[str, Path]:
    """Return supported image files keyed by filename."""
    return {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }


def validate_processed_images(names: set[str], directory: Path, label: str) -> None:
    """Check completeness, readability, and exact dimensions for one output set."""
    files = image_files(directory)
    missing = names - set(files)
    extra = set(files) - names
    if missing or extra:
        raise ValueError(f"{label}: missing={len(missing)}, extra={len(extra)}")
    for filename, path in files.items():
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"{label}: unreadable file {path}")
        if image.shape[1] != 128 or image.shape[0] != 160:
            raise ValueError(f"{label}: wrong dimensions for {filename}: {image.shape}")


def validate_splits(rows: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """Validate installed split files against the approved manifest."""
    manifest_by_name = {row["filename"]: row for row in rows}
    split_names: Dict[str, List[str]] = {}
    split_ids: Dict[str, set[str]] = {}
    for split_name, expected_count in EXPECTED_SHAPES.items():
        path = SPLITS_DIR / f"{split_name}.txt"
        names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(names) != expected_count or len(set(names)) != expected_count:
            raise ValueError(f"{split_name}: expected {expected_count} unique names, found {len(names)}")
        if any(name not in manifest_by_name for name in names):
            raise ValueError(f"{split_name}: contains a name absent from the approved manifest")
        split_names[split_name] = names
        split_ids[split_name] = {manifest_by_name[name]["identity_id"] for name in names}

    all_names = set().union(*(set(names) for names in split_names.values()))
    if len(all_names) != EXPECTED:
        raise ValueError(f"Splits contain {len(all_names)} unique names instead of {EXPECTED}")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_ids[left] & split_ids[right]
        if overlap:
            raise ValueError(f"Identity leakage between {left} and {right}: {len(overlap)} identities")
    return split_names


def validate_loader(split_names: Dict[str, List[str]]) -> Dict[str, tuple]:
    """Load each split and validate one real DataLoader batch."""
    batch_shapes: Dict[str, tuple] = {}
    for split_name, names in split_names.items():
        dataset = ColorizationDataset(image_names=names)
        loader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=False, num_workers=0)
        inputs, targets = next(iter(loader))
        expected_batch = min(16, len(names))
        if tuple(inputs.shape) != (expected_batch, 1, 160, 128):
            raise ValueError(f"{split_name}: unexpected input shape {tuple(inputs.shape)}")
        if tuple(targets.shape) != (expected_batch, 2, 160, 128):
            raise ValueError(f"{split_name}: unexpected target shape {tuple(targets.shape)}")
        batch_shapes[split_name] = (tuple(inputs.shape), tuple(targets.shape))
    return batch_shapes


def main() -> None:
    """Run all expanded-dataset checks and print a factual summary."""
    rows = read_manifest()
    if len(rows) != EXPECTED:
        raise ValueError(f"Manifest contains {len(rows)} rows instead of {EXPECTED}")
    names = {row["filename"] for row in rows}
    raw = image_files(RAW_DATA_DIR)
    if set(raw) != names:
        raise ValueError(f"Raw image set mismatch: raw={len(raw)}, manifest={len(names)}")
    manifest_hashes = {row["sha256"] for row in rows}
    actual_hashes = {sha256_file(path) for path in raw.values()}
    if manifest_hashes != actual_hashes or len(actual_hashes) != EXPECTED:
        raise ValueError("Raw SHA-256 hashes do not exactly match the approved manifest.")

    validate_processed_images(names, COLOR_DIR, "color")
    validate_processed_images(names, GRAYSCALE_DIR, "grayscale")
    validate_processed_images(names, LAB_DIR, "lab")
    split_names = validate_splits(rows)
    batch_shapes = validate_loader(split_names)
    print(f"Raw images: {len(raw)}")
    print(f"Processed color/grayscale/LAB: {len(names)}/{len(names)}/{len(names)}")
    print(f"Split counts: {', '.join(f'{name}={len(value)}' for name, value in split_names.items())}")
    print(f"DataLoader batch shapes: {batch_shapes}")
    print("Expanded dataset validation passed.")


if __name__ == "__main__":
    main()
