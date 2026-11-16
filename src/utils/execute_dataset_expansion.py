"""Execute the approved CHROMIFY 5,000-image expansion manifest."""

from __future__ import annotations

import csv
import hashlib
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = Path(r"C:\Users\icon\Downloads\img_align_celeba\img_align_celeba")
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = PROJECT_ROOT / "data" / "expansion_preview" / "selection_manifest.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
EXPECTED_COUNT = 5_000


def sha256_file(path: Path) -> str:
    """Hash a file incrementally without loading it fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> List[Dict[str, str]]:
    """Load and validate the approved manifest."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Approved manifest was not found: {MANIFEST_PATH}")
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"filename", "identity_id", "sha256", "proposed_split", "source"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest is missing required columns: {sorted(required)}")
    if len(rows) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} manifest rows, found {len(rows)}.")
    names = [row["filename"] for row in rows]
    if len(set(names)) != EXPECTED_COUNT:
        raise ValueError("Manifest contains duplicate filenames.")
    if any(row["proposed_split"] not in {"train", "val", "test"} for row in rows):
        raise ValueError("Manifest contains an invalid proposed split.")
    return rows


def copy_and_verify_images(rows: List[Dict[str, str]]) -> None:
    """Copy missing approved images and verify every selected hash."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, 1):
        filename = row["filename"]
        source = SOURCE_DIR / filename
        destination = RAW_DIR / filename
        if not source.is_file():
            raise FileNotFoundError(f"Manifest image is missing from source: {source}")
        source_hash = sha256_file(source)
        if source_hash != row["sha256"]:
            raise ValueError(f"Source hash does not match manifest for {filename}")

        if destination.exists():
            destination_hash = sha256_file(destination)
            if destination_hash != row["sha256"]:
                raise ValueError(
                    f"Existing destination hash does not match manifest; refusing overwrite: "
                    f"{destination}"
                )
        else:
            shutil.copy2(source, destination)

        if sha256_file(destination) != row["sha256"]:
            raise ValueError(f"Copied image failed hash verification: {destination}")
        if index % 500 == 0:
            print(f"Verified {index}/{len(rows)} approved images.")


def validate_rows(rows: List[Dict[str, str]]) -> None:
    """Validate counts, hashes, filenames, and identity isolation."""
    split_names: Dict[str, List[str]] = defaultdict(list)
    split_identities: Dict[str, set[str]] = defaultdict(set)
    hashes = set()
    for row in rows:
        destination = RAW_DIR / row["filename"]
        if not destination.is_file():
            raise FileNotFoundError(f"Selected raw image is missing: {destination}")
        image_hash = sha256_file(destination)
        if image_hash != row["sha256"]:
            raise ValueError(f"Final raw hash mismatch for {row['filename']}")
        hashes.add(image_hash)
        split_names[row["proposed_split"]].append(row["filename"])
        split_identities[row["proposed_split"]].add(row["identity_id"])

    if len(hashes) != EXPECTED_COUNT:
        raise ValueError("Selected images contain duplicate image contents.")
    expected_counts = {"train": 4_000, "val": 500, "test": 500}
    actual_counts = {name: len(split_names[name]) for name in expected_counts}
    if actual_counts != expected_counts:
        raise ValueError(f"Approved split counts do not match: {actual_counts}")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_identities[left] & split_identities[right]
        if overlap:
            raise ValueError(f"Identity overlap between {left} and {right}: {len(overlap)}")


def backup_and_install_splits(rows: List[Dict[str, str]]) -> Path:
    """Back up existing split files and install approved manifest assignments."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = SPLITS_DIR / f"backup_before_5000_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    split_names: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        split_names[row["proposed_split"]].append(row["filename"])

    for split_name in ("train", "val", "test"):
        current = SPLITS_DIR / f"{split_name}.txt"
        if current.exists():
            shutil.copy2(current, backup_dir / current.name)
        target = SPLITS_DIR / f"{split_name}.txt"
        target.write_text(
            "".join(f"{name}\n" for name in sorted(split_names[split_name])),
            encoding="utf-8",
        )
    return backup_dir


def main() -> None:
    """Execute, validate, and install the approved expansion."""
    rows = load_manifest()
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Source directory was not found: {SOURCE_DIR}")
    validate_rows(rows) if all((RAW_DIR / row["filename"]).is_file() for row in rows) else None
    copy_and_verify_images(rows)
    validate_rows(rows)
    backup_dir = backup_and_install_splits(rows)
    print(f"Expansion complete: {len(rows)} images verified in {RAW_DIR}")
    print(f"Split files backed up under: {backup_dir}")
    print("Approved identity-aware split assignments installed.")


if __name__ == "__main__":
    main()
