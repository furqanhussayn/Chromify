"""Create a read-only preview plan for expanding CHROMIFY to 5,000 images."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from config import IDENTITY_ANNOTATIONS_PATH, RAW_DATA_DIR
from src.dataset import _load_identity_annotations


SOURCE_DIR = Path(r"C:\Users\icon\Downloads\img_align_celeba\img_align_celeba")
PREVIEW_DIR = Path(__file__).resolve().parents[2] / "data" / "expansion_preview"
TARGET_IMAGE_COUNT = 5_000
EXISTING_IMAGE_COUNT = 500
ADDITIONAL_IMAGE_COUNT = TARGET_IMAGE_COUNT - EXISTING_IMAGE_COUNT
EXPANSION_RANDOM_SEED = 20261114
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _sha256_file(path: Path) -> str:
    """Hash one file at a time without loading the complete image into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_source_images() -> List[Path]:
    """Find source images recursively in deterministic filename order."""
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Original CelebA directory was not found: {SOURCE_DIR}")
    return sorted(
        path
        for path in SOURCE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _find_existing_images() -> List[Path]:
    """Find the existing selected images without modifying the selected directory."""
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Existing raw directory was not found: {RAW_DATA_DIR}")
    return sorted(
        path
        for path in RAW_DATA_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _build_identity_lookup(identity_annotations: Mapping[str, str]) -> Dict[str, str]:
    """Normalize annotation keys to filenames."""
    return {Path(filename).name: str(identity) for filename, identity in identity_annotations.items()}


def _assign_identity_groups(
    selected_names: Sequence[str],
    identity_by_filename: Mapping[str, str],
) -> Dict[str, str]:
    """Assign complete identity groups to near-target train/validation/test bins."""
    targets = {"train": 4_000, "val": 500, "test": 500}
    groups: Dict[str, List[str]] = defaultdict(list)
    for filename in selected_names:
        identity = identity_by_filename.get(filename)
        if identity is None:
            raise ValueError(f"Selected filename is missing an identity: {filename}")
        groups[identity].append(filename)

    rng = random.Random(EXPANSION_RANDOM_SEED)
    identities = list(groups)
    rng.shuffle(identities)
    tie_break_order = {identity: index for index, identity in enumerate(identities)}
    identities.sort(key=lambda identity: (-len(groups[identity]), tie_break_order[identity]))

    assigned: Dict[str, str] = {}
    counts = Counter()
    for identity in identities:
        group_size = len(groups[identity])
        available_splits = [
            split_name
            for split_name, target in targets.items()
            if counts[split_name] + group_size <= target
        ]
        if available_splits:
            split_name = max(
                available_splits,
                key=lambda name: (targets[name] - counts[name], -list(targets).index(name)),
            )
        else:
            split_name = max(
                targets,
                key=lambda name: (targets[name] - counts[name], -list(targets).index(name)),
            )

        for filename in groups[identity]:
            assigned[filename] = split_name
        counts[split_name] += group_size

    return assigned


def _write_manifest(rows: Sequence[Dict[str, str]]) -> Path:
    """Write the proposed selection manifest."""
    manifest_path = PREVIEW_DIR / "selection_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "identity_id", "sha256", "proposed_split", "source"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def create_preview() -> Dict[str, object]:
    """Build and validate the expansion plan without copying or changing images."""
    source_files = _find_source_images()
    existing_files = _find_existing_images()
    source_names = {path.name for path in source_files}
    existing_names = {path.name for path in existing_files}

    if len(existing_files) != EXISTING_IMAGE_COUNT:
        raise ValueError(
            f"Expected {EXISTING_IMAGE_COUNT} existing images, found {len(existing_files)}."
        )
    if len(existing_names) != EXISTING_IMAGE_COUNT:
        raise ValueError("Existing selected images contain duplicate filenames.")
    if not existing_names.issubset(source_names):
        missing = sorted(existing_names - source_names)
        raise ValueError(f"Existing selected images are missing from source: {missing[:10]}")

    annotations = _load_identity_annotations(IDENTITY_ANNOTATIONS_PATH)
    if annotations is None:
        raise ValueError("Identity annotations were not loaded.")
    identity_by_filename = _build_identity_lookup(annotations)
    missing_annotations = sorted(source_names - set(identity_by_filename))
    if missing_annotations:
        raise ValueError(
            f"{len(missing_annotations)} source images are missing identity annotations: "
            f"{missing_annotations[:10]}"
        )

    hash_to_names: Dict[str, List[str]] = defaultdict(list)
    for index, source_path in enumerate(source_files, 1):
        image_hash = _sha256_file(source_path)
        hash_to_names[image_hash].append(source_path.name)
        if index % 50_000 == 0:
            print(f"Hashed {index}/{len(source_files)} source images.")

    existing_hashes = {_sha256_file(path): path.name for path in existing_files}
    duplicate_existing_hashes = [
        image_hash
        for image_hash, names in hash_to_names.items()
        if len(set(names) & existing_names) > 1
    ]
    if duplicate_existing_hashes:
        raise ValueError("Existing selected images contain duplicate image contents.")

    canonical_by_hash: Dict[str, str] = {}
    excluded_duplicate_names: List[str] = []
    for image_hash, names in hash_to_names.items():
        existing_names_in_group = sorted(set(names) & existing_names)
        canonical_name = existing_names_in_group[0] if existing_names_in_group else sorted(names)[0]
        canonical_by_hash[image_hash] = canonical_name
        excluded_duplicate_names.extend(name for name in names if name != canonical_name)

    canonical_names = set(canonical_by_hash.values())
    candidate_names = sorted(canonical_names - existing_names)
    rng = random.Random(EXPANSION_RANDOM_SEED)
    rng.shuffle(candidate_names)
    additional_names = candidate_names[:ADDITIONAL_IMAGE_COUNT]
    if len(additional_names) != ADDITIONAL_IMAGE_COUNT:
        raise ValueError(
            f"Only {len(additional_names)} eligible additional images are available; "
            f"{ADDITIONAL_IMAGE_COUNT} are required."
        )

    selected_names = sorted(existing_names | set(additional_names))
    if len(selected_names) != TARGET_IMAGE_COUNT:
        raise AssertionError("The proposed selection does not contain exactly 5,000 images.")

    proposed_splits = _assign_identity_groups(selected_names, identity_by_filename)
    if set(proposed_splits) != set(selected_names):
        raise AssertionError("Every selected image must have exactly one proposed split.")

    selected_hashes = {
        image_hash
        for image_hash, canonical_name in canonical_by_hash.items()
        if canonical_name in selected_names
    }
    if len(selected_hashes) != TARGET_IMAGE_COUNT:
        raise AssertionError("The proposed selection contains duplicate image contents.")

    split_identity_sets = {
        split_name: {
            identity_by_filename[filename]
            for filename, assigned_split in proposed_splits.items()
            if assigned_split == split_name
        }
        for split_name in ("train", "val", "test")
    }
    identity_overlap = {
        f"{left}_vs_{right}": sorted(split_identity_sets[left] & split_identity_sets[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    if any(identity_overlap.values()):
        raise AssertionError("Identity overlap detected between proposed splits.")

    hash_by_name = {
        canonical_name: image_hash
        for image_hash, canonical_name in canonical_by_hash.items()
        if canonical_name in selected_names
    }
    rows = [
        {
            "filename": filename,
            "identity_id": identity_by_filename[filename],
            "sha256": hash_by_name[filename],
            "proposed_split": proposed_splits[filename],
            "source": "existing_500" if filename in existing_names else "newly_selected",
        }
        for filename in selected_names
    ]
    manifest_path = _write_manifest(rows)

    split_counts = Counter(proposed_splits.values())
    split_identity_counts = {
        split_name: len(identity_set)
        for split_name, identity_set in split_identity_sets.items()
    }
    report: Dict[str, object] = {
        "status": "preview_only",
        "random_seed": EXPANSION_RANDOM_SEED,
        "source_directory": str(SOURCE_DIR),
        "existing_directory": str(RAW_DATA_DIR),
        "annotation_file": str(IDENTITY_ANNOTATIONS_PATH),
        "target_image_count": TARGET_IMAGE_COUNT,
        "existing_images_preserved": len(existing_names),
        "new_images_selected": len(additional_names),
        "source_images_examined": len(source_files),
        "source_unique_image_hashes": len(hash_to_names),
        "source_duplicate_content_groups": sum(len(names) > 1 for names in hash_to_names.values()),
        "source_duplicate_copies_excluded": len(excluded_duplicate_names),
        "proposed_selected_unique_hashes": len(selected_hashes),
        "proposed_split_image_counts": dict(split_counts),
        "proposed_split_percentages": {
            split_name: round(count / TARGET_IMAGE_COUNT * 100, 3)
            for split_name, count in split_counts.items()
        },
        "proposed_split_identity_counts": split_identity_counts,
        "identity_overlap_counts": {
            key: len(value) for key, value in identity_overlap.items()
        },
        "selected_image_count_check": len(selected_names) == TARGET_IMAGE_COUNT,
        "zero_duplicate_hash_check": len(selected_hashes) == TARGET_IMAGE_COUNT,
        "zero_identity_overlap_check": not any(identity_overlap.values()),
        "manifest_path": str(manifest_path),
    }
    report_path = PREVIEW_DIR / "selection_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    print(json.dumps(create_preview(), indent=2))
