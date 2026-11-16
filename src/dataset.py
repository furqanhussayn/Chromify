"""Dataset utilities for CHROMIFY.

This module contains the dataset split helpers and the PyTorch dataset loader used
for grayscale-to-color prediction tasks on portrait images.
"""

from __future__ import annotations

import csv
import io
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import (
    GRAYSCALE_DIR,
    LAB_DIR,
    RANDOM_SEED,
    SPLITS_DIR,
    SUPPORTED_IMAGE_EXTENSIONS,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.preprocessing.image_utils import normalize_lab_channels


SplitDict = Dict[str, List[str]]
PathLike = Union[str, Path]


def _calculate_split_counts(
    total_groups: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    """Calculate group counts while keeping feasible splits non-empty."""
    if total_groups < 1:
        return 0, 0, 0

    requested = [train_ratio, val_ratio, test_ratio]
    if any(ratio < 0 for ratio in requested) or not np.isclose(sum(requested), 1.0):
        raise ValueError("Train, validation, and test ratios must be non-negative and sum to 1.")

    counts = [round(total_groups * ratio) for ratio in requested]
    if total_groups >= 3:
        for index in (0, 1, 2):
            if counts[index] == 0:
                counts[index] = 1

    while sum(counts) > total_groups:
        largest_index = max(range(3), key=lambda index: counts[index])
        if counts[largest_index] > (1 if total_groups >= 3 else 0):
            counts[largest_index] -= 1
        else:
            break

    while sum(counts) < total_groups:
        largest_remainder_index = max(
            range(3),
            key=lambda index: total_groups * requested[index] - counts[index],
        )
        counts[largest_remainder_index] += 1

    return counts[0], counts[1], counts[2]


def _normalize_file_name(file_name: str) -> str:
    """Return only the file name, ignoring directory paths."""
    return Path(file_name).name


def _load_identity_annotations(identity_annotations: Optional[Union[PathLike, Mapping[str, str]]]) -> Optional[Mapping[str, str]]:
    """Read a mapping of filename -> identity label from a file or dictionary."""
    if identity_annotations is None:
        return None

    if isinstance(identity_annotations, Mapping):
        return {str(key): str(value) for key, value in identity_annotations.items()}

    annotation_path = Path(identity_annotations)
    if not annotation_path.exists():
        raise FileNotFoundError(f"Identity annotation file was not found: {annotation_path}")

    raw_text = annotation_path.read_text(encoding="utf-8")
    if not raw_text.strip():
        return {}

    # Try a few common delimiters; CelebA-style files are usually whitespace or CSV-like.
    for delimiter in [",", ";", "\t", " "]:
        if delimiter == " ":
            rows = [line.split() for line in raw_text.splitlines()]
        else:
            rows = list(csv.reader(io.StringIO(raw_text), delimiter=delimiter))
        valid_rows = [row for row in rows if len(row) >= 2]
        if valid_rows:
            mapping: Dict[str, str] = {}
            for row in valid_rows:
                if row[0].strip().lower() in {"filename", "image", "image_name", "file", "name", "id"}:
                    continue
                mapping[_normalize_file_name(row[0])] = row[1].strip()
            return mapping

    raise ValueError(
        f"Could not parse identity annotation file '{annotation_path}'. "
        "Expected a CSV-like file or a two-column text file with filename and identity ID."
    )


def split_dataset(
    image_paths: Sequence[PathLike],
    identity_annotations: Optional[Union[PathLike, Mapping[str, str]]] = None,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
    output_dir: PathLike = SPLITS_DIR,
    allow_image_level_split: bool = False,
) -> SplitDict:
    """Split images into train/validation/test sets.

    The default behavior uses identity-based grouping. If identity annotations are
    unavailable and `allow_image_level_split` is false, a clear error is raised
    to avoid silently creating inconsistent dataset splits.
    """
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    file_names = sorted({_normalize_file_name(path) for path in image_paths})

    if identity_annotations is not None:
        identity_map = _load_identity_annotations(identity_annotations)
        if not identity_map:
            raise ValueError("Identity annotation file exists but no valid rows were found.")

        identity_to_images: Dict[str, List[str]] = defaultdict(list)
        identity_by_filename = {
            _normalize_file_name(file_name): str(identity)
            for file_name, identity in identity_map.items()
        }
        identity_by_stem = {
            Path(file_name).stem: str(identity)
            for file_name, identity in identity_map.items()
        }
        unknown_count = 0

        for file_name in file_names:
            identity_value = identity_by_filename.get(file_name)
            if identity_value is None:
                identity_value = identity_by_stem.get(Path(file_name).stem)
            if identity_value is None:
                unknown_count += 1
                continue
            identity_to_images[identity_value].append(file_name)

        if unknown_count > 0:
            raise ValueError(
                f"{unknown_count} image(s) are missing identity labels and cannot be split safely. "
                "Provide a complete annotation file before splitting."
            )

        if not identity_to_images:
            raise ValueError("No valid identity groups were found in the annotation file.")

        identities = list(identity_to_images.keys())
        rng = random.Random(seed)
        rng.shuffle(identities)

        train_count, val_count, test_count = _calculate_split_counts(
            len(identities), train_ratio, val_ratio, test_ratio
        )

        split_map: SplitDict = {"train": [], "val": [], "test": []}
        for index, identity in enumerate(identities):
            image_group = identity_to_images[identity]
            if index < train_count:
                split_map["train"].extend(image_group)
            elif index < train_count + val_count:
                split_map["val"].extend(image_group)
            else:
                split_map["test"].extend(image_group)

        split_map["train"] = sorted(split_map["train"])
        split_map["val"] = sorted(split_map["val"])
        split_map["test"] = sorted(split_map["test"])
    else:
        if not allow_image_level_split:
            raise ValueError(
                "Identity annotations are required for dataset splitting. "
                "Provide an identity mapping or pass allow_image_level_split=True to create a random image-level split."
            )

        print(
            "WARNING: No identity annotation file was supplied. Falling back to image-level random splitting "
            "because allow_image_level_split=True was set."
        )

        rng = random.Random(seed)
        shuffled_names = file_names[:]
        rng.shuffle(shuffled_names)

        train_count, val_count, test_count = _calculate_split_counts(
            len(shuffled_names), train_ratio, val_ratio, test_ratio
        )

        split_map = {
            "train": sorted(shuffled_names[:train_count]),
            "val": sorted(shuffled_names[train_count : train_count + val_count]),
            "test": sorted(shuffled_names[train_count + val_count : train_count + val_count + test_count]),
        }

    _save_split_files(split_map, output_directory)
    return split_map


def _save_split_files(split_map: SplitDict, output_dir: Path) -> None:
    """Write split filenames into text files under the dataset split directory."""
    for split_name, filenames in split_map.items():
        file_path = output_dir / f"{split_name}.txt"
        with file_path.open("w", encoding="utf-8") as handle:
            for file_name in filenames:
                handle.write(f"{file_name}\n")


class ColorizationDataset(Dataset):
    """A dataset that pairs grayscale portraits with their corresponding Lab targets."""

    def __init__(
        self,
        grayscale_dir: PathLike = GRAYSCALE_DIR,
        lab_dir: PathLike = LAB_DIR,
        image_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.grayscale_dir = Path(grayscale_dir)
        self.lab_dir = Path(lab_dir)

        if image_names is None:
            self.image_names = self._find_matching_image_names()
        else:
            self.image_names = [str(name) for name in image_names]

        if not self.image_names:
            raise FileNotFoundError(
                f"No matching grayscale/LAB image pairs were found under {self.grayscale_dir} and {self.lab_dir}."
            )

    def _find_matching_image_names(self) -> List[str]:
        if not self.grayscale_dir.exists():
            raise FileNotFoundError(f"Grayscale dataset directory was not found: {self.grayscale_dir}")
        if not self.lab_dir.exists():
            raise FileNotFoundError(f"LAB dataset directory was not found: {self.lab_dir}")

        grayscale_files = {
            path.name
            for path in self.grayscale_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        }
        lab_files = {
            path.name
            for path in self.lab_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        }
        return sorted(grayscale_files & lab_files)

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int):
        file_name = self.image_names[index]
        grayscale_path = self.grayscale_dir / file_name
        lab_path = self.lab_dir / file_name

        if not grayscale_path.exists():
            raise FileNotFoundError(f"Missing grayscale image for index {index}: {grayscale_path}")
        if not lab_path.exists():
            raise FileNotFoundError(f"Missing LAB image for index {index}: {lab_path}")

        grayscale_image = cv2.imread(str(grayscale_path), cv2.IMREAD_GRAYSCALE)
        lab_image = cv2.imread(str(lab_path), cv2.IMREAD_COLOR)

        if grayscale_image is None:
            raise ValueError(f"Could not open grayscale image: {grayscale_path}")
        if lab_image is None:
            raise ValueError(f"Could not open LAB image: {lab_path}")

        grayscale = np.asarray(grayscale_image, dtype=np.float32)
        lab = np.asarray(lab_image, dtype=np.float32)

        # Input: grayscale L channel, normalized to [0, 1]
        input_tensor = np.expand_dims(grayscale / 255.0, axis=0).astype(np.float32)

        # Target: LAB a and b channels; normalized according to the project convention.
        normalized_lab = normalize_lab_channels(lab)
        target_tensor = normalized_lab[..., 1:3].transpose(2, 0, 1).astype(np.float32)

        return torch.from_numpy(input_tensor), torch.from_numpy(target_tensor)
