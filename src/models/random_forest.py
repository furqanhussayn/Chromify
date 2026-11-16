"""Random Forest colorization baseline for CHROMIFY."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from config import (
    GRAYSCALE_DIR,
    LAB_DIR,
    RF_CHECKPOINT_PATH,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    RF_N_ESTIMATORS,
    RF_N_JOBS,
    RF_PREDICTION_CHUNK_SIZE,
    RF_RANDOM_STATE,
)
from src.preprocessing.image_utils import normalize_lab_channels


@dataclass(frozen=True)
class RandomForestConfig:
    """Hyperparameters and paths for the Random Forest baseline."""

    n_estimators: int = RF_N_ESTIMATORS
    max_depth: int = RF_MAX_DEPTH
    min_samples_leaf: int = RF_MIN_SAMPLES_LEAF
    n_jobs: int = RF_N_JOBS
    random_state: int = RF_RANDOM_STATE
    prediction_chunk_size: int = RF_PREDICTION_CHUNK_SIZE


def extract_local_features(grayscale_image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Extract flattened local grayscale neighborhoods for every pixel.

    Reflection padding gives boundary pixels a complete neighborhood without
    introducing artificial black or white values.
    """
    grayscale = np.asarray(grayscale_image, dtype=np.float32)
    if grayscale.ndim != 2:
        raise ValueError("Grayscale image must have shape (height, width).")
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("kernel_size must be a positive odd number.")

    radius = kernel_size // 2
    padded = cv2.copyMakeBorder(
        grayscale,
        radius,
        radius,
        radius,
        radius,
        borderType=cv2.BORDER_REFLECT_101,
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        (kernel_size, kernel_size),
    )
    return windows.reshape(grayscale.shape[0] * grayscale.shape[1], kernel_size * kernel_size) / 255.0


def _load_training_pair(
    image_name: str,
    grayscale_dir: Path = GRAYSCALE_DIR,
    lab_dir: Path = LAB_DIR,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one grayscale image and its normalized a/b target channels."""
    grayscale_path = grayscale_dir / image_name
    lab_path = lab_dir / image_name
    grayscale = cv2.imread(str(grayscale_path), cv2.IMREAD_GRAYSCALE)
    lab = cv2.imread(str(lab_path), cv2.IMREAD_COLOR)
    if grayscale is None:
        raise ValueError(f"Could not read grayscale image: {grayscale_path}")
    if lab is None:
        raise ValueError(f"Could not read LAB image: {lab_path}")
    if grayscale.shape != lab.shape[:2]:
        raise ValueError(
            f"Image dimensions do not match for {image_name}: "
            f"grayscale={grayscale.shape}, lab={lab.shape[:2]}"
        )

    normalized_lab = normalize_lab_channels(lab)
    return grayscale, normalized_lab[..., 1:3].reshape(-1, 2)


def sample_training_pixels(
    image_names: Sequence[str],
    pixels_per_image: int,
    seed: int = RF_RANDOM_STATE,
    grayscale_dir: Path = GRAYSCALE_DIR,
    lab_dir: Path = LAB_DIR,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample local features and a/b targets from training images only."""
    if pixels_per_image < 1:
        raise ValueError("pixels_per_image must be positive.")
    if not image_names:
        raise ValueError("At least one training image is required.")

    rng = np.random.default_rng(seed)
    feature_batches: List[np.ndarray] = []
    target_batches: List[np.ndarray] = []

    for image_name in image_names:
        grayscale, targets = _load_training_pair(image_name, grayscale_dir, lab_dir)
        features = extract_local_features(grayscale)
        sample_count = min(pixels_per_image, features.shape[0])
        indices = rng.choice(features.shape[0], size=sample_count, replace=False)
        feature_batches.append(features[indices])
        target_batches.append(targets[indices])

    return np.concatenate(feature_batches, axis=0), np.concatenate(target_batches, axis=0)


class RandomForestColorizer:
    """Multi-output Random Forest regressor for normalized Lab a/b channels."""

    def __init__(self, config: RandomForestConfig = RandomForestConfig()) -> None:
        self.config = config
        self.regressor = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            n_jobs=config.n_jobs,
            random_state=config.random_state,
        )

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "RandomForestColorizer":
        """Fit both a and b outputs using the supplied training samples."""
        if features.ndim != 2 or targets.ndim != 2 or targets.shape[1] != 2:
            raise ValueError("Expected features shape (samples, features) and targets shape (samples, 2).")
        self.regressor.fit(features, targets)
        return self

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        """Predict normalized a/b values in chunks to limit memory use."""
        if not hasattr(self.regressor, "estimators_"):
            raise RuntimeError("The Random Forest must be fitted before prediction.")
        predictions = []
        chunk_size = self.config.prediction_chunk_size
        for start in range(0, len(features), chunk_size):
            predictions.append(self.regressor.predict(features[start : start + chunk_size]))
        return np.concatenate(predictions, axis=0)

    def predict_image(self, grayscale_image: np.ndarray) -> np.ndarray:
        """Predict normalized a/b channels for one grayscale image."""
        grayscale = np.asarray(grayscale_image)
        features = extract_local_features(grayscale)
        predictions = self.predict_features(features)
        return predictions.reshape(grayscale.shape[0], grayscale.shape[1], 2)

    def save(self, path: Path = RF_CHECKPOINT_PATH) -> None:
        """Serialize the fitted model and configuration."""
        if not hasattr(self.regressor, "estimators_"):
            raise RuntimeError("Cannot save an unfitted Random Forest.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path = RF_CHECKPOINT_PATH) -> "RandomForestColorizer":
        """Load a previously fitted model checkpoint."""
        if not path.exists():
            raise FileNotFoundError(f"Random Forest checkpoint not found: {path}")
        with path.open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"Checkpoint does not contain a {cls.__name__}.")
        return model
