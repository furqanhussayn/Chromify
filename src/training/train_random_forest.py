"""Train and evaluate the CHROMIFY Random Forest colorization baseline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from config import (
    COLOR_DIR,
    GRAYSCALE_DIR,
    INITIAL_EPOCHS,
    RANDOM_SEED,
    RF_CHECKPOINT_PATH,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    RF_N_ESTIMATORS,
    RF_N_JOBS,
    RF_PIXELS_PER_IMAGE,
    RF_PLOTS_DIR,
    RF_PREDICTIONS_DIR,
    SPLITS_DIR,
)
from src.models.random_forest import RandomForestColorizer, RandomForestConfig, sample_training_pixels
from src.preprocessing.image_utils import load_rgb_image
from src.utils.evaluation import (
    average_metrics,
    calculate_rgb_metrics,
    reconstruct_rgb_from_prediction,
)


def read_split(split_name: str) -> List[str]:
    """Read one persisted split file."""
    path = SPLITS_DIR / f"{split_name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"Split file is empty: {path}")
    return names


def evaluate_split(
    model: RandomForestColorizer,
    image_names: List[str],
    split_name: str,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Evaluate one split and retain predictions for optional visualization."""
    metrics: List[Dict[str, float]] = []
    predictions: Dict[str, np.ndarray] = {}
    for image_name in tqdm(image_names, desc=f"Evaluating {split_name}", unit="image"):
        grayscale = cv2.imread(str(GRAYSCALE_DIR / image_name), cv2.IMREAD_GRAYSCALE)
        target_rgb = load_rgb_image(COLOR_DIR / image_name)
        if grayscale is None:
            raise ValueError(f"Could not read grayscale image: {GRAYSCALE_DIR / image_name}")
        predicted_ab = model.predict_image(grayscale)
        predicted_rgb = reconstruct_rgb_from_prediction(grayscale, predicted_ab)
        metrics.append(calculate_rgb_metrics(predicted_rgb, target_rgb))
        predictions[image_name] = predicted_rgb
    return average_metrics(metrics), predictions


def save_comparison_sheet(
    image_names: List[str],
    predictions: Dict[str, np.ndarray],
    output_path: Path,
    count: int = 10,
) -> None:
    """Save original, grayscale, and Random Forest predictions in one sheet."""
    selected = image_names[:count]
    width, height = 192, 240
    row_height = height + 24
    sheet = Image.new("RGB", (width * 3, row_height * len(selected)), "white")
    draw = ImageDraw.Draw(sheet)

    for row, image_name in enumerate(selected):
        original = Image.open(COLOR_DIR / image_name).convert("RGB").resize(
            (width, height), Image.Resampling.LANCZOS
        )
        grayscale = Image.open(GRAYSCALE_DIR / image_name).convert("L").resize(
            (width, height), Image.Resampling.LANCZOS
        ).convert("RGB")
        predicted = Image.fromarray(predictions[image_name]).resize(
            (width, height), Image.Resampling.LANCZOS
        )
        for column, image in enumerate((original, grayscale, predicted)):
            sheet.paste(image, (column * width, row * row_height))
        draw.text((4, row * row_height + height + 3), image_name, fill="black")

    for column, title in enumerate(("Original RGB", "Grayscale", "Random Forest")):
        draw.text((column * width + 4, 3), title, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def save_metric_plot(metrics: Dict[str, Dict[str, float]], output_path: Path) -> None:
    """Save a compact validation/test metric summary plot."""
    metric_names = ["mae", "mse", "psnr", "ssim"]
    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    for axis, metric_name in zip(axes, metric_names):
        split_names = list(metrics)
        axis.bar(split_names, [metrics[name][metric_name] for name in split_names])
        axis.set_title(metric_name.upper())
        axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    """Train once on train.txt, validate on val.txt, then evaluate test.txt."""
    train_names = read_split("train")
    val_names = read_split("val")
    test_names = read_split("test")

    start_time = time.perf_counter()
    features, targets = sample_training_pixels(
        train_names,
        pixels_per_image=RF_PIXELS_PER_IMAGE,
        seed=RANDOM_SEED,
    )
    print(f"Training samples: {len(features)}")

    model = RandomForestColorizer(
        RandomForestConfig(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            n_jobs=RF_N_JOBS,
        )
    )
    model.fit(features, targets)
    model.save(RF_CHECKPOINT_PATH)
    training_duration = time.perf_counter() - start_time
    print(f"Training duration: {training_duration:.2f} seconds")

    val_metrics, _ = evaluate_split(model, val_names, "validation")
    test_metrics, test_predictions = evaluate_split(model, test_names, "test")

    metrics = {
        "training_samples": len(features),
        "training_duration_seconds": training_duration,
        "validation": val_metrics,
        "test": test_metrics,
    }
    RF_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    (RF_PLOTS_DIR / "random_forest_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    save_metric_plot(
        {"validation": val_metrics, "test": test_metrics},
        RF_PLOTS_DIR / "random_forest_metrics.png",
    )
    save_comparison_sheet(
        test_names,
        test_predictions,
        RF_PREDICTIONS_DIR / "random_forest_test_comparisons.png",
        count=10,
    )
    print(json.dumps(metrics, indent=2))
    print(f"Checkpoint: {RF_CHECKPOINT_PATH}")
    print(f"Test visualizations: {RF_PREDICTIONS_DIR / 'random_forest_test_comparisons.png'}")


if __name__ == "__main__":
    main()
