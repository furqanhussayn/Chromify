"""Train, evaluate, and visualize the CHROMIFY CNN colorization baseline."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    CNN_BATCH_SIZE,
    CNN_CHECKPOINT_PATH,
    CNN_EARLY_STOPPING_PATIENCE,
    CNN_EPOCHS,
    CNN_LEARNING_RATE,
    COLOR_DIR,
    GRAYSCALE_DIR,
    RANDOM_SEED,
    RF_CHECKPOINT_PATH,
    RF_PLOTS_DIR,
    RF_PREDICTIONS_DIR,
    SPLITS_DIR,
)
from src.dataset import ColorizationDataset
from src.models.cnn import ColorizationCNN, count_parameters
from src.models.random_forest import RandomForestColorizer
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


def set_random_seeds(seed: int) -> None:
    """Set reproducible seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(image_names: List[str], shuffle: bool) -> DataLoader:
    """Create a deterministic DataLoader using the existing dataset implementation."""
    dataset = ColorizationDataset(image_names=image_names)
    return DataLoader(
        dataset,
        batch_size=min(CNN_BATCH_SIZE, len(dataset)),
        shuffle=shuffle,
        num_workers=0,
    )


def run_epoch(
    model: ColorizationCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> float:
    """Run one training or validation epoch and return mean L1 loss."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = inputs.shape[0]
            total_loss += loss.item() * batch_size
            total_items += batch_size

    return total_loss / total_items


def save_checkpoint(
    model: ColorizationCNN,
    optimizer: Adam,
    epoch: int,
    validation_loss: float,
) -> None:
    """Save the best CNN state without touching the Random Forest checkpoint."""
    CNN_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "validation_loss": validation_loss,
        },
        CNN_CHECKPOINT_PATH,
    )


def load_best_checkpoint(model: ColorizationCNN, device: torch.device) -> Dict[str, float]:
    """Load and return metadata from the saved best checkpoint."""
    if not CNN_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"CNN checkpoint not found: {CNN_CHECKPOINT_PATH}")
    checkpoint = torch.load(CNN_CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return {
        "epoch": int(checkpoint["epoch"]),
        "validation_loss": float(checkpoint["validation_loss"]),
    }


def evaluate_cnn(
    model: ColorizationCNN,
    image_names: List[str],
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Evaluate RGB metrics for a split using batched CNN inference."""
    loader = make_loader(image_names, shuffle=False)
    model.eval()
    metrics: List[Dict[str, float]] = []
    predictions: Dict[str, np.ndarray] = {}
    offset = 0

    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc="Evaluating CNN", unit="batch"):
            predicted_ab = model(inputs.to(device)).cpu().numpy()
            grayscale_batch = (inputs[:, 0].numpy() * 255.0).astype(np.uint8)
            for batch_index in range(predicted_ab.shape[0]):
                image_name = image_names[offset + batch_index]
                grayscale = grayscale_batch[batch_index]
                predicted_rgb = reconstruct_rgb_from_prediction(
                    grayscale,
                    predicted_ab[batch_index].transpose(1, 2, 0),
                )
                target_rgb = load_rgb_image(COLOR_DIR / image_name)
                metrics.append(calculate_rgb_metrics(predicted_rgb, target_rgb))
                predictions[image_name] = predicted_rgb
            offset += predicted_ab.shape[0]

    return average_metrics(metrics), predictions


def save_comparison_sheet(
    image_names: List[str],
    rf_predictions: Dict[str, np.ndarray],
    cnn_predictions: Dict[str, np.ndarray],
    output_path: Path,
    count: int = 10,
) -> None:
    """Save original, grayscale, RF, and CNN results for the same test images."""
    selected = image_names[:count]
    width, height = 160, 200
    row_height = height + 24
    sheet = Image.new("RGB", (width * 4, row_height * len(selected)), "white")
    draw = ImageDraw.Draw(sheet)

    for row, image_name in enumerate(selected):
        original = Image.open(COLOR_DIR / image_name).convert("RGB").resize(
            (width, height), Image.Resampling.LANCZOS
        )
        grayscale = Image.open(GRAYSCALE_DIR / image_name).convert("L").resize(
            (width, height), Image.Resampling.LANCZOS
        ).convert("RGB")
        rf_image = Image.fromarray(rf_predictions[image_name]).resize(
            (width, height), Image.Resampling.LANCZOS
        )
        cnn_image = Image.fromarray(cnn_predictions[image_name]).resize(
            (width, height), Image.Resampling.LANCZOS
        )
        for column, image in enumerate((original, grayscale, rf_image, cnn_image)):
            sheet.paste(image, (column * width, row * row_height))
        draw.text((4, row * row_height + height + 3), image_name, fill="black")

    for column, title in enumerate(("Original", "Grayscale", "Random Forest", "CNN")):
        draw.text((column * width + 4, 3), title, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def save_comparison_plot(
    random_forest_metrics: Dict[str, float],
    cnn_metrics: Dict[str, float],
    output_path: Path,
) -> None:
    """Plot test metrics for the Random Forest and CNN baselines."""
    metric_names = ["mae", "mse", "psnr", "ssim"]
    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    for axis, metric_name in zip(axes, metric_names):
        axis.bar(
            ["Random Forest", "CNN"],
            [random_forest_metrics[metric_name], cnn_metrics[metric_name]],
        )
        axis.set_title(metric_name.upper())
        axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    """Train on train.txt, select the best epoch on val.txt, then evaluate test.txt."""
    set_random_seeds(RANDOM_SEED)
    train_names = read_split("train")
    val_names = read_split("val")
    test_names = read_split("test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ColorizationCNN().to(device)
    optimizer = Adam(model.parameters(), lr=CNN_LEARNING_RATE)
    criterion = nn.L1Loss()
    train_loader = make_loader(train_names, shuffle=True)
    val_loader = make_loader(val_names, shuffle=False)
    history = {"train_loss": [], "validation_loss": []}
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    start_time = time.perf_counter()
    for epoch in range(1, CNN_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss = run_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        print(
            f"Epoch {epoch:02d}/{CNN_EPOCHS}: "
            f"train_loss={train_loss:.6f}, validation_loss={validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, epoch, validation_loss)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= CNN_EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    training_duration = time.perf_counter() - start_time
    checkpoint_metadata = load_best_checkpoint(model, device)
    validation_metrics, _ = evaluate_cnn(model, val_names, device)
    cnn_test_metrics, cnn_test_predictions = evaluate_cnn(model, test_names, device)

    random_forest = RandomForestColorizer.load(RF_CHECKPOINT_PATH)
    rf_test_metrics, rf_test_predictions = _evaluate_random_forest(
        random_forest,
        test_names,
    )

    metrics = {
        "device": str(device),
        "parameter_count": count_parameters(model),
        "training_duration_seconds": training_duration,
        "best_epoch": checkpoint_metadata["epoch"],
        "best_validation_loss": checkpoint_metadata["validation_loss"],
        "validation": validation_metrics,
        "test": cnn_test_metrics,
        "random_forest_test": rf_test_metrics,
    }
    RF_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    (RF_PLOTS_DIR / "cnn_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _save_training_history(history, RF_PLOTS_DIR / "cnn_training_history.png")
    save_comparison_plot(
        rf_test_metrics,
        cnn_test_metrics,
        RF_PLOTS_DIR / "random_forest_vs_cnn_metrics.png",
    )
    save_comparison_sheet(
        test_names,
        rf_test_predictions,
        cnn_test_predictions,
        RF_PREDICTIONS_DIR / "random_forest_vs_cnn_test_comparisons.png",
    )
    print(json.dumps(metrics, indent=2))
    print(f"CNN checkpoint: {CNN_CHECKPOINT_PATH}")


def _evaluate_random_forest(
    model: RandomForestColorizer,
    image_names: List[str],
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Evaluate the finalized Random Forest on the same test images."""
    metrics: List[Dict[str, float]] = []
    predictions: Dict[str, np.ndarray] = {}
    for image_name in tqdm(image_names, desc="Evaluating Random Forest", unit="image"):
        grayscale = cv2.imread(str(GRAYSCALE_DIR / image_name), cv2.IMREAD_GRAYSCALE)
        if grayscale is None:
            raise ValueError(f"Could not read grayscale image: {GRAYSCALE_DIR / image_name}")
        predicted_rgb = reconstruct_rgb_from_prediction(
            grayscale,
            model.predict_image(grayscale),
        )
        target_rgb = load_rgb_image(COLOR_DIR / image_name)
        metrics.append(calculate_rgb_metrics(predicted_rgb, target_rgb))
        predictions[image_name] = predicted_rgb
    return average_metrics(metrics), predictions


def _save_training_history(history: Dict[str, List[float]], output_path: Path) -> None:
    """Save training and validation L1 loss curves."""
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(history["train_loss"], label="Train")
    axis.plot(history["validation_loss"], label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("L1 loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
