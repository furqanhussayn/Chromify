"""Evaluate and summarize the corrected LAB-L RF and CNN baselines."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from config import COLOR_DIR, GRAYSCALE_DIR, LAB_DIR
from src.dataset import ColorizationDataset
from src.models.cnn import ColorizationCNN, count_parameters
from src.models.random_forest import RandomForestColorizer
from src.preprocessing.image_utils import normalize_lab_channels
from src.training import train_cnn as workflow


workflow.CNN_CHECKPOINT_PATH = ROOT / "checkpoints" / "cnn_colorizer_5000_lab_l.pt"
workflow.RF_CHECKPOINT_PATH = ROOT / "checkpoints" / "random_forest_colorizer_5000_lab_l.pkl"
workflow.RF_PLOTS_DIR = ROOT / "results" / "plots" / "5000_lab_l"
workflow.RF_PREDICTIONS_DIR = ROOT / "results" / "predictions" / "5000_lab_l"

CHROMA_BINS = np.array([0, 2, 5, 10, 20, 40, 80, 160, 320], dtype=np.float64)


class ChromaAccumulator:
    """Accumulate a/b and chroma statistics without retaining all pixels."""

    def __init__(self) -> None:
        self.count = 0
        self.sum = np.zeros(2, dtype=np.float64)
        self.sum_squared = np.zeros(2, dtype=np.float64)
        self.chroma_sum = 0.0
        self.chroma_squared_sum = 0.0
        self.neutral_count = 0
        self.high_chroma_count = 0
        self.histogram = np.zeros(len(CHROMA_BINS) - 1, dtype=np.int64)

    def add(self, ab: np.ndarray) -> None:
        values = np.asarray(ab, dtype=np.float64).reshape(-1, 2)
        chroma = np.sqrt(np.sum(values * values, axis=1))
        self.count += len(values)
        self.sum += values.sum(axis=0)
        self.sum_squared += (values * values).sum(axis=0)
        self.chroma_sum += float(chroma.sum())
        self.chroma_squared_sum += float((chroma * chroma).sum())
        self.neutral_count += int(np.sum(chroma < 5.0))
        self.high_chroma_count += int(np.sum(chroma > 40.0))
        self.histogram += np.histogram(chroma, bins=CHROMA_BINS)[0]

    def to_dict(self) -> Dict[str, object]:
        mean = self.sum / self.count
        standard_deviation = np.sqrt(np.maximum(self.sum_squared / self.count - mean * mean, 0.0))
        chroma_mean = self.chroma_sum / self.count
        chroma_std = np.sqrt(max(self.chroma_squared_sum / self.count - chroma_mean**2, 0.0))
        return {
            "a_mean": float(mean[0]),
            "a_std": float(standard_deviation[0]),
            "b_mean": float(mean[1]),
            "b_std": float(standard_deviation[1]),
            "mean_chroma": float(chroma_mean),
            "chroma_std": float(chroma_std),
            "near_neutral_pct": float(100.0 * self.neutral_count / self.count),
            "high_chroma_over_40_pct": float(100.0 * self.high_chroma_count / self.count),
            "chroma_histogram_counts": self.histogram.tolist(),
            "chroma_histogram_edges": CHROMA_BINS.tolist(),
            "pixels": self.count,
        }


def read_split(name: str) -> list[str]:
    """Read an existing split without modifying it."""
    path = ROOT / "data" / "splits" / f"{name}.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def accumulate_chroma(
    names: list[str],
    random_forest: RandomForestColorizer,
    cnn: ColorizationCNN,
) -> Dict[str, Dict[str, object]]:
    """Collect ground-truth, RF, and CNN chroma statistics for one split."""
    accumulators = {
        "ground_truth": ChromaAccumulator(),
        "random_forest": ChromaAccumulator(),
        "cnn": ChromaAccumulator(),
    }
    dataset = ColorizationDataset(image_names=names)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    offset = 0
    with torch.no_grad():
        for inputs, _ in loader:
            cnn_predictions = cnn(inputs).cpu().numpy().transpose(0, 2, 3, 1) * 110.0
            grayscale_batch = (inputs[:, 0].numpy() * 255.0).astype(np.uint8)
            for batch_index, grayscale in enumerate(grayscale_batch):
                name = names[offset + batch_index]
                lab = cv2.imread(str(LAB_DIR / name), cv2.IMREAD_COLOR)
                if lab is None:
                    raise ValueError(f"Could not read LAB target: {LAB_DIR / name}")
                ground_truth_ab = lab[..., 1:3].astype(np.float32) - 128.0
                random_forest_ab = random_forest.predict_image(grayscale) * 110.0
                accumulators["ground_truth"].add(ground_truth_ab)
                accumulators["random_forest"].add(random_forest_ab)
                accumulators["cnn"].add(cnn_predictions[batch_index])
            offset += len(grayscale_batch)
    return {name: accumulator.to_dict() for name, accumulator in accumulators.items()}


def save_chroma_plot(stats: Dict[str, Dict[str, object]], path: Path) -> None:
    """Save a histogram comparison for ground truth and both predictions."""
    labels = {"ground_truth": "Ground truth", "random_forest": "Random Forest", "cnn": "CNN"}
    figure, axes = plt.subplots(1, 2, figsize=(13, 4))
    for key, label in labels.items():
        values = stats[key]["chroma_histogram_counts"]
        edges = stats[key]["chroma_histogram_edges"]
        centers = (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2
        axes[0].plot(centers, values, marker="o", label=label)
        axes[1].bar(
            [label],
            [stats[key]["mean_chroma"]],
            label=label,
        )
    axes[0].set_title("Chroma distribution")
    axes[0].set_xlabel("Chroma magnitude")
    axes[0].set_ylabel("Pixels")
    axes[0].legend()
    axes[1].set_title("Mean chroma")
    axes[1].set_ylabel("Mean chroma")
    axes[1].tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_comparison_table(
    metrics: Dict[str, Dict[str, float]],
    stats: Dict[str, Dict[str, object]],
    path: Path,
) -> None:
    """Save a single visual table of RGB and chroma comparisons."""
    columns = ["Ground truth", "Random Forest", "CNN"]
    rows = [
        ("Test MAE", "-", f"{metrics['random_forest']['mae']:.5f}", f"{metrics['cnn']['mae']:.5f}"),
        ("Test PSNR", "-", f"{metrics['random_forest']['psnr']:.3f}", f"{metrics['cnn']['psnr']:.3f}"),
        ("Test SSIM", "-", f"{metrics['random_forest']['ssim']:.5f}", f"{metrics['cnn']['ssim']:.5f}"),
        ("a std", f"{stats['ground_truth']['a_std']:.3f}", f"{stats['random_forest']['a_std']:.3f}", f"{stats['cnn']['a_std']:.3f}"),
        ("b std", f"{stats['ground_truth']['b_std']:.3f}", f"{stats['random_forest']['b_std']:.3f}", f"{stats['cnn']['b_std']:.3f}"),
        ("Mean chroma", f"{stats['ground_truth']['mean_chroma']:.3f}", f"{stats['random_forest']['mean_chroma']:.3f}", f"{stats['cnn']['mean_chroma']:.3f}"),
        ("High chroma > 40 (%)", f"{stats['ground_truth']['high_chroma_over_40_pct']:.3f}", f"{stats['random_forest']['high_chroma_over_40_pct']:.3f}", f"{stats['cnn']['high_chroma_over_40_pct']:.3f}"),
        ("Near-neutral (%)", f"{stats['ground_truth']['near_neutral_pct']:.3f}", f"{stats['random_forest']['near_neutral_pct']:.3f}", f"{stats['cnn']['near_neutral_pct']:.3f}"),
    ]
    figure, axis = plt.subplots(figsize=(11, 4.5))
    axis.axis("off")
    table = axis.table(
        cellText=[row[1:] for row in rows],
        rowLabels=[row[0] for row in rows],
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Evaluate the finalized corrected-input CNN and compare both baselines."""
    test_names = read_split("test")
    val_names = read_split("val")
    device = torch.device("cpu")
    cnn = ColorizationCNN().to(device)
    checkpoint = torch.load(workflow.CNN_CHECKPOINT_PATH, map_location=device, weights_only=False)
    cnn.load_state_dict(checkpoint["model_state_dict"])
    cnn.eval()
    random_forest = RandomForestColorizer.load(workflow.RF_CHECKPOINT_PATH)

    validation_metrics, _ = workflow.evaluate_cnn(cnn, val_names, device)
    cnn_test_metrics, cnn_predictions = workflow.evaluate_cnn(cnn, test_names, device)
    rf_test_metrics, rf_predictions = workflow._evaluate_random_forest(random_forest, test_names)
    stats = accumulate_chroma(test_names, random_forest, cnn)
    workflow.RF_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    workflow.RF_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "device": "cpu",
        "parameter_count": count_parameters(cnn),
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_loss": float(checkpoint["validation_loss"]),
        "validation": validation_metrics,
        "test": cnn_test_metrics,
        "random_forest_test": rf_test_metrics,
    }
    (workflow.RF_PLOTS_DIR / "cnn_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    workflow.save_comparison_plot(
        rf_test_metrics,
        cnn_test_metrics,
        workflow.RF_PLOTS_DIR / "cnn_metrics.png",
    )
    workflow.save_comparison_sheet(
        test_names,
        rf_predictions,
        cnn_predictions,
        workflow.RF_PREDICTIONS_DIR / "cnn_test_comparisons.png",
    )
    chroma_report = {
        "split": "test",
        "ground_truth": stats["ground_truth"],
        "random_forest": stats["random_forest"],
        "cnn": stats["cnn"],
    }
    (workflow.RF_PLOTS_DIR / "cnn_chroma_report.json").write_text(
        json.dumps(chroma_report, indent=2),
        encoding="utf-8",
    )
    save_chroma_plot(chroma_report, workflow.RF_PLOTS_DIR / "cnn_chroma_histogram.png")
    comparison = {"metrics": metrics, "test_chroma": chroma_report}
    (workflow.RF_PLOTS_DIR / "corrected_baseline_comparison.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )
    save_comparison_table(
        {"random_forest": rf_test_metrics, "cnn": cnn_test_metrics},
        stats,
        workflow.RF_PLOTS_DIR / "corrected_baseline_comparison.png",
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
