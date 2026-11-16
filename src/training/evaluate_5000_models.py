"""Evaluate the completed 5,000-image Random Forest and CNN checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from src.training import train_cnn as workflow
from src.models.cnn import ColorizationCNN, count_parameters


workflow.CNN_CHECKPOINT_PATH = ROOT / "checkpoints" / "cnn_colorizer_5000.pt"
workflow.RF_CHECKPOINT_PATH = ROOT / "checkpoints" / "random_forest_colorizer_5000.pkl"
workflow.RF_PLOTS_DIR = ROOT / "results" / "plots" / "5000"
workflow.RF_PREDICTIONS_DIR = ROOT / "results" / "predictions" / "5000"


def main() -> None:
    """Run held-out evaluation and write distinct 5,000-image artifacts."""
    train_names = workflow.read_split("train")
    val_names = workflow.read_split("val")
    test_names = workflow.read_split("test")
    device = torch.device("cpu")

    model = ColorizationCNN().to(device)
    checkpoint_metadata = workflow.load_best_checkpoint(model, device)
    validation_metrics, _ = workflow.evaluate_cnn(model, val_names, device)
    cnn_test_metrics, cnn_test_predictions = workflow.evaluate_cnn(model, test_names, device)

    random_forest = workflow.RandomForestColorizer.load(workflow.RF_CHECKPOINT_PATH)
    rf_test_metrics, rf_test_predictions = workflow._evaluate_random_forest(
        random_forest,
        test_names,
    )

    metrics = {
        "device": str(device),
        "parameter_count": count_parameters(model),
        "best_epoch": checkpoint_metadata["epoch"],
        "best_validation_loss": checkpoint_metadata["validation_loss"],
        "validation": validation_metrics,
        "test": cnn_test_metrics,
        "random_forest_test": rf_test_metrics,
        "training_samples": len(train_names) * 150,
    }
    workflow.RF_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    (workflow.RF_PLOTS_DIR / "cnn_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    workflow.save_comparison_plot(
        rf_test_metrics,
        cnn_test_metrics,
        workflow.RF_PLOTS_DIR / "random_forest_vs_cnn_metrics.png",
    )
    workflow.save_comparison_sheet(
        test_names,
        rf_test_predictions,
        cnn_test_predictions,
        workflow.RF_PREDICTIONS_DIR / "random_forest_vs_cnn_test_comparisons.png",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
