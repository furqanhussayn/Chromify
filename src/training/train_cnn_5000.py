"""Run the established CNN workflow with 5,000-image artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from src.training import train_cnn as workflow


workflow.CNN_CHECKPOINT_PATH = ROOT / "checkpoints" / "cnn_colorizer_5000.pt"
workflow.RF_CHECKPOINT_PATH = ROOT / "checkpoints" / "random_forest_colorizer_5000.pkl"
workflow.RF_PLOTS_DIR = ROOT / "results" / "plots" / "5000"
workflow.RF_PREDICTIONS_DIR = ROOT / "results" / "predictions" / "5000"


if __name__ == "__main__":
    workflow.main()
