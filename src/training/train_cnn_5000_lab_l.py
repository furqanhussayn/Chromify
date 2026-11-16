"""Train the existing CNN workflow on corrected LAB-L inputs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from src.training import train_cnn as workflow


workflow.CNN_CHECKPOINT_PATH = ROOT / "checkpoints" / "cnn_colorizer_5000_lab_l.pt"
workflow.RF_CHECKPOINT_PATH = ROOT / "checkpoints" / "random_forest_colorizer_5000_lab_l.pkl"
workflow.RF_PLOTS_DIR = ROOT / "results" / "plots" / "5000_lab_l"
workflow.RF_PREDICTIONS_DIR = ROOT / "results" / "predictions" / "5000_lab_l"


if __name__ == "__main__":
    workflow.main()
