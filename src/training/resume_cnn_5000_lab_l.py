"""Resume the corrected LAB-L CNN experiment without starting over."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from config import CNN_BATCH_SIZE, CNN_LEARNING_RATE, RANDOM_SEED
from src.models.cnn import ColorizationCNN
from src.training import train_cnn as workflow


workflow.CNN_CHECKPOINT_PATH = ROOT / "checkpoints" / "cnn_colorizer_5000_lab_l.pt"
workflow.RF_CHECKPOINT_PATH = ROOT / "checkpoints" / "random_forest_colorizer_5000_lab_l.pkl"
workflow.RF_PLOTS_DIR = ROOT / "results" / "plots" / "5000_lab_l"
workflow.RF_PREDICTIONS_DIR = ROOT / "results" / "predictions" / "5000_lab_l"


def main() -> None:
    """Resume from the existing epoch-3 checkpoint and close out the CNN run."""
    workflow.set_random_seeds(RANDOM_SEED)
    train_names = workflow.read_split("train")
    val_names = workflow.read_split("val")
    train_loader = workflow.make_loader(train_names, shuffle=True)
    val_loader = workflow.make_loader(val_names, shuffle=False)
    device = torch.device("cpu")
    model = ColorizationCNN().to(device)
    optimizer = Adam(model.parameters(), lr=CNN_LEARNING_RATE)
    criterion = nn.L1Loss()

    start = time.perf_counter()
    history, resume_metadata = workflow.resume_training(
        model,
        optimizer,
        train_loader,
        val_loader,
        criterion,
        device,
        checkpoint_path=workflow.CNN_CHECKPOINT_PATH,
        additional_epochs=10,
        patience=3,
    )
    duration = time.perf_counter() - start

    workflow.RF_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    (workflow.RF_PLOTS_DIR / "cnn_resume_history.json").write_text(
        json.dumps(
            {
                "training_duration_seconds": duration,
                "history": history,
                "resume": resume_metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    workflow._save_training_history(
        history,
        workflow.RF_PLOTS_DIR / "cnn_resume_history.png",
    )
    print(json.dumps({"training_duration_seconds": duration, **resume_metadata}, indent=2))


if __name__ == "__main__":
    main()
