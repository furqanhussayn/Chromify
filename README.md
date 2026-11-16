# CHROMIFY

CHROMIFY is a lightweight machine learning project that turns grayscale face portraits into plausible color versions. The initial goal is to work with passport-style portraits, build a strong preprocessing pipeline, and prepare a clean dataset structure for future experiments.

## Objective

The project aims to estimate the color information missing from grayscale portrait images. We will focus on:

- Face-only portraits
- 128 x 160 images
- CelebA dataset preparation
- CIELAB color space modeling
- A small benchmark of classical and neural approaches

## Planned ML approaches

1. Random Forest Regression baseline
2. A basic CNN
3. A lightweight U-Net

The current work includes dataset preparation, preprocessing, and baseline
training experiments. Additional architecture implementations are organized
under the architecture-first directories described above.

## Project structure

- `data/`: local-only dataset files; see [`data/README.md`](data/README.md)
- `src/`: preprocessing, dataset, model, and training source code
- `src/models/baselines/`: classical and initial baseline implementations
- `src/models/unet_furqan/`: U-Net implementation owned by Furqan
- `src/models/resnet_kazi/`: ResNet implementation owned by Kazi
- `src/models/autoencoder_yasir/`: autoencoder implementation owned by Yasir
- `src/models/mobilenetv2_huzaifa/`: MobileNetV2 implementation owned by Huzaifa
- `src/training/`: matching architecture-first training workflows
- `checkpoints/`: local-only trained model files
- `results/`: local-only predictions, plots, and metrics
- `notebooks/`: exploratory notebooks
- `app/`: application-oriented notes or future UI code

Architecture names come first in model, training, checkpoint, and result
directory names. Existing baseline modules remain in place to avoid unnecessary
changes; new implementations should use the architecture-first directories.

## Installation

From the project root, create a virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset placement

Place your dataset images inside the folder below before running preprocessing:

```text
CHROMIFY/data/raw/
```

The raw dataset should remain untouched. The preprocessing pipeline copies and
processes the images into `data/processed/` without modifying the originals.
Dataset files are intentionally excluded from Git; see [`data/README.md`](data/README.md).

## Run preprocessing

From the project root:

```bash
python src/preprocess.py
```

This script reads images recursively from `data/raw/`, crops them to a 4:5 portrait aspect ratio, resizes them to 128 x 160, converts them to grayscale and LAB, and saves the results under `data/processed/`.

## Development status

CHROMIFY is currently in its dataset preparation stage.

The project has the foundational preprocessing pipeline, dataset loaders, and split utilities in place. The model training and deep learning experiments are planned for a later stage.

## Random Forest baseline

The first baseline is implemented with a multi-output `RandomForestRegressor`.
It samples local 5 x 5 grayscale neighborhoods from `train.txt`, predicts the
normalized LAB a and b channels, evaluates on `val.txt` and `test.txt`, and
saves its checkpoint under `checkpoints/`.

Run it from the project root:

```bash
python -m src.training.train_random_forest
```

CNN and U-Net models are not implemented yet.



