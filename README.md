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

The current work includes dataset preparation, preprocessing, and completed
corrected Random Forest and CNN baseline experiments. Additional architecture
implementations are organized under the architecture-first directories described
above.

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

The dataset pipeline and corrected baseline experiments are complete. The next
planned architecture is a lightweight U-Net, followed by additional
architecture-specific experiments.

## Random Forest baseline

The first baseline is implemented with a multi-output `RandomForestRegressor`.
It samples local 5 x 5 grayscale neighborhoods from `train.txt`, predicts the
normalized LAB a and b channels, evaluates on `val.txt` and `test.txt`, and
saves its checkpoint under `checkpoints/`.

Run it from the project root:

```bash
python -m src.training.train_random_forest
```

## Corrected baselines complete

The corrected baselines use the CIELAB L channel as input and predict the
normalized a and b channels. Both models were trained on the identity-disjoint
5,000-image dataset with 4,000 training, 500 validation, and 500 test images.

Corrected Random Forest test results:

- MAE: `0.04289`
- PSNR: `24.946 dB`
- SSIM: `0.92237`

Corrected CNN test results:

- MAE: `0.04281`
- PSNR: `24.962 dB`
- SSIM: `0.93518`

Test chroma diagnostics:

| Statistic | Ground truth | Random Forest | CNN |
|---|---:|---:|---:|
| a standard deviation | 10.807 | 4.995 | 4.432 |
| b standard deviation | 14.608 | 6.933 | 6.392 |
| Mean chroma | 16.568 | 12.836 | 11.677 |
| Near-neutral pixels | 25.497% | 23.756% | 27.409% |
| Chroma above 40 | 5.868% | 0.00005% | 0% |

The metrics improved after correcting the input convention, but both baselines
still produce conservative chroma distributions. The U-Net experiment should
address the limited spatial context before changing the loss function.



