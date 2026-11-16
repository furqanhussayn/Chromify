# CHROMIFY Project Context Report

This document summarizes the CHROMIFY repository, completed experiments,
diagnostics, GitHub preparation, and the next recommended work. It is intended
to provide context to another engineering or research assistant.

## 1. Project objective

CHROMIFY is a Python/PyTorch project for colorizing grayscale face portraits.
The pipeline uses:

- CelebA face images
- 128 x 160 pixel portrait outputs
- OpenCV CIELAB color space
- Grayscale/L input to predict LAB a/b chroma channels
- PyTorch for neural models
- Scikit-learn for the Random Forest baseline

Planned model families include Random Forest, a lightweight CNN, U-Net,
ResNet, an autoencoder, and MobileNetV2-based experiments.

## 2. Dataset and split status

The local dataset is intentionally excluded from GitHub.

- Original CelebA source:
  `C:\Users\icon\Downloads\img_align_celeba\img_align_celeba`
- Local selected dataset:
  `data/raw/`
- Identity annotation:
  `data/annotations/identity_CelebA.txt`
- Selected images: **5,000**
- Existing preserved images: **500**
- Newly selected images: **4,500**
- Selection seed: **20261114**
- Duplicate image hashes in selected set: **0**

Approved identity-disjoint split:

| Split | Images | Identities |
|---|---:|---:|
| Train | 4,000 | 2,799 |
| Validation | 500 | 500 |
| Test | 500 | 500 |

No identity appears in more than one split. The split files remain local and
must not be committed.

## 3. Preprocessing pipeline

The pipeline:

1. Reads raw JPG/JPEG/PNG files recursively.
2. Crops to a 4:5 portrait aspect ratio.
3. Resizes to exactly 128 x 160 using Pillow LANCZOS.
4. Saves lossless PNG-encoded color outputs.
5. Converts the resized RGB image to OpenCV LAB.
6. Saves lossless LAB outputs.
7. Saves the LAB L channel as the grayscale model input.

### LAB convention

OpenCV 8-bit LAB is used:

- L: `0..255`, normalized as `L / 255`
- a: centered with `a - 128`, normalized as `(a - 128) / 110`
- b: centered with `b - 128`, normalized as `(b - 128) / 110`

The helper functions are in:

- `src/preprocessing/image_utils.py`
- `src/preprocess.py`

## 4. Important preprocessing bug and fix

The original pipeline generated grayscale inputs with `RGB2GRAY`, but inference
treated those values as LAB L. This caused a mean grayscale/L difference of
approximately 5.7 intensity levels and approximately 6.67 MAE when reconstructing
RGB with grayscale incorrectly used as LAB L.

The fix:

- Added `lab_l_channel()` in `src/preprocessing/image_utils.py`.
- Changed preprocessing to derive grayscale from the same LAB conversion used
  for targets.
- Regenerated only `data/processed/grayscale/`.
- Preserved all raw, color, and LAB files.

Verification after the fix:

- 5,000 grayscale files checked.
- Exact grayscale/L matches: **5,000/5,000**
- Mismatches: **0**
- Maximum absolute difference: **0**
- Five-image direct-LAB vs extracted-L reconstruction MAE: **0.0**
- Normal 8-bit LAB round-trip MAE: approximately **0.39–0.52**

Existing model checkpoints were not silently treated as corrected-input models;
they were trained before this correction.

## 5. Dataset validation completed

The expanded dataset passed:

- 5,000 raw images
- 5,000 processed color images
- 5,000 processed grayscale images
- 5,000 processed LAB images
- Exact 128 x 160 dimensions
- No missing processed files
- No unreadable processed files
- No duplicate selected hashes
- Identity-disjoint splits
- DataLoader input shape: `[16, 1, 160, 128]`
- DataLoader target shape: `[16, 2, 160, 128]`

Validation utility:

```powershell
python src/utils/validate_expanded_dataset.py
```

## 6. Completed model experiments

### Original 500-image baselines

Random Forest test metrics:

- MAE: **0.05040**
- MSE: **0.006610**
- PSNR: **24.087 dB**
- SSIM: **0.89006**

CNN test metrics:

- MAE: **0.05246**
- MSE: **0.006571**
- PSNR: **23.497 dB**
- SSIM: **0.92160**

### 5,000-image baselines before LAB-L correction

Random Forest:

- Training samples: **600,000**
- Training duration: **562.78 seconds**
- Test MAE: **0.04711**
- Test MSE: **0.005290**
- Test PSNR: **24.403 dB**
- Test SSIM: **0.91926**

CNN:

- Architecture parameters: **38,114**
- Best epoch: **18**
- Best validation loss: **0.067465**
- Test MAE: **0.04582**
- Test MSE: **0.005162**
- Test PSNR: **24.625 dB**
- Test SSIM: **0.93134**

These checkpoints used the old RGB2GRAY-derived input and should not be used
as final corrected-input comparisons.

### Corrected LAB-L Random Forest

The corrected-input Random Forest completed successfully:

- Training samples: **600,000**
- Training duration: **403.62 seconds**
- Validation MAE: **0.04294**
- Validation MSE: **0.004927**
- Validation PSNR: **24.935 dB**
- Validation SSIM: **0.91981**
- Test MAE: **0.04289**
- Test MSE: **0.004981**
- Test PSNR: **24.946 dB**
- Test SSIM: **0.92237**

Checkpoint:

```text
checkpoints/random_forest_colorizer_5000_lab_l.pkl
```

Results:

```text
results/plots/5000_lab_l/random_forest_metrics.json
results/plots/5000_lab_l/random_forest_metrics.png
results/predictions/5000_lab_l/random_forest_test_comparisons.png
```

### Corrected LAB-L CNN

The corrected-input CNN resumed from epoch 3 with the original architecture,
Adam optimizer, L1 loss, batch size, CPU execution, and early stopping settings.
It completed resumed epochs 4–9 and stopped after three non-improving validation
epochs.

- Best epoch: **6**
- Best validation loss: **0.069960**
- Test MAE: **0.04281**
- Test MSE: **0.005006**
- Test PSNR: **24.962 dB**
- Test SSIM: **0.93518**

Checkpoint:

```text
checkpoints/cnn_colorizer_5000_lab_l.pt
```

The corrected CNN experiment is complete. Final evaluation, comparison images,
and chroma diagnostics are stored under the local `results/` directory.

## 7. Color-collapse diagnosis

Using the pre-correction 5,000-image test split:

| Statistic | Ground truth | Random Forest | CNN |
|---|---:|---:|---:|
| a standard deviation | 10.81 | 4.78 | 5.06 |
| b standard deviation | 14.61 | 6.90 | 7.10 |
| Mean chroma | 16.57 | 12.80 | 12.08 |
| Near-neutral pixels | 25.50% | 23.50% | 29.00% |

Both models collapsed toward conservative average chroma values. Ground-truth
test pixels with chroma above 40 were approximately 5.9%, while both models
produced effectively zero pixels in that range.

Likely causes:

1. The original L-channel mismatch.
2. L1 regression averaging when multiple colors are plausible for the same
   luminance.
3. Limited local/contextual information in the shallow CNN and 5 x 5
   Random Forest features.
4. A warm dataset average, approximately positive a and b, producing a
   brown/sepia bias.

The LAB normalization and reconstruction code itself passed round-trip tests;
the main confirmed pipeline issue was the grayscale/L mismatch.

## 8. Repository and GitHub status

The GitHub repository is:

```text
https://github.com/furqanhussayn/Chromify
```

Current branch:

```text
main
```

Latest pushed commit:

```text
cb0aebe docs: expose architecture-first directories
```

The repository is clean and local `main` matches `origin/main`.

GitHub excludes:

- `data/raw/`
- `data/processed/`
- `data/annotations/`
- `data/splits/`
- `data/expansion_preview/`
- `checkpoints/`
- `results/`
- Model files such as `.pt`, `.pkl`, `.onnx`, `.h5`, and `.weights`
- Images, logs, caches, virtual environments, and environment files

Tracked repository content includes source code, documentation, configuration,
requirements, and architecture directory README placeholders. No dataset,
checkpoint, generated result, or secret-like path is tracked.

## 9. Architecture-first collaboration structure

Visible source directories now include:

```text
src/
├── models/
│   ├── baselines/
│   ├── unet_furqan/
│   ├── resnet_kazi/
│   ├── autoencoder_yasir/
│   └── mobilenetv2_huzaifa/
└── training/
    ├── unet_furqan/
    ├── resnet_kazi/
    ├── autoencoder_yasir/
    └── mobilenetv2_huzaifa/
```

The directories contain tracked README placeholders so GitHub displays them.
Existing baseline modules were not moved unnecessarily.

Checkpoint and result architecture directories should remain local-only unless
the team later chooses to publish metadata-only documentation files.

## 10. Important tracked source files

- `config.py`: centralized paths and hyperparameters
- `src/preprocess.py`: raw-image preprocessing
- `src/preprocessing/image_utils.py`: RGB, LAB, normalization, and reconstruction
- `src/dataset.py`: identity splitting and PyTorch Dataset
- `src/models/random_forest.py`: Random Forest baseline
- `src/models/cnn.py`: lightweight CNN
- `src/training/train_random_forest.py`: RF training/evaluation workflow
- `src/training/train_cnn.py`: CNN training/evaluation workflow
- `src/utils/evaluation.py`: shared RGB metrics
- `src/utils/plan_dataset_expansion.py`: preview-only 5,000-image planner
- `src/utils/execute_dataset_expansion.py`: approved expansion executor
- `src/utils/validate_expanded_dataset.py`: expanded dataset validator
- `data/README.md`: local dataset setup and Git exclusion policy

## 11. Recommended next plan

### Immediate

1. Keep the corrected RF and CNN artifacts as the locked baseline checkpoint.
2. Review the corrected RF/CNN comparison table and chroma diagnostics.
3. Begin the lightweight U-Net experiment only after this baseline checkpoint.

### Before any new architecture

1. Confirm corrected-input RGB metrics are measured with the same evaluation
   methodology.
2. Compare a/b standard deviations, mean chroma, high-chroma pixel frequency,
   and near-neutral percentage.
3. Inspect corrected-input visual outputs.
4. Keep README baseline metrics synchronized with finalized experiments.

### Next modeling experiment

If corrected-input results remain desaturated, implement a lightweight U-Net
with larger spatial context. Keep it in:

```text
src/models/unet_furqan/
src/training/unet_furqan/
```

Use separate local-only artifacts under matching checkpoint/result directories.
Do not overwrite Random Forest or CNN checkpoints.

### Longer-term improvements

- Use larger spatial context and skip connections.
- Consider color-aware or distribution-aware losses after establishing a
  corrected-input baseline.
- Increase dataset diversity if available.
- Keep all model comparisons on the same identity-disjoint test set.
- Never tune architecture or loss using the test split.

## 12. Safety rules for future agents

- Do not commit or push CelebA images, annotations, split files, manifests,
  checkpoints, prediction images, or generated datasets.
- Do not use `git add .`; stage explicitly reviewed files.
- Run `git status --short`, `git ls-files`, `git diff --cached --name-only`,
  and forbidden-path checks before committing.
- Do not force-push or rewrite remote history.
- Do not regenerate identity splits without explicit approval.
- Do not claim a model experiment is complete without a saved checkpoint,
  validation metrics, test metrics, and artifact verification.
