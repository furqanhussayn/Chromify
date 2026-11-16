# Local Dataset Setup

CHROMIFY does not include CelebA or any other dataset in the GitHub repository.
Each collaborator must obtain the approved dataset separately and keep it on
their local machine.

## Expected local structure

```text
data/
├── raw/                         # Selected CelebA image files
├── processed/
│   ├── color/
│   ├── grayscale/
│   └── lab/
├── annotations/
│   └── identity_CelebA.txt
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

The current experiment uses 5,000 selected CelebA images at 128 x 160 pixels.
The split files are identity-disjoint and must be obtained through the approved
team workflow rather than regenerated independently.

## Dataset consistency

All team members must use the same CelebA version, preprocessing convention,
selection manifest, and identity-aware split assignments. Do not commit images,
annotations, split files, manifests containing dataset records, or generated
datasets.

## Dataset loading smoke test

After placing the local data and installing dependencies, run from the project
root:

```powershell
python src/utils/validate_expanded_dataset.py
```

The smoke test checks raw and processed file completeness, image dimensions,
identity-disjoint split counts, and a DataLoader batch. It should report input
shape `[16, 1, 160, 128]` and target shape `[16, 2, 160, 128]`.
