from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Image configuration
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 160
TARGET_IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
PORTRAIT_ASPECT_RATIO = 4 / 5

# Dataset split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

# Training configuration
BATCH_SIZE = 16
INITIAL_LR = 0.001
INITIAL_EPOCHS = 5

# Image formats
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Dataset paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
COLOR_DIR = PROCESSED_DATA_DIR / "color"
GRAYSCALE_DIR = PROCESSED_DATA_DIR / "grayscale"
LAB_DIR = PROCESSED_DATA_DIR / "lab"
SPLITS_DIR = DATA_DIR / "splits"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
IDENTITY_ANNOTATIONS_PATH = ANNOTATIONS_DIR / "identity_CelebA.txt"

# Random Forest baseline configuration
RF_PIXELS_PER_IMAGE = 150
RF_N_ESTIMATORS = 50
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_LEAF = 2
RF_N_JOBS = -1
RF_PREDICTION_CHUNK_SIZE = 16384
RF_RANDOM_STATE = RANDOM_SEED
RF_CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "random_forest_colorizer.pkl"
RF_PREDICTIONS_DIR = BASE_DIR / "results" / "predictions"
RF_PLOTS_DIR = BASE_DIR / "results" / "plots"

# CNN baseline configuration
CNN_BATCH_SIZE = BATCH_SIZE
CNN_LEARNING_RATE = INITIAL_LR
CNN_EPOCHS = 20
CNN_EARLY_STOPPING_PATIENCE = 5
CNN_CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "cnn_colorizer.pt"


def ensure_project_directories() -> None:
    """Create the directory tree used by CHROMIFY if it is missing."""
    for directory in [
        RAW_DATA_DIR,
        COLOR_DIR,
        GRAYSCALE_DIR,
        LAB_DIR,
        SPLITS_DIR,
        ANNOTATIONS_DIR,
        BASE_DIR / "checkpoints",
        BASE_DIR / "results" / "predictions",
        BASE_DIR / "results" / "plots",
    ]:
        directory.mkdir(parents=True, exist_ok=True)
