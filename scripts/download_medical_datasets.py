import os
from pathlib import Path

import kaggle
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NIH_DATASET_DIR = PROJECT_ROOT / "datasets" / "chest_xray" / "nih"


def download_nih_chestxray():
    """Download NIH ChestX-ray14 dataset."""
    print("Downloading NIH ChestX-ray14...")
    NIH_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    # Kaggle dataset: nih-chest-xrays/data
    kaggle.api.dataset_download_files(
        "nih-chest-xrays/data",
        path=str(NIH_DATASET_DIR),
        unzip=True,
    )


def download_chexpert():
    """Download Stanford CheXpert dataset."""
    print("Downloading CheXpert...")
    # Requires login at https://aimi.stanford.edu/chexpert-chest-x-rays
    print("Please download manually from CheXpert website")


def prepare_training_data():
    """Prepare data for training."""
    # Create training/validation/test splits
    pass


if __name__ == "__main__":
    download_nih_chestxray()
    prepare_training_data()
