import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import opendatasets as od
import torch
from tqdm import tqdm

from .constants import DATA_PATH, DATASET_URL, KAGGLE_CREDENTIALS, TRAIN_SPLIT


def download_dataset() -> None:
    """
    Downloads the dataset from Kaggle using opendatasets
    """

    with open("kaggle.json", "w") as file:
        json.dump(KAGGLE_CREDENTIALS, file)

    data_path = Path(DATA_PATH)

    od.download(
        DATASET_URL,
        data_path,
    )

    os.remove("kaggle.json")

    split_dataset(
        base_dir=data_path / "tyre-quality-classification",
        output_dir=data_path / "tyre-quality-clf-splitted",
        split_ratio=TRAIN_SPLIT,
    )


def split_dataset(
    base_dir: Path, output_dir: Path, split_ratio: float = 0.2, remove_after_split: bool = False
):
    """
    Splits a dataset into training and validation sets.

    Parameters:
    - base_dir: Directory where the original dataset resides
    - output_dir: Directory where the train and val directories
                  should be created
    - split_ratio: Ratio of validation set.
                   Default is 0.2 (i.e., 80% training, 20% validation)
    """

    print("Splitting dataset...")

    print(f"Curr dir: {Path.cwd()}")
    base_dir = Path(base_dir)
    print(f"Base directory: {base_dir}")
    output_dir = Path(output_dir)
    print(f"Output directory: {output_dir}")

    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} does not exist.")
        return

    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # List classes using pathlib
    list_classes = list(Path(base_dir).glob("*"))
    print(f"List of classes: {list_classes}")

    # Iterate through classes
    for class_dir in tqdm(base_dir.glob("*"), desc="Iterating through classes"):
        # Check if it's a directory
        if not os.path.isdir(class_dir):
            continue

        # Get class name
        class_name = class_dir.stem

        # Check if it's a directory
        if not os.path.isdir(class_dir):
            continue

        # Create output directories for the class
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)

        # List all images and shuffle
        images = list(class_dir.glob("*"))
        print(f"Found {len(images)} images for class {class_name}")
        random.shuffle(images)

        # Split the images
        split_idx = int(len(images) * (1 - split_ratio))
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Copy images to respective sets
        for image in tqdm(train_images, desc="Train"):
            shutil.copy(
                image,
                train_dir / class_name / image.name,
            )
        for image in tqdm(val_images, desc="Val"):
            shutil.copy(
                image,
                val_dir / class_name / image.name,
            )

        # Remove original directory if specified
        if remove_after_split:
            shutil.rmtree(base_dir)


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
