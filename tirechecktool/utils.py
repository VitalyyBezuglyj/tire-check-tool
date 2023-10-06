import json
import os
import random
import shutil

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

    od.download(
        DATASET_URL,
        DATA_PATH,
    )

    os.remove("kaggle.json")

    split_dataset(
        base_dir=DATA_PATH + "/tyre-quality-classification",
        output_dir=DATA_PATH + "/tyre-quality-clf-splitted",
        split_ratio=TRAIN_SPLIT,
    )


def split_dataset(
    base_dir, output_dir, split_ratio=0.2, remove_after_split=False
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

    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} does not exist.")
        return

    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Iterate through classes
    for class_name in tqdm(
        os.listdir(base_dir), desc="Iterating through classes"
    ):
        class_dir = os.path.join(base_dir, class_name)

        # Check if it's a directory
        if not os.path.isdir(class_dir):
            continue

        # Create output directories for the class
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        # List all images and shuffle
        images = os.listdir(class_dir)
        random.shuffle(images)

        # Split the images
        split_idx = int(len(images) * (1 - split_ratio))
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Copy images to respective sets
        for image in tqdm(train_images, desc="Train"):
            shutil.copy(
                os.path.join(class_dir, image),
                os.path.join(train_dir, class_name, image),
            )
        for image in tqdm(val_images, desc="Val"):
            shutil.copy(
                os.path.join(class_dir, image),
                os.path.join(val_dir, class_name, image),
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
