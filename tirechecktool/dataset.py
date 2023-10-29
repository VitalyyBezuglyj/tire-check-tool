import shutil
from pathlib import Path

import pytorch_lightning as pl
from dvc.repo import Repo
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .constants import DATA_PATH


class TireCheckDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning data module for tire check classification.
    """

    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = Path(DATA_PATH)

        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        self.train_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.Resize((150, 150)),
                transforms.ToTensor(),
                transforms.Normalize(img_mean, img_std),
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                transforms.Resize((150, 150)),
                transforms.ToTensor(),
                transforms.Normalize(img_mean, img_std),
            ]
        )
        self._teardown = False

    def prepare_data(self) -> None:
        """
        Downloads the dataset using dvc.api.
        """
        # check if data already exists
        if (self.data_dir).exists():
            return super().prepare_data()
        Repo.get("git@github.com:VitalyyBezuglyj/tire-check-tool.git", "data")
        return super().prepare_data()

    def setup(self, stage="fit"):
        self.prepare_data()

        if stage == "fit":
            # split dataset
            tires_dataset = datasets.ImageFolder(
                self.data_dir / "tyre-quality-classification", transform=self.train_transforms
            )
            train_size = int(0.8 * len(tires_dataset))
            val_size = len(tires_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(tires_dataset, [train_size, val_size])
        elif stage == "test":
            self.test_dataset = datasets.ImageFolder(
                self.data_dir / "tyre-quality-classification", transform=self.val_transforms
            )

        elif stage == "predict":
            self.predict_dataset = datasets.ImageFolder(
                self.data_dir / "tyre-quality-classification", transform=self.val_transforms
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def teardown(self, stage: str) -> None:
        """
        Removes the downloaded dataset.
        """
        if self._teardown:
            shutil.rmtree(self.data_dir / "tyre-quality-classification")
        return super().teardown(stage)
