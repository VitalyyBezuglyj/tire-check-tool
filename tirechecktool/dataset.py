import shutil
from pathlib import Path

import pytorch_lightning as pl
from dvc.repo import Repo
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class TireCheckDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning data module for tire check classification.
    """

    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.batch_size = cfg.train.batch_size
        self.data_dir = Path(cfg.data.root_path)

        img_mean = cfg.data.preprocessing.mean
        img_std = cfg.data.preprocessing.std
        img_size = cfg.data.preprocessing.resize
        self.train_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(img_mean, img_std),
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(img_mean, img_std),
            ]
        )
        self._teardown = False
        self.cfg = cfg

    def prepare_data(self) -> None:
        """
        Downloads the dataset using dvc.api.
        """
        # check if data already exists
        if (self.data_dir).exists():
            return super().prepare_data()
        Repo.get(self.cfg.data.git_url, self.cfg.data.root_path)
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
