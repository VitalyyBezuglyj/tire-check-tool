import shutil
from pathlib import Path

import pytorch_lightning as pl
from dvc.repo import Repo
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tirechecktool.utils import get_git_info


class TireCheckDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning data module for tire check classification.
    """

    def __init__(
        self,
        batch_size: int = 32,
        data_dir: Path = Path("data"),
        git_url: str = "",
        img_size: int = 150,
        img_mean: tuple = (0.5, 0.5, 0.5),
        img_std: tuple = (0.5, 0.5, 0.5),
        random_rotation: int = 10,
        teardown: bool = False,
        train_split: float = 0.8,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = Path(data_dir)
        resize = (img_size, img_size)

        self.train_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(random_rotation),
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(img_mean, img_std),
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(img_mean, img_std),
            ]
        )
        self._teardown = teardown
        self.git_url = git_url
        self.train_split = train_split

    def prepare_data(self) -> None:
        """
        Downloads the dataset using dvc.api.
        """
        # check if data already exists
        if (self.data_dir).exists():
            return super().prepare_data()
        Repo.get(
            url=str(self.git_url),
            path=str(self.data_dir),
            rev=get_git_info(),
        )
        return super().prepare_data()

    def setup(self, stage="fit"):
        self.prepare_data()

        if stage == "fit":
            # split dataset
            tires_dataset = datasets.ImageFolder(
                self.data_dir / "tyre-quality-classification",
                transform=self.train_transforms,
            )
            train_size = int(self.train_split * len(tires_dataset))
            val_size = len(tires_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                tires_dataset, [train_size, val_size]
            )
        elif stage == "test":
            self.test_dataset = datasets.ImageFolder(
                self.data_dir / "tyre-quality-classification",
                transform=self.val_transforms,
            )

        elif stage == "predict":
            self.predict_dataset = datasets.ImageFolder(
                self.data_dir / "tyre-quality-classification",
                transform=self.val_transforms,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def teardown(self, stage: str) -> None:
        """
        Removes the downloaded dataset.
        """
        if self._teardown:
            shutil.rmtree(self.data_dir / "tyre-quality-classification")
        return super().teardown(stage)
