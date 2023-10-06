import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .constants import DATA_PATH


def get_dataloader(
    type: str = "train",
    shuffle: bool = True,
    batch_size: int = 32,
) -> DataLoader:
    """
    Returns a dataloader for the dataset
    """

    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.Resize((150, 150)),
                transforms.ToTensor(),
                transforms.Normalize(img_mean, img_std),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((150, 150)),
                transforms.ToTensor(),
                transforms.Normalize(img_mean, img_std),
            ]
        ),
    }

    data_dir = DATA_PATH + "/tyre-quality-clf-splitted"
    image_dataset = datasets.ImageFolder(
        os.path.join(data_dir, type), data_transforms[type]
    )

    dataloader = DataLoader(image_dataset, batch_size, shuffle)

    return dataloader, image_dataset.classes
