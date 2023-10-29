from pathlib import Path

import pytorch_lightning as pl
import torch

from .constants import CKPT_PATH
from .dataset import TireCheckDataModule
from .model import TireCheckModel


def infer():
    pl.seed_everything(1244)

    ckpt_path = Path(CKPT_PATH)
    best_model_name = ckpt_path / "best.txt"

    with open(best_model_name, "r") as f:
        best_checkpoint_name = f.readline()

    model = TireCheckModel.load_from_checkpoint(best_checkpoint_name)

    dm = TireCheckDataModule()
    dm.setup(stage="predict")

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        accelerator=accelerator,
    )

    trainer.test(model, dm)


if __name__ == "__main__":
    infer()
