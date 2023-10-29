from pathlib import Path

import pytorch_lightning as pl
import torch

from .constants import CKPT_PATH, SAVE_CKPT
from .dataset import TireCheckDataModule
from .model import TireCheckModel


def train():
    pl.seed_everything(1244)
    torch.set_float32_matmul_precision("medium")

    dm = TireCheckDataModule(batch_size=32)
    model = TireCheckModel()

    loggers = [
        pl.loggers.CSVLogger("logs", name="default"),
    ]
    ckpt_path = Path(CKPT_PATH)

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(max_depth=1),
    ]
    if SAVE_CKPT:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_path,
            monitor="val_loss",
            filename="{epoch}-{val_loss:.2f}-{other_metric:.2f}",
            save_top_k=1,
            mode="min",
            save_last=True,
        )

        callbacks.append(checkpoint_callback)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        accelerator=accelerator,
        log_every_n_steps=10,
        precision="16-mixed",
        max_epochs=5,
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(model, dm)

    if SAVE_CKPT:
        print(checkpoint_callback.best_model_path)

        best_model_name = Path(ckpt_path / "best.txt")

        with open(best_model_name, "w") as f:
            f.write(checkpoint_callback.best_model_path)


if __name__ == "__main__":
    train()
