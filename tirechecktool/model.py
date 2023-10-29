from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


class SimpleCNN(nn.Module):
    """
    Simple CNN model for image classification.
    """

    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 18 * 18, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 18 * 18)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TireCheckModel(pl.LightningModule):
    """
    Pytorch Lightning model for tire check classification.
    """

    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg
        self.model = SimpleCNN(cfg.model.n_classes)
        self.learning_rate = cfg.train.learning_rate
        self.loss = nn.CrossEntropyLoss()
        metrics = MetricCollection(
            [
                MulticlassAccuracy(num_classes=cfg.model.n_classes),
                MulticlassPrecision(num_classes=cfg.model.n_classes),
                MulticlassRecall(num_classes=cfg.model.n_classes),
                MulticlassF1Score(num_classes=cfg.model.n_classes),
            ]
        )

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.train_metrics.update(logits, y)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        train_metrics = self.train_metrics.compute()
        self.log_dict(train_metrics, prog_bar=True)
        self.train_metrics.reset()
        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        self.val_metrics.update(logits, y)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return {"val_loss": loss}

    def on_validation_epoch_end(self) -> None:
        val_metrics = self.val_metrics.compute()
        self.log_dict(val_metrics, prog_bar=True)
        self.val_metrics.reset()
        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.test_metrics.update(logits, y)

    def on_test_epoch_end(self) -> None:
        test_metrics = self.test_metrics.compute()

        test_metrics = {k: [v.item()] for k, v in test_metrics.items()}
        print(test_metrics)

        if self.cfg.artifacts.metrics.save:
            metric_filename = Path(self.cfg.artifacts.base_path) / "metrics.csv"

            pd.DataFrame(test_metrics).to_csv(metric_filename, index=False)
            self.test_metrics.reset()
            return super().on_test_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        return {"optimizer": optimizer}

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2))
        super().on_before_optimizer_step(optimizer)
