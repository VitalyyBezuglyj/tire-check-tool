from pathlib import Path

import pandas as pd
import torch
from torch import nn, optim
from tqdm import tqdm

from .constants import (
    CKPT_PATH,
    DATA_PATH,
    METRICS_PATH,
    SAVE_CKPT,
    SAVE_METRICS,
    TRAIN_SPLIT,
)
from .dataset import get_dataloader
from .model import SimpleCNN
from .utils import download_dataset, set_seed, split_dataset


def train():
    set_seed(1244)

    # Create dataset directory
    # Path(DATA_PATH).mkdir(parents=True, exist_ok=True)

    # Check if dataset is present and download if not present
    if not Path(DATA_PATH).exists():
        download_dataset()

    if not Path(DATA_PATH + "/tyre-quality-clf-splitted").exists():
        split_dataset(
            base_dir=DATA_PATH + "/tyre-quality-classification",
            output_dir=DATA_PATH + "/tyre-quality-clf-splitted",
            split_ratio=TRAIN_SPLIT,
        )
    # Dataloader
    dataloaders = {}
    dataloaders["train"], classes = get_dataloader(
        type="train", shuffle=True, batch_size=32
    )
    dataloaders["val"], _ = get_dataloader(
        type="val", shuffle=False, batch_size=32
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop with saving checkpoints
    num_epochs = 5
    best_acc = 0.0

    # Create checlpoints directory
    if SAVE_CKPT:
        Path(CKPT_PATH).mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(num_epochs), desc="Training"):
        metrics = {
            "train_loss": 0.0,
            "train_acc": 0.0,
            "val_loss": 0.0,
            "val_acc": 0.0,
        }

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_accuracy = correct / total
            metrics[f"{phase}_loss"] = epoch_loss
            metrics[f"{phase}_acc"] = epoch_accuracy

            print(
                f"{phase.capitalize()} \
                    Loss: {epoch_loss:.4f} \
                    Accuracy: {epoch_accuracy:.4f}"
            )

            # Save the best model checkpoint based on validation accuracy
            if SAVE_CKPT and phase == "val" and epoch_accuracy > best_acc:
                best_acc = epoch_accuracy
                save_path = f"{CKPT_PATH}/best_model_checkpoint.pth"
                torch.save(model.state_dict(), save_path)

        # Save metrics to CSV
        if SAVE_METRICS:
            Path(METRICS_PATH).mkdir(parents=True, exist_ok=True)
            df_metrics = pd.DataFrame(metrics, index=[epoch])
            if epoch == 0:
                df_metrics.to_csv(
                    f"{METRICS_PATH}/training_metrics.csv",
                    mode="w",
                    header=True,
                )
            else:
                df_metrics.to_csv(
                    f"{METRICS_PATH}/training_metrics.csv",
                    mode="a",
                    header=False,
                )

    print("Training complete!")


if __name__ == "__main__":
    train()
