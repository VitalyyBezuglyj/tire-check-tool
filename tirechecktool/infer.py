import os
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from .constants import CKPT_PATH, DATA_PATH, METRICS_PATH, NUM_CLASSES
from .dataset import get_dataloader
from .model import SimpleCNN
from .utils import download_dataset, set_seed


def infer():
    set_seed(1244)
    # Create dataset directory
    Path(DATA_PATH).mkdir(parents=True, exist_ok=True)

    # Check if dataset is present and download if not present
    if not Path(DATA_PATH).exists():
        download_dataset()

    # Dataloader
    batch_size = 32
    shuffle = True
    d_type = "val"
    dataloader, class_names = get_dataloader(d_type, shuffle, batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)

    # Load the best model for evaluation
    if os.path.exists(f"{CKPT_PATH}/best_model_checkpoint.pth"):
        model_path = f"{CKPT_PATH}/best_model_checkpoint.pth"
        model.load_state_dict(torch.load(model_path))
    else:
        raise Exception(
            "Error: No model checkpoint found. Please train the model first."
        )

    # Calculate classification metrics on validation set
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Map numeric labels to class names
    mapped_gt_labels = [class_names[label] for label in all_labels]
    mapped_pred_labels = [class_names[label] for label in all_predictions]

    # Save predictions, ground truth, and their mappings to CSV
    data_to_save = {
        "Ground_Truth_Numeric": all_labels,
        "Ground_Truth_Label": mapped_gt_labels,
        "Prediction_Numeric": all_predictions,
        "Prediction_Label": mapped_pred_labels,
    }
    df = pd.DataFrame(data_to_save)
    df.to_csv(
        f"{METRICS_PATH}/predictions_ground_truth_mapped.csv", index=False
    )

    # Calculate and print classification report and confusion matrix
    class_names = class_names
    print("Classification Report:")
    print(
        classification_report(
            all_labels, all_predictions, target_names=class_names
        )
    )
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions))


if __name__ == "__main__":
    infer()
