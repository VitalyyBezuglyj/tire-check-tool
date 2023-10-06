# Data
DATASET_URL = "https://www.kaggle.com/datasets/warcoder/tyre-quality-classification"  # noqa: E501
DATA_PATH = "./data"
TRAIN_SPLIT = 0.8
NUM_CLASSES = 2

# Preprocessing

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# Training

SAVE_METRICS = True
METRICS_PATH = "./outputs/metrics"
SAVE_CKPT = True
CKPT_PATH = "./outputs/checkpoints"

# Kaggle
KAGGLE_CREDENTIALS = {
    "username": "creatorofuniverses",
    "key": "ece6171c341d90d57960f49bcf10af93",
}
