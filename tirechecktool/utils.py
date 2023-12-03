import subprocess
from pathlib import Path

import cv2
import numpy as np
from dvc.repo import Repo
from omegaconf import OmegaConf
from PIL import Image


def get_git_info():
    """
    Get git info.

    Returns:
        str: git commit id
    """
    return subprocess.check_output(["git", "describe", "--always"]).strip().decode()


def log_git_info(cfg: OmegaConf):
    """
    Log git info to the config.

    Args:
        cfg: hydra config
    """
    cfg.code_version.git_commit_id = get_git_info()
    return cfg.code_version.git_commit_id


def preprocess_image(image_path, cfg_data: OmegaConf):
    width = height = cfg_data.img_size
    channel = len(cfg_data.img_std)
    with Image.open(image_path) as image:
        #  image = Image.open(image_path)
        image = image.resize((width, height), Image.LANCZOS)
        image_data = np.asarray(image).astype(np.float32)
        image_data = image_data.transpose([2, 0, 1])  # transpose to CHW
        mean = np.array(cfg_data.img_mean)
        std = np.array(cfg_data.img_std)
        for channel in range(image_data.shape[0]):
            image_data[channel, :, :] = (image_data[channel, :, :] / 255 - mean[channel]) / std[channel]
        image_data = np.expand_dims(image_data, 0)
    return image_data


def preprocess_image_binary(image_path, cfg_data: OmegaConf):
    width = height = cfg_data.img_size

    raw_image = cv2.imread(image_path)
    blob = cv2.dnn.blobFromImage(raw_image, 1.0, (width, height), cfg_data.img_mean, True, False)
    return blob


def postprocess(output: np.ndarray):
    probs = [float(x) * 100 for x in softmax(output)]
    result = "Good: {0:.2f}% Defective {1:.2f}%".format(*probs)

    return result


def sigmoid(x):
    """Compute sigmoid values for each sets of scores in x."""
    return 1 / (1 + np.exp(-x))


def log_softmax(x):
    """Compute log softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum() + 1e-6)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def load_pretained_models(cfg: OmegaConf):
    """
    Downloads the pretrained model using dvc.api.
    """
    # check if data already exists
    data_path = Path(cfg.pretrained_dir)
    if data_path.exists():
        print("Pretrained models already loaded")
        return

    log_git_info(cfg)
    Repo.get(
        cfg.data_module.git_url,
        cfg.pretrained_dir,
        rev=cfg.code_version.git_commit_id,
    )
    print("Pretrained models loaded")


def get_model_path(cfg: OmegaConf):
    """
    Choose the pretrained model and return path.
    """

    if cfg.use_pretrained:
        load_pretained_models(cfg)
        # check if data already exists
        data_path = Path(cfg.pretrained_dir)
        if len(cfg.pretrained_model) > 0:
            model_path = data_path / cfg.pretrained_model
            if model_path.exists():
                return model_path
            else:
                raise ValueError(f"Model {cfg.pretrained_model} does not exist at {model_path.absolute()}")
        else:
            pretrained_model_names = sorted(list(data_path.glob("*.ckpt")), reverse=True)
            if len(pretrained_model_names) == 0:
                raise ValueError(f"No pretrained model found at {data_path.absolute()}")
            return pretrained_model_names[0]
    else:
        # Choose best ckpt from training
        ckpt_path = Path(cfg.callbacks.model_ckpt.dirpath)
        best_model_names = sorted(list(ckpt_path.glob("best_*.ckpt")), reverse=True)
        if len(best_model_names) == 0:
            raise ValueError(
                f"No best model found at {ckpt_path.absolute()}, please train model first,\
                    or use pretrained model (cfg: use_pretrained)"
            )
        return best_model_names[0]
