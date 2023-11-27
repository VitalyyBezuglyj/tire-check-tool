from omegaconf import OmegaConf
import subprocess
import numpy as np
from PIL import Image
import mlflow
from os import environ


def set_run_id():
    environ["MLFLOW_RUN_ID"] = str(mlflow.active_run().run_id)


def get_run_id():
    if environ.get("Foo") is not None:
        return environ["MLFLOW_RUN_ID"]
    else:
        return None


def log_git_info(cfg: OmegaConf):
    """
    Log git info to the config.

    Args:
        cfg: hydra config
    """
    cfg.code_version.git_commit_id = subprocess.check_output(
        ["git", "describe", "--always"]
    ).strip()


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
            image_data[channel, :, :] = (
                image_data[channel, :, :] / 255 - mean[channel]
            ) / std[channel]
        image_data = np.expand_dims(image_data, 0)
    return image_data


def postprocess(output: dict):
    probs = [float(x) * 100 for x in softmax(output["CLASS_PROBS"])[0]]
    result = "Defective: {0:.2f}% Good {1:.2f}%".format(*probs)

    return result


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# def load_image(img_path: str, size: list) -> np.ndarray:
#     img_path = Path(img_path)

#     with Image.open(img_path) as im:
#         img = np.asarray(im.convert("RGB").resize(tuple(size))) / 255.0
#         img.
#         ic(img)
#         return img
