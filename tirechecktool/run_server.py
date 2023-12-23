from functools import lru_cache
from logging import getLogger

import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput

from tirechecktool.model import TireCheckModel
from tirechecktool.utils import get_model_path, postprocess, preprocess_image


@lru_cache(maxsize=1)
def get_triton_client(cfg: OmegaConf):
    return InferenceServerClient(url=cfg.triton.server_url)


def infer_triton(cfg: OmegaConf, image_path: str):
    log = getLogger(__name__)
    log.setLevel(cfg.log_level.upper())

    triton_client = get_triton_client(cfg)
    inputs = []
    outputs = []
    input_data = preprocess_image(image_path=image_path, cfg_data=cfg.data_module)
    inputs.append(InferInput("IMAGES", input_data.shape, "FP32"))
    inputs[-1].set_data_from_numpy(input_data, binary_data=False)

    # Create the data for the two output tensors
    outputs.append(InferRequestedOutput("CLASS_PROBS", binary_data=True))

    # Infer
    results = triton_client.infer(cfg.triton.model_name, inputs, outputs=outputs)

    # Print the results
    print("Results:")
    for output in results.as_numpy("CLASS_PROBS"):
        print(postprocess(output))


def test_triton_inference(cfg: OmegaConf):
    """Test Triton inference

    Compare triton server inference results with torch inference results

    """
    dm = instantiate(cfg.data_module)
    dm.setup(stage="predict")
    dataset = dm.predict_dataset
    imgs = []
    for i in range(cfg.test_triton.test_num):
        imgs.append(dataset[i][0])
    imgs = torch.stack(imgs, dim=0)
    imgs = imgs.to("cuda")

    ckpt_path = get_model_path(cfg)
    model = TireCheckModel.load_from_checkpoint(ckpt_path).to("cuda")
    model.eval()
    model.freeze()

    with torch.no_grad():
        torch_output = model(imgs).cpu().numpy()

    triton_client = get_triton_client(cfg)
    inputs = []
    outputs = []
    input_data = imgs.cpu().numpy()
    inputs.append(InferInput("IMAGES", input_data.shape, "FP32"))
    inputs[-1].set_data_from_numpy(input_data, binary_data=False)

    # Create the data for the two output tensors
    outputs.append(InferRequestedOutput("CLASS_PROBS", binary_data=True))
    results = triton_client.infer(cfg.triton.model_name, inputs, outputs=outputs)
    triton_output = results.as_numpy("CLASS_PROBS")

    assert np.allclose(torch_output, triton_output, atol=cfg.test_triton.tolerance)
    logger.info("Triton inference test passed with tolerance thresh: {cfg.test_triton.tolerance}")


@logger.catch
def run_server(
    image: str,
    config_path: str = "../configs",
    config_name: str = "default",
    **kwargs,
):
    """
    Run inference. `infer -- --help` for more info.

    Args:
        :param image: Path to image to predict
        :param config_path: path to config dir, relative to project root or absolute
        :param config_name: name of config file, inside config_path
        :param **kwargs: additional arguments, overrides for config. Can be passed as `--arg=value`.
    """
    initialize(
        version_base="1.3",
        config_path=config_path,
        job_name="tirechecktool-train",
    )
    cfg = compose(
        config_name=config_name,
        overrides=[f"{k}={v}" for k, v in kwargs.items()],
    )
    infer_triton(cfg, image)


@logger.catch
def test_server(
    config_path: str = "../configs",
    config_name: str = "default",
    **kwargs,
):
    """
    Run inference. `infer -- --help` for more info.

    Args:
        :param image: Path to image to predict
        :param config_path: path to config dir, relative to project root or absolute
        :param config_name: name of config file, inside config_path
        :param **kwargs: additional arguments, overrides for config. Can be passed as `--arg=value`.
    """
    initialize(
        version_base="1.3",
        config_path=config_path,
        job_name="tirechecktool-train",
    )
    cfg = compose(
        config_name=config_name,
        overrides=[f"{k}={v}" for k, v in kwargs.items()],
    )
    test_triton_inference(cfg)


if __name__ == "__main__":
    raise RuntimeError("Use `python commands.py infer`")
