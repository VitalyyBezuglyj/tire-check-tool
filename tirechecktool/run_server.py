from logging import getLogger
from pathlib import Path

import mlflow

import onnx
import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf

from tirechecktool.utils import (
    preprocess_image,
    postprocess,
)


def infer_mlflow(cfg: OmegaConf, image_path: str):
    log = getLogger(__name__)
    log.setLevel(cfg.log_level.upper())

    model_name = cfg.export.export_name
    if cfg.export.name_version:
        model_name = f"{model_name}_{cfg.code_version.version}.onnx"
    else:
        model_name = f"{model_name}.onnx"

    filepath = Path(cfg.export.export_path) / model_name
    if not filepath.exists():
        raise ValueError(
            f"Model {model_name} does not exist at {filepath.parent.absolute()}"
        )

    onnx_model = onnx.load_model(filepath)

    # log the model into a mlflow run
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    input_sample = np.random.randn(*cfg.export.input_sample_shape)

    with mlflow.start_run():
        model_info = mlflow.onnx.log_model(
            onnx_model,
            model_name,
            input_example=input_sample,
        )

    # load the logged model and make a prediction
    onnx_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)
    predictions = onnx_pyfunc.predict(
        preprocess_image(image_path=image_path, cfg_data=cfg.data_module)
    )

    print(postprocess(predictions))


def run_server(
    image: str,
    config_path: str = "conf",
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
    print(OmegaConf.to_yaml(cfg))
    infer_mlflow(cfg, image)


if __name__ == "__main__":
    raise RuntimeError("Use `python commands.py infer`")
