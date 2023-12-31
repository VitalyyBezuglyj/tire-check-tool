from logging import getLogger
from pathlib import Path

import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from loguru import logger
from omegaconf import OmegaConf

from tirechecktool.model import TireCheckModel
from tirechecktool.utils import get_model_path


def export_to_onnx(cfg: OmegaConf):
    log = getLogger(__name__)
    log.setLevel(cfg.log_level.upper())
    pl.seed_everything(cfg.random_seed)

    ckpt_path = get_model_path(cfg)

    model = TireCheckModel.load_from_checkpoint(ckpt_path)

    if cfg.export.export_triton:
        model_name = "model.onnx"
        filepath = Path(cfg.triton.models_path) / model_name
    else:
        model_name = cfg.export.export_name
        if cfg.export.name_version:
            model_name = f"{model_name}_{cfg.code_version.version}.onnx"
        else:
            model_name = f"{model_name}.onnx"
        filepath = Path(cfg.export.export_path) / model_name
        filepath.parent.mkdir(parents=True, exist_ok=True)

    input_sample = torch.randn(tuple(cfg.export.input_sample_shape))
    model.to_onnx(
        filepath,
        input_sample,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["IMAGES"],
        output_names=["CLASS_PROBS"],
        dynamic_axes={
            "IMAGES": {0: "BATCH_SIZE"},
            "CLASS_PROBS": {0: "BATCH_SIZE"},
        },
    )

    log.info(f"Exported to {filepath}")


@logger.catch
def export_onnx(config_name: str = "default", config_path: str = "../configs", **kwargs):
    """
    Run training. `train -- --help` for more info.

    Args:
    :param config_path: path to config dir, relative to project root or absolute
    :param config_name: name of config file, inside config_path
    :param **kwargs: additional arguments, overrides for config. Can be passed as `--arg=value`.
    """
    logger.info(f"Curr path: {Path.cwd()}")
    initialize(
        version_base="1.3",
        config_path=config_path,
        job_name="tirechecktool-train",
    )
    cfg = compose(
        config_name=config_name,
        overrides=[f"{k}={v}" for k, v in kwargs.items()],
    )
    logger.info(OmegaConf.to_yaml(cfg))
    export_to_onnx(cfg)


if __name__ == "__main__":
    raise RuntimeError("Use `python commands.py train`")
