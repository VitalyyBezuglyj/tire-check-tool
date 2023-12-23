from logging import getLogger
from pathlib import Path

import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from tirechecktool.utils import log_git_info


def run_training(cfg: OmegaConf):
    log = getLogger(__name__)
    log.setLevel(cfg.log_level.upper())

    log_git_info(cfg)

    pl.seed_everything(cfg.random_seed)
    torch.set_float32_matmul_precision(cfg.float_precision)

    dm = instantiate(cfg.data_module)
    model = instantiate(cfg.tire_check_model)

    loggers = []
    if cfg.save_logs:
        for _logger in cfg.loggers.log_to:
            new_logger = instantiate(cfg.loggers[_logger])
            new_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
            loggers.append(new_logger)

    callbacks = [
        instantiate(cfg.callbacks.lr_monitor),
        instantiate(cfg.callbacks.device_stats),
        instantiate(cfg.callbacks.model_summary),
    ]
    if cfg.save_ckpt:
        checkpoint_callback = instantiate(cfg.callbacks.model_ckpt)

        callbacks.append(checkpoint_callback)

    if not torch.cuda.is_available():
        log.warning("CUDA is not available, using CPU")
        cfg.trainer.accelerator = "cpu"

    trainer = instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    trainer.fit(model, dm)


@logger.catch
def train(config_name: str = "default", config_path: str = "../configs", **kwargs):
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
    run_training(cfg)


if __name__ == "__main__":
    raise RuntimeError("Use `python commands.py train`")
