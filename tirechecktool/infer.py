from logging import getLogger
from pathlib import Path

import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .model import TireCheckModel


def run_inferring(cfg: OmegaConf):
    log = getLogger(__name__)
    log.setLevel(cfg.log_level.upper())
    pl.seed_everything(cfg.random_seed)

    ckpt_path = Path(cfg.callbacks.model_ckpt.dirpath)
    best_model_names = list(ckpt_path.glob("best_*.ckpt"))
    best_checkpoint_name = best_model_names[0]

    model = TireCheckModel.load_from_checkpoint(best_checkpoint_name)

    dm = instantiate(cfg.data_module)
    dm.setup(stage="predict")

    if not torch.cuda.is_available():
        log.warning("CUDA is not available, using CPU")
        cfg.trainer.accelerator = "cpu"

    trainer = instantiate(cfg.inferer)

    trainer.test(model, dm)


def infer(config_path: str = "conf", config_name: str = "default", **kwargs):
    """
    Run inference. `infer -- --help` for more info.

    Args:
        :param config_path: path to config dir, relative to project root or absolute
        :param config_name: name of config file, inside config_path
        :param **kwargs: additional arguments, overrides for config. Can be passed as `--arg=value`.
    """
    initialize(version_base="1.3", config_path=config_path, job_name="tirechecktool-train")
    cfg = compose(config_name=config_name, overrides=[f"{k}={v}" for k, v in kwargs.items()])
    print(OmegaConf.to_yaml(cfg))
    run_inferring(cfg)


if __name__ == "__main__":
    raise RuntimeError("Use `python commands.py infer`")
