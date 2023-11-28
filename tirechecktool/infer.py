from logging import getLogger

import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tirechecktool.model import TireCheckModel
from tirechecktool.utils import get_model_path


def run_inferring(cfg: OmegaConf):
    log = getLogger(__name__)
    log.setLevel(cfg.log_level.upper())
    pl.seed_everything(cfg.random_seed)

    ckpt_path = get_model_path(cfg)

    model = TireCheckModel.load_from_checkpoint(ckpt_path)

    dm = instantiate(cfg.data_module)
    dm.setup(stage="predict")

    if not torch.cuda.is_available():
        log.warning("CUDA is not available, using CPU")
        cfg.trainer.accelerator = "cpu"

    trainer = instantiate(cfg.inferer)

    trainer.test(model, dm)


def infer(config_path: str = "../configs", config_name: str = "default", **kwargs):
    """
    Run inference. `infer -- --help` for more info.

    Args:
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
    run_inferring(cfg)


if __name__ == "__main__":
    raise RuntimeError("Use `python commands.py infer`")
