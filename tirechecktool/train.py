from logging import getLogger
from pathlib import Path

import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from .dataset import TireCheckDataModule
from .model import TireCheckModel


def run_training(cfg: OmegaConf):
    log = getLogger(__name__)
    log.setLevel(cfg.log_level.upper())

    pl.seed_everything(cfg.random_seed)
    torch.set_float32_matmul_precision(cfg.float_precision)

    dm = TireCheckDataModule(cfg=cfg)
    model = TireCheckModel(cfg=cfg)

    outs_path = Path(cfg.artifacts.base_path)

    loggers = []
    if cfg.artifacts.logs.save:
        loggers.append(pl.loggers.CSVLogger(outs_path / cfg.artifacts.logs.path, name=cfg.exp_name)),

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval=cfg.callbacks.lr_monitor.interval),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(max_depth=cfg.callbacks.model_summary.max_depth),
    ]
    if cfg.artifacts.checkpoint.save:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=outs_path / cfg.artifacts.checkpoint.path,
            monitor=cfg.artifacts.checkpoint.monitor,
            filename=cfg.artifacts.checkpoint.filename,
            save_top_k=cfg.artifacts.checkpoint.save_top_k,
            mode=cfg.artifacts.checkpoint.mode,
            save_last=cfg.artifacts.checkpoint.save_last,
        )

        callbacks.append(checkpoint_callback)

    if torch.cuda.is_available():
        accelerator = cfg.train.accelerator
    else:
        log.warning("CUDA is not available, using CPU")
        accelerator = "cpu"

    trainer = pl.Trainer(
        accelerator=accelerator,
        log_every_n_steps=cfg.train.log_every_n_steps,
        precision=cfg.train.precision,
        max_epochs=cfg.train.n_epoch,
        logger=loggers,
        callbacks=callbacks,
    )

    trainer.fit(model, dm)

    if cfg.artifacts.checkpoint.save:
        print(checkpoint_callback.best_model_path)

        best_model_name = outs_path / cfg.artifacts.checkpoint.path / "best.txt"

        with open(best_model_name, "w") as f:
            f.write(checkpoint_callback.best_model_path)


def train(config_name: str = "default", config_path: str = "conf", **kwargs):
    """
    Run training. `train -- --help` for more info.

    Args:
    :param config_path: path to config dir, relative to project root or absolute
    :param config_name: name of config file, inside config_path
    :param **kwargs: additional arguments, overrides for config. Can be passed as `--arg=value`.
    """
    print(f"Curr path: {Path.cwd()}")
    initialize(version_base="1.3", config_path=config_path, job_name="tirechecktool-train")
    cfg = compose(config_name=config_name, overrides=[f"{k}={v}" for k, v in kwargs.items()])
    print(OmegaConf.to_yaml(cfg))
    run_training(cfg)


if __name__ == "__main__":
    raise RuntimeError("Use `python commands.py train`")
