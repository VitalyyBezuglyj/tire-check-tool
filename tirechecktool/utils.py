from omegaconf import OmegaConf
import subprocess


def log_git_info(cfg: OmegaConf):
    """
    Log git info to the config.

    Args:
        cfg: hydra config
    """
    cfg.code_version.git_commit_id = subprocess.check_output(
        ["git", "describe", "--always"]
    ).strip()
