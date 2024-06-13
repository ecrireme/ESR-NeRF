import os
import random
import socket
from datetime import datetime

import numpy as np
import rich
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from wandb import sdk as wandb_sdk


def customize_cfg(cfg: DictConfig, save_config=True):
    if cfg.system.debug:
        cfg.log.project = "debug"
        cfg.log.group = "debug"
        cfg.log.name = None

    if cfg.log.name is None:
        cfg.log.name = datetime.now().strftime("%y-%m-%d_%H-%M-%S_%f")

    cfg.log.root = str(cfg.log.root)
    cfg.log.project = str(cfg.log.project)
    cfg.log.group = str(cfg.log.group)
    cfg.log.name = str(cfg.log.name)

    cfg.app.phase = str(cfg.app.phase).lower()
    if cfg.app.phase != "train":
        cfg.system.tqdm_iters = 1

    exp_id = os.path.join(
        cfg.log.project,
        cfg.log.group,
        cfg.log.name,
        cfg.app.phase,
    )
    cfg.log.dir = os.path.join(cfg.log.root, "info", exp_id)
    cfg.log.ckpt_dir = os.path.join(cfg.log.root, "ckpt", exp_id)

    os.makedirs(cfg.log.dir, exist_ok=True)
    os.makedirs(cfg.log.ckpt_dir, exist_ok=True)
    if save_config:
        save_cfg(cfg)

    return cfg


def save_cfg(cfg: DictConfig):
    os.makedirs(cfg.log.dir, exist_ok=True)
    with open(os.path.join("cfg", "__hydra__.yaml"), "r") as f:
        hydra_custom_settings = f.readlines()

    with open(os.path.join(cfg.log.dir, "cfg.yaml"), "w") as f:
        f.writelines(hydra_custom_settings)
        f.write("\n\n")
        OmegaConf.save(config=cfg, f=f, resolve=True)

    OmegaConf.save(
        config=HydraConfig.get(),
        f=os.path.join(cfg.log.dir, "hydra_cfg.yaml"),
        resolve=True,
    )


def welcome_message(
    cfg: DictConfig, run: wandb_sdk.wandb_run.Run, detailed: bool = False
):
    """print welcome message with cfg
    cfg:
        cfg: namespace objects
    """
    if detailed:
        table = Table(title="Config")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="bright_green")

        for name, age in sorted(cfg.items(), key=lambda k: k[0]):  # type: ignore
            table.add_row(name, str(age))

        console = Console()
        console.print(table)

    rich.print(
        f"Logging: [magenta]{cfg.log.dir}[/magenta]\n"
        f"wandb id: [magenta]{run.id}[/magenta]\n"
        f"Machine: [blue]{socket.gethostname()}[/blue]\n"
    )


def finish_message():
    """print finsih message"""
    rich.print(
        Panel.fit(
            "[bold white]Experiment was done successfully! :raised_hands:",
            title="Finish",
        )
    )


def seed_everything(seed: int) -> int:
    """Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random In addition,
    sets the following environment variables:
    Code borrowed from Pytorch Lightning

    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if not (min_seed_value <= seed <= max_seed_value):
        rich.print(
            f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}"
        )
        seed = random.randint(min_seed_value, max_seed_value)

    # so users can verify the seed is properly set in distributed training.
    rich.print(f"Global seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed
