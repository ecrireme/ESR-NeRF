"""
run jobs
"""

import os
import shutil
import warnings

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from wandb.sdk.wandb_run import Run

from app import AppClass
from utils2 import manager, utils

# enable OpenEXR
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


@hydra.main(version_base=None, config_path=".")
def main(cfg: DictConfig) -> None:
    cfg = manager.customize_cfg(cfg)
    os.makedirs(cfg.log.dir, exist_ok=True)

    # copy codes
    if not os.path.exists(os.path.join(cfg.log.dir, "code")) and not cfg.system.debug:
        shutil.copytree(
            ".",
            os.path.join(cfg.log.dir, "code"),
            ignore=shutil.ignore_patterns(
                "dataset",
                "logs",
                ".*",
                "assets",
                "slurm_outputs",
                "build",
                "__pycache__",
                cfg.log.root,
            ),
            symlinks=True,
            dirs_exist_ok=True,
        )

    wandb_run: Run = wandb.init(
        entity=cfg.log.entity,
        project=cfg.log.project,
        group=cfg.log.group,
        name=cfg.log.name,
        job_type=cfg.app.phase,
        dir=cfg.log.dir,
        config=OmegaConf.to_object(cfg),  # type: ignore
        mode="offline" if cfg.log.offline else None,
        resume="auto",
        settings=wandb.Settings(_service_wait=3600),
    )

    manager.welcome_message(cfg, wandb_run, detailed=False)
    manager.seed_everything(cfg.system.seed)

    method: AppClass = utils.import_class("app." + cfg.app.cls)(cfg)
    method.load_dataset()

    method.load_model()
    method.process()

    manager.finish_message()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
