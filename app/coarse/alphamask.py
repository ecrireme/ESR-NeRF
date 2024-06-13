import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import imageio
import numpy as np
import rich
import torch
import torch.nn.functional as F
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from app import AppClass
from app.coarse.model import DVGO
from app.utils.optimizer import create_optimizer_or_freeze_model
from data import DataClass
from utils2.manager import save_cfg
from utils2.metric import loss2psnr, rgb_lpips, rgb_ssim
from utils2.utils import BatchSampler, import_class, tqdm_safe


class AlphaMask(AppClass):
    """
    Leaarn alphamask following the Voxurf training strategy.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # train
        self.train_ckpt = cfg.app.trainer.ckpt

        self.train_bs = cfg.app.trainer.batch_size
        self.n_iters = cfg.app.trainer.n_iters
        self.lr_decay = cfg.app.trainer.lr_decay
        self.lrs = cfg.app.trainer.lrs

        self.world_bound_scale = cfg.app.trainer.world_bound_scale

        self.weight_entropy_last = cfg.app.trainer.weight_entropy_last
        self.weight_rgbper = cfg.app.trainer.weight_rgbper

        self.vis_every = cfg.app.trainer.vis_every
        self.N_vis = cfg.app.trainer.N_vis
        self.save_every = cfg.app.trainer.save_every
        self.save_all = cfg.app.trainer.save_all

        self.data_keys = ["rgbs", "rays_o", "rays_d", "em_modes"]

        # eval
        self.eval_ckpt = cfg.app.eval.ckpt

        self.eval_bs = cfg.app.eval.batch_size

    def load_dataset(self):
        assert self.phase in [
            "train",
            "test_nv",
        ], "AlphaMask stage only supports train or test_nv phases."

        assert (
            self.cfg.data.batch_type == "nerf"
        ), "ESR-NeRF requires the nerf style ray batching."

        _data_cls = import_class("data." + self.cfg.data.cls)
        if self.phase == "train":
            self.train_dataset: DataClass = _data_cls(self.cfg, "train")
            self.test_dataset: DataClass = _data_cls(self.cfg, "test_nv")
        else:
            self.test_dataset: DataClass = _data_cls(self.cfg, self.phase)

    def load_model(self):
        if self.phase == "train":
            self.load_train_model()
        else:
            self.load_eval_model()

    def load_train_model(self):
        # select checkpoints
        is_resume = False
        ckpt = os.path.join(self.cfg.log.dir, "checkpoints", "last.ckpt")
        if not os.path.exists(ckpt):  # resume first
            if self.train_ckpt is not None:
                if os.path.exists(self.train_ckpt):
                    ckpt = self.train_ckpt
                else:
                    ckpt = None
                    rich.print(
                        f"wrong ckpt path: {self.train_ckpt}. [bold red]Unable to load weights!"
                    )
            else:
                ckpt = None
        else:
            is_resume = True

        # init model
        if ckpt is None:
            rich.print("Couldn't find the ckpt files. Training from scratch!")
            self.global_step = 0

            data = self.train_dataset.all_data
            near, far = self.train_dataset.near_far

            # compute init args
            print("compute_bbox_by_cam_frustrm: start")
            xyz_min = torch.Tensor([np.inf, np.inf, np.inf]).to(self.device)
            xyz_max = -xyz_min
            for rays_o, viewdirs in zip(
                data["rays_o"].to(self.device).split(self.eval_bs, dim=0),
                data["viewdirs"].to(self.device).split(self.eval_bs, dim=0),
            ):
                pts_nf = torch.stack(
                    [rays_o + viewdirs * near, rays_o + viewdirs * far]
                )
                xyz_min = torch.minimum(xyz_min, pts_nf.amin((0, 1)))
                xyz_max = torch.maximum(xyz_max, pts_nf.amax((0, 1)))
            print("compute_bbox_by_cam_frustrm: xyz_min {}".format(xyz_min))
            print("compute_bbox_by_cam_frustrm: xyz_max {}".format(xyz_max))
            print("compute_bbox_by_cam_frustrm: finish")

            if abs(self.world_bound_scale - 1) > 1e-9:
                xyz_shift = (xyz_max - xyz_min) * (self.world_bound_scale - 1) / 2
                xyz_min -= xyz_shift
                xyz_max += xyz_shift

            # instantiate the model
            self.renderer = DVGO(self.cfg, near, far, xyz_min, xyz_max).to(self.device)

            # init model parameters
            wh = self.train_dataset.image_size[0] * self.train_dataset.image_size[1]
            rays_o = data["rays_o"].view(-1, wh, 3)
            rays_d = data["rays_d"].view(-1, wh, 3)
            self.renderer.maskout_near_cam_vox(rays_o[:, 0].to(self.device))

            cnt = self.renderer.voxel_count_views(
                rays_o.to(self.device), rays_d.to(self.device), self.eval_bs
            )
            with torch.no_grad():
                self.renderer.density[cnt <= 2] = -100

            # setup optimizer
            self.optimizer = create_optimizer_or_freeze_model(self.renderer, **self.lrs)
            self.optimizer.set_pervoxel_lr(cnt)
            rich.print(self.optimizer)

            self.sampler = BatchSampler(self.cfg, data, self.data_keys, self.train_bs)
            self.sampler.shuffle()

        elif not is_resume:
            rich.print(f"load pre-trained weights from {ckpt}")
            raise NotImplementedError

        else:
            params = torch.load(ckpt, map_location=self.device)

            self.global_step = params["trainer"]["global_step"] + 1

            batch_st = params["trainer"]["batch_st"]
            data_idxs = params["trainer"]["data_idxs"]

            data = self.train_dataset.all_data

            self.sampler = BatchSampler(
                self.cfg, data, self.data_keys, self.train_bs, batch_st, data_idxs
            )

            near = params["renderer"]["near"]
            far = params["renderer"]["far"]
            xyz_min = params["renderer"]["xyz_min"]
            xyz_max = params["renderer"]["xyz_max"]

            self.renderer = DVGO(self.cfg, near, far, xyz_min, xyz_max).to(self.device)
            self.renderer.load_state_dict(params["renderer"]["params"])
            self.optimizer = create_optimizer_or_freeze_model(self.renderer, **self.lrs)
            self.optimizer.load_state_dict(params["trainer"]["optimizer"])
            rich.print(self.optimizer)

            rich.print(
                f"resume training starting from the global step: {self.global_step}"
            )

    def load_eval_model(self):
        ckpt = self.eval_ckpt
        if ckpt is None:
            hydra_conf = HydraConfig.get()
            cn = Path(hydra_conf.job.config_name)  # type: ignore
            ckpt = str(cn.parent / "checkpoints" / "last.ckpt")
            print(f"ckpt is None, use the last ckpt in the {cn.parent} dir.")

        assert os.path.exists(
            ckpt
        ), f"wrong ckpt path: {ckpt}. [bold red]Unable to load weights!"

        params = torch.load(ckpt, map_location=self.device)

        self.global_step = params["trainer"]["global_step"]

        near = params["renderer"]["near"]
        far = params["renderer"]["far"]
        xyz_min = params["renderer"]["xyz_min"]
        xyz_max = params["renderer"]["xyz_max"]

        self.renderer = DVGO(self.cfg, near, far, xyz_min, xyz_max).to(self.device)
        self.renderer.load_state_dict(params["renderer"]["params"])

        rich.print(
            f"Loaded the checkpoints from {ckpt}. \n global step: {self.global_step}."
        )

    def process(self):
        if self.phase == "train":
            self.learn()
        else:
            self.evaluate()

    def learn(self):
        self.renderer.train()
        decay_factor = 0.1 ** (1 / (self.lr_decay * 1000))

        pbar = tqdm_safe(range(self.global_step, self.n_iters), colour="green")

        ckpt_dir = os.path.join(self.cfg.log.dir, "checkpoints")
        if not os.path.exists(ckpt_dir):
            os.symlink(
                os.path.abspath(self.cfg.log.ckpt_dir),
                ckpt_dir,
                target_is_directory=True,
            )
        ckpt_path = os.path.join(ckpt_dir, "last.ckpt")

        logs: Dict[str, List] = {"srgb/MSE": [], "srgb/PSNR": []}

        for self.global_step in pbar:
            batch = self.sampler.sample()

            self.optimizer.zero_grad(set_to_none=True)
            results = self.renderer(**batch)
            rgbs = batch["rgbs"]

            results["srgb/rgb"] = (
                results["srgb/rgb"] + results["etc/white_bg"] * self.white_bg
            ).clamp(min=0.0, max=1.0)

            loss = F.mse_loss(results["srgb/rgb"], rgbs)
            render_loss = loss.detach().item()

            pout = results["etc/alphainv_cum"][..., -1].clamp(1e-6, 1 - 1e-6)
            entropy_last_loss = -(
                pout * torch.log(pout) + (1 - pout) * torch.log(1 - pout)
            ).mean()
            loss += self.weight_entropy_last * entropy_last_loss

            rgbper = (results["srgb/raw_rgb"] - rgbs.unsqueeze(-2)).pow(2).sum(-1)
            rgbper_loss = (rgbper * results["etc/weights"].detach()).sum(-1).mean()
            loss += self.weight_rgbper * rgbper_loss

            loss.backward()
            self.optimizer.step()

            logs["srgb/MSE"].append(render_loss)
            logs["srgb/PSNR"].append(loss2psnr(render_loss))

            # update lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= decay_factor

            if self.global_step % self.cfg.system.tqdm_iters == 0:
                mse = float(np.mean(logs["srgb/MSE"]))
                psnr = float(np.mean(logs["srgb/PSNR"]))
                logs["srgb/MSE"] = []
                logs["srgb/PSNR"] = []
                if not self.cfg.system.debug:
                    pbar.set_description(
                        f"Iter {self.global_step:05d}"
                        + " (s)"
                        + f" psnr = {psnr:.2f}"
                        + f" mse = {mse:.6f}"
                    )
                wandb.log(
                    {"train/metric/srgb/MSE": mse, "train/metric/srgb/PSNR": psnr},
                    step=self.global_step,
                )

            if (
                self.global_step % self.vis_every == self.vis_every - 1
                or self.global_step == self.n_iters - 1
            ):
                assert (
                    self.N_vis != 0
                ), "N_vis has to be larger than 0. or -1 for render all iamges"
                self.evaluate(self.N_vis)

            if (
                self.global_step % self.save_every == self.save_every - 1
                or self.global_step == self.n_iters - 1
            ):
                torch.save(
                    {
                        "renderer": {
                            "cfg": self.renderer.cfg,
                            "near": self.renderer.near,
                            "far": self.renderer.far,
                            "xyz_min": self.renderer.xyz_min,
                            "xyz_max": self.renderer.xyz_max,
                            "params": self.renderer.state_dict(),
                        },
                        "trainer": {
                            "global_step": self.global_step,
                            "batch_st": self.sampler.batch_st,
                            "data_idxs": self.sampler.data_idxs,
                            "optimizer": self.optimizer.state_dict(),
                        },
                    },
                    ckpt_path,
                )
                if self.save_all:
                    shutil.copy2(
                        ckpt_path,
                        os.path.join(ckpt_dir, f"{self.pretty_global_step}.ckpt"),
                    )

        self.cfg.app.eval.ckpt = ckpt_path
        save_cfg(self.cfg)

    @torch.no_grad()
    def evaluate(self, N_vis: int = -1):
        text_dir = os.path.join(self.cfg.log.dir, "text", self.pretty_global_step)
        image_dir = os.path.join(self.cfg.log.dir, "image", self.pretty_global_step)
        video_dir = os.path.join(self.cfg.log.dir, "video", self.pretty_global_step)
        mesh_dir = os.path.join(self.cfg.log.dir, "mesh", self.pretty_global_step)
        os.makedirs(text_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(mesh_dir, exist_ok=True)

        if N_vis > 0:
            interval = len(self.test_dataset) // math.ceil(N_vis / 2)
            img_idxes = np.sort(
                np.concatenate(
                    [
                        np.arange(0, len(self.test_dataset), interval),
                        np.arange(1, len(self.test_dataset), interval),
                    ],
                    axis=-1,
                )
            )
        else:
            img_idxes = np.arange(0, len(self.test_dataset))

        width, height = self.test_dataset.image_size
        metrics: Dict[str, List[float]] = {
            "srgb/MSE": [],
            "srgb/PSNR": [],
            "srgb/SSIM": [],
            "srgb/LPIPS_ALEX": [],
        }
        targets = []
        renders: Dict[str, List[torch.Tensor]] = {}

        self.renderer.eval()
        for i in tqdm_safe(img_idxes, desc="eval", leave=False):
            data = self.test_dataset[i]
            rgbs = data["rgbs"]
            batch = {
                "rays_o": data["rays_o"].to(self.device),
                "rays_d": data["rays_d"].to(self.device),
                "em_modes": data["em_modes"][..., 0].to(self.device),
            }

            results = {}
            for chunk_idx in torch.arange(len(data["rgbs"])).split(self.eval_bs):
                chunk = {
                    "rays_o": batch["rays_o"][chunk_idx],
                    "rays_d": batch["rays_d"][chunk_idx],
                    "em_modes": batch["em_modes"],
                }
                for k, v in self.renderer(**chunk).items():
                    if k not in results:
                        results[k] = []
                    results[k].append(v)

            for k, v in results.items():
                results[k] = (
                    torch.cat(v, dim=0).reshape(height, width, -1).squeeze(-1).cpu()
                )

            for k, v in results.items():
                if k != "etc/white_bg":
                    wbg = results["etc/white_bg"] * self.white_bg
                    if v.dim() == 3:
                        wbg = wbg.unsqueeze(-1)
                    results[k] = (v + wbg).clamp(min=0.0, max=1.0)
                else:
                    results[k] = v.clamp(min=0.0, max=1.0)

            rgbs = rgbs.reshape(height, width, 3)
            pred = results["srgb/rgb"]

            metrics["srgb/MSE"].append(F.mse_loss(pred, rgbs).item())
            metrics["srgb/PSNR"].append(loss2psnr(metrics["srgb/MSE"][-1]))
            metrics["srgb/SSIM"].append(rgb_ssim(pred, rgbs, 1))
            metrics["srgb/LPIPS_ALEX"].append(
                rgb_lpips(rgbs.numpy(), pred.numpy(), "alex", self.device)
            )

            targets.append((rgbs.numpy() * 255).astype("uint8"))
            for k, v in results.items():
                if k not in renders:
                    renders[k] = []
                renders[k].append((v.numpy() * 255).astype("uint8"))

        # save to HDD
        for k, v in renders.items():
            result_dir = os.path.join(image_dir, *k.split("/"))
            os.makedirs(result_dir, exist_ok=True)
            for i, _v in enumerate(v):
                imageio.imwrite(os.path.join(result_dir, f"{i:03d}.png"), _v)

        for k, v in renders.items():
            _k = k.split("/")
            result_dir = os.path.join(video_dir, *_k[:-1])
            os.makedirs(result_dir, exist_ok=True)
            imageio.mimwrite(
                os.path.join(result_dir, f"{_k[-1]}.mp4"),
                v,  # type: ignore
                fps=30,
                codec="h264",
                quality=10,
            )

        with open(os.path.join(text_dir, "mean.txt"), "w") as f:
            ks = sorted(metrics.keys())
            f.write(
                "Image metrics: \n"
                + ", ".join([f"{k}: {float(np.mean(metrics[k]))}" for k in ks])
                + "\n"
            )
            for i in range(len(img_idxes)):
                f.write(
                    f"Index {i}, "
                    + ", ".join([f"{k}: {float(metrics[k][i])}" for k in ks])
                    + "\n"
                )

        # logging on wandb
        prefix = self.test_dataset.phase + "/"
        logs: Dict[str, Any] = {}
        for k, v in metrics.items():
            logs[prefix + "metric/" + k] = float(np.mean(v))
        for k, v in renders.items():
            _k = k.split("/")
            logs[prefix + "image/" + k] = [wandb.Image(_v) for _v in v]
            logs[prefix + "video/" + k] = wandb.Video(
                os.path.join(video_dir, *_k[:-1], f"{_k[-1]}.mp4"), fps=30, format="mp4"
            )

        wandb.log(logs, step=self.global_step)
        self.renderer.train()

    @property
    def pretty_global_step(self):
        return f"{self.global_step:010}"
