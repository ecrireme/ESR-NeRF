import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import imageio
import numpy as np
import rich
import torch
import torch.nn.functional as F
import trimesh
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from app import AppClass
from app.coarse.model import DVGO, VoxurfC
from app.utils.optimizer import create_optimizer_or_freeze_model
from data import DataClass
from data.dtu.dtu import DTU
from utils2.manager import save_cfg
from utils2.metric import DTU_CD, loss2psnr, rgb_lpips, rgb_ssim
from utils2.utils import BatchSampler, import_class, tqdm_safe


class Coarse(AppClass):
    """
    Pretrain radiance and SDF fields following the Voxurf's coarse training strategy.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # train
        self.train_ckpt = cfg.app.trainer.ckpt

        self.world_bound_scale = cfg.app.trainer.world_bound_scale
        self.bbox_thres = cfg.app.trainer.bbox_thres

        self.s_start = cfg.app.trainer.s_start
        self.s_inv_ratio = cfg.app.trainer.s_inv_ratio
        self.step_start = cfg.app.trainer.step_start
        self.step_end = cfg.app.trainer.step_end

        self.train_bs = cfg.app.trainer.batch_size
        self.n_iters = cfg.app.trainer.n_iters
        self.lrs = cfg.app.trainer.lrs
        self.lr_decay = cfg.app.trainer.lr_decay
        self.decay_steps = cfg.app.trainer.decay_steps

        self.weight_entropy_last = cfg.app.trainer.weight_entropy_last
        self.weight_tv_density = cfg.app.trainer.weight_tv_density
        self.weight_tv_color = cfg.app.trainer.weight_tv_color

        self.tvs = cfg.app.trainer.tvs
        self.tv_updates = cfg.app.trainer.tv_updates
        self.tv_from = cfg.app.trainer.tv_from
        self.tv_end = cfg.app.trainer.tv_end
        self.tv_every = cfg.app.trainer.tv_every

        self.vis_every = cfg.app.trainer.vis_every
        self.N_vis = cfg.app.trainer.N_vis
        self.save_every = cfg.app.trainer.save_every
        self.save_all = cfg.app.trainer.save_all

        if self.step_end < 0:
            self.step_end = self.n_iters * 10

        self.data_keys = ["rgbs", "rays_o", "rays_d", "viewdirs", "em_modes"]

        # eval
        self.eval_ckpt = cfg.app.eval.ckpt

        self.eval_bs = cfg.app.eval.batch_size

    def load_dataset(self):
        assert self.phase in [
            "train",
            "test_nv",
        ], "Coarse stage only supports train or test_nv phases."

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
        if self.cfg.app.phase == "train":
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
                ckpt = ckpt.replace("coarse.Coarse", "coarse.AlphaMask")
                if not os.path.exists(ckpt):
                    ckpt = None
                else:
                    rich.print(
                        "Found the pre-trained ckpt file by looking the experiment under the same project."
                    )
        else:
            is_resume = True

        # init model
        if ckpt is None:
            rich.print("Couldn't find the ckpt files. Training from scratch!")
            raise NotImplementedError(
                "Coarse stage needs the pre-trained alphamask model."
            )

        elif not is_resume:
            rich.print(f"load pre-trained weights from {ckpt}")

            self.global_step = 0

            data = self.train_dataset.all_data

            # prepare init args
            params = torch.load(ckpt, map_location=self.device)
            near = params["renderer"]["near"]
            far = params["renderer"]["far"]
            mask_xyz_min = params["renderer"]["xyz_min"]
            mask_xyz_max = params["renderer"]["xyz_max"]
            mask_alpha_init = params["renderer"]["cfg"].app.model.alpha_init
            mask_density = params["renderer"]["params"]["density"]
            alphamask = DVGO(
                params["renderer"]["cfg"], near, far, mask_xyz_min, mask_xyz_max
            ).to(self.device)
            alphamask.load_state_dict(params["renderer"]["params"])

            print("compute_bbox_by_coarse_geo: start")
            eps_time = time.time()
            interp = torch.stack(
                torch.meshgrid(
                    torch.linspace(
                        0, 1, alphamask.density.shape[2], device=self.device
                    ),
                    torch.linspace(
                        0, 1, alphamask.density.shape[3], device=self.device
                    ),
                    torch.linspace(
                        0, 1, alphamask.density.shape[4], device=self.device
                    ),
                ),
                -1,
            )
            dense_xyz = mask_xyz_min * (1 - interp) + mask_xyz_max * interp
            density = alphamask.grid_sampler(dense_xyz, alphamask.density)
            alpha = alphamask.activate_density(density)
            mask = alpha > self.bbox_thres
            active_xyz = dense_xyz[mask]
            xyz_min = active_xyz.amin(0)
            xyz_max = active_xyz.amax(0)
            print("compute_bbox_by_coarse_geo: xyz_min {}".format(xyz_min))
            print("compute_bbox_by_coarse_geo: xyz_max {}".format(xyz_max))
            eps_time = time.time() - eps_time
            print(
                "compute_bbox_by_coarse_geo: finish (eps time: {} secs)".format(
                    eps_time
                )
            )

            if abs(self.world_bound_scale - 1) > 1e-9:
                xyz_shift = (xyz_max - xyz_min) * (self.world_bound_scale - 1) / 2
                xyz_min -= xyz_shift
                xyz_max += xyz_shift

            # instantiate the model
            self.renderer = VoxurfC(
                self.cfg,
                near,
                far,
                xyz_min,
                xyz_max,
                mask_xyz_min,
                mask_xyz_max,
                mask_alpha_init,
                mask_density,
                self.s_start,
            ).to(self.device)

            # setup optimizer
            self.optimizer = create_optimizer_or_freeze_model(self.renderer, **self.lrs)
            rich.print(self.optimizer)

            # trim training dataset
            mask = self.renderer.filter_training_rays_in_maskcache_sampling(
                data["rays_o"].to(self.device),
                data["rays_d"].to(self.device),
                self.eval_bs,
            )
            self.sampler = BatchSampler(self.cfg, data, self.data_keys, self.train_bs)
            self.sampler.filter(mask)
            self.sampler.shuffle()

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
            mask_xyz_min = params["renderer"]["mask_xyz_min"]
            mask_xyz_max = params["renderer"]["mask_xyz_max"]
            mask_alpha_init = params["renderer"]["mask_alpha_init"]
            mask_density = params["renderer"]["mask_density"]
            s_val = params["renderer"]["s_val"]

            self.renderer = VoxurfC(
                self.cfg,
                near,
                far,
                xyz_min,
                xyz_max,
                mask_xyz_min,
                mask_xyz_max,
                mask_alpha_init,
                mask_density,
                s_val,
            ).to(self.device)
            self.renderer.load_state_dict(params["renderer"]["params"])
            self.optimizer = create_optimizer_or_freeze_model(self.renderer, **self.lrs)
            self.optimizer.load_state_dict(params["trainer"]["optimizer"])
            rich.print(self.optimizer)

            self.tvs = params["trainer"]["tvs"]

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
        mask_xyz_min = params["renderer"]["mask_xyz_min"]
        mask_xyz_max = params["renderer"]["mask_xyz_max"]
        mask_alpha_init = params["renderer"]["mask_alpha_init"]
        mask_density = params["renderer"]["mask_density"]
        s_val = params["renderer"]["s_val"]

        self.renderer = VoxurfC(
            self.cfg,
            near,
            far,
            xyz_min,
            xyz_max,
            mask_xyz_min,
            mask_xyz_max,
            mask_alpha_init,
            mask_density,
            s_val,
        ).to(self.device)
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
            s_val = (
                min(self.global_step, self.step_end) - self.step_start
            ) / self.s_inv_ratio + self.s_start

            self.optimizer.zero_grad(set_to_none=True)
            results = self.renderer(s_val=s_val, **batch)
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

            if (
                self.global_step > self.tv_from
                and self.global_step < self.tv_end
                and self.global_step % self.tv_every == 0
            ):
                loss += self.weight_tv_density * self.renderer.density_total_variation(
                    sdf_tv=self.tvs["sdf"],
                    smooth_grad_tv=self.tvs["smooth_grad"],
                )
                loss += self.weight_tv_color * self.renderer.color_total_variation()

            loss.backward()
            self.optimizer.step()

            logs["srgb/MSE"].append(render_loss)
            logs["srgb/PSNR"].append(loss2psnr(render_loss))

            # update lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= decay_factor

            if self.global_step in self.decay_steps:
                for k, v in self.decay_steps[self.global_step].items():
                    self.optimizer.name2pg[k]["lr"] *= v
                    print(
                        "- " * 10
                        + "[Decay lrate] for {} by {}".format(k, v)
                        + " -" * 10
                    )

            if self.global_step in self.tv_updates:
                for k, v in self.tv_updates[self.global_step].items():
                    self.tvs[k] = v
                print(
                    "- " * 10
                    + "[Update tv]: "
                    + str(self.tv_updates[self.global_step])
                    + " -" * 10
                )

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
                            "mask_xyz_min": self.renderer.mask_xyz_min,
                            "mask_xyz_max": self.renderer.mask_xyz_max,
                            "mask_alpha_init": self.renderer.mask_alpha_init,
                            "mask_density": self.renderer.mask_density,
                            "s_val": self.renderer.s_val,
                            "params": self.renderer.state_dict(),
                        },
                        "trainer": {
                            "global_step": self.global_step,
                            "batch_st": self.sampler.batch_st,
                            "data_idxs": self.sampler.data_idxs,
                            "tvs": self.tvs,
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
            ).astype("int8")
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
                "viewdirs": data["viewdirs"].to(self.device),
                "em_modes": data["em_modes"][..., 0].to(self.device),
                "pos_rt": data["poses"][:3, :3].to(self.device),
            }

            results = {}
            for chunk_idx in torch.arange(len(data["rgbs"])).split(self.eval_bs):
                chunk = {
                    "rays_o": batch["rays_o"][chunk_idx],
                    "rays_d": batch["rays_d"][chunk_idx],
                    "viewdirs": batch["viewdirs"][chunk_idx],
                    "em_modes": batch["em_modes"],
                    "pos_rt": batch["pos_rt"],
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

        # render non-image-based results
        scn_metrics: Dict[str, float] = {}

        # mesh extraction
        vertices, triangles = self.renderer.extract_geometry(batch_size=self.eval_bs)
        scale_mat = self.test_dataset.scale_mat.cpu().numpy()
        vertices = vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]
        mesh = trimesh.Trimesh(vertices, triangles)

        if isinstance(self.test_dataset, DTU):
            _, _, mean = DTU_CD(mesh, *self.test_dataset.pcd)
            metrics["mesh/CD"] = [mean] + [None] * (len(img_idxes) - 1)

        # save to HDD
        mesh.export(os.path.join(mesh_dir, "mesh.ply"))

        # image related
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
            f.write(
                "Scene metrics: \n"
                + ", ".join(
                    [f"{k}: {scn_metrics[k]}" for k in sorted(scn_metrics.keys())]
                )
                + "\n\n"
            )

            ks = sorted(metrics.keys())
            metrics_compact = {k: [v for v in metrics[k] if v is not None] for k in ks}
            f.write(
                "Image metrics: \n"
                + ", ".join([f"{k}: {float(np.mean(metrics_compact[k]))}" for k in ks])
                + "\n"
            )
            for i in range(len(img_idxes)):
                f.write(
                    f"Index {i}, "
                    + ", ".join(
                        [
                            f"{k}: "
                            + (
                                f"{float(metrics[k][i])}"
                                if metrics[k][i] is not None
                                else "null"
                            )
                            for k in ks
                        ]
                    )
                    + "\n"
                )
            metrics = metrics_compact

        # logging on wandb
        prefix = self.test_dataset.phase + "/"
        logs: Dict[str, Any] = {}
        for k, v in scn_metrics.items():
            logs[prefix + "metric/" + k] = v
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
        return f"{self.global_step:010}"
