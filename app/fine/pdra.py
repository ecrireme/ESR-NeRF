import math
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import cv2
import imageio
import numpy as np
import rich
import torch
import torch.nn.functional as F
import trimesh
import wandb
from einops import rearrange
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from app import AppClass
from app.fine.model import ESRNeRF
from app.utils.optimizer import CosineLR, create_optimizer_or_freeze_model
from data import DataClass
from utils2.image import apply_gamma_curve
from utils2.manager import save_cfg
from utils2.metric import IoU, loss2psnr, rgb_lpips, rgb_ssim
from utils2.utils import LightDict, RayGroupManager, import_class, tqdm_safe


class PDRA(AppClass):
    """
    Light Transport Segements learning with Progressive Discovery of Reflection Areas
    Progressively filtering real emissive sources.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # train
        self.train_ckpt = cfg.app.trainer.ckpt

        self.brdf_color_init = cfg.app.trainer.brdf_color_init

        self.s_start = cfg.app.trainer.s_start
        self.s_inv_ratio = cfg.app.trainer.s_inv_ratio
        self.step_start = cfg.app.trainer.step_start
        self.step_end = cfg.app.trainer.step_end

        self.group_interval = cfg.app.trainer.group_interval
        self.prog_start = cfg.app.trainer.prog_start
        self.prog_slope = cfg.app.trainer.prog_slope
        self.prog_end_step = cfg.app.trainer.prog_end_step
        if self.prog_end_step == -1:
            self.prog_end_step = cfg.app.trainer.n_iters

        self.train_uncert_bs = cfg.app.trainer.uncert_batch_size
        self.train_cert_bs = cfg.app.trainer.cert_batch_size
        self.n_iters = cfg.app.trainer.n_iters
        self.lrs = cfg.app.trainer.lrs
        self.decay_steps = cfg.app.trainer.decay_steps

        self.weight_entropy_last = cfg.app.trainer.weight_entropy_last
        self.weight_tv_density = cfg.app.trainer.weight_tv_density
        self.weight_linear = cfg.app.trainer.weight_linear
        self.weight_lts = cfg.app.trainer.weight_lts
        self.weight_lts_l = cfg.app.trainer.weight_lts_l
        self.weight_lts_r = cfg.app.trainer.weight_lts_r
        self.weight_normal_smooth = cfg.app.trainer.weight_normal_smooth
        self.weight_emit_smooth = cfg.app.trainer.weight_emit_smooth
        self.weight_emit_supp = cfg.app.trainer.weight_emit_supp

        self.normal_eps = cfg.app.trainer.normal_eps
        self.emit_eps = cfg.app.trainer.emit_eps

        self.tvs = cfg.app.trainer.tvs
        self.tv_from = cfg.app.trainer.tv_from
        self.tv_end = cfg.app.trainer.tv_end
        self.tv_every = cfg.app.trainer.tv_every
        self.tv_dense_before = cfg.app.trainer.tv_dense_before

        self.em_every = cfg.app.trainer.em_every
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

        self.eval_uncert_bs = cfg.app.eval.uncert_batch_size
        self.eval_cert_bs = cfg.app.eval.cert_batch_size
        self.eval_niters = cfg.app.eval.n_iters

        self.mask_dilation_ks = cfg.app.eval.mask_dilation_ks

        self.eval_lrs = cfg.app.eval.lrs
        self.eval_weight_lts = cfg.app.eval.weight_lts

        self.render_pbr = cfg.app.eval.render_pbr
        self.chunk_sz = cfg.app.eval.chunk_size
        self.envmap_height = cfg.app.eval.envmap_height
        self.envmap_width = cfg.app.eval.envmap_width

        # some fixed var
        self.quantile = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=self.device)

    def load_dataset(self):
        assert (
            self.cfg.data.batch_type == "nerf"
        ), "ESR-NeRF requires the nerf style ray batching."

        _data_cls = import_class("data." + self.cfg.data.cls)
        if self.phase == "train":
            self.train_dataset: DataClass = _data_cls(self.cfg, "train")
            self.test_dataset: DataClass = _data_cls(self.cfg, "test_nv")
        else:
            self.train_dataset: DataClass = _data_cls(self.cfg, "train")
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
                ckpt = ckpt.replace("fine.PDRA", "fine.LTS")
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
            raise NotImplementedError("PDRA stage needs the pre-trained LTS model.")

        elif not is_resume:
            rich.print(f"load pre-trained weights from {ckpt}")

            self.global_step = 0

            data = self.train_dataset.all_data

            # load init args
            params = torch.load(ckpt, map_location=self.device)

            near = params["renderer"]["near"]
            far = params["renderer"]["far"]
            xyz_min = params["renderer"]["xyz_min"]
            xyz_max = params["renderer"]["xyz_max"]
            mask_xyz_min = params["renderer"]["mask_xyz_min"]
            mask_xyz_max = params["renderer"]["mask_xyz_max"]
            mask_alpha_init = params["renderer"]["mask_alpha_init"]
            mask_density = params["renderer"]["mask_density"]
            s_val = params["renderer"]["s_val"]
            num_voxels = params["renderer"]["num_voxels"]

            # instantiate the model
            self.renderer = ESRNeRF(
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
                num_voxels,
            ).to(self.device)

            # load weights from the fine stage
            self.renderer.load_state_dict(params["renderer"]["params"], strict=False)

            if self.brdf_color_init:
                self.renderer.brdf.load_state_dict(self.renderer.off_color.state_dict())

            # setup optimizer
            self.optimizer = create_optimizer_or_freeze_model(self.renderer, **self.lrs)
            self.lr_scheduler = CosineLR(self.cfg, self.global_step)
            rich.print(self.optimizer)

            # trim training dataset
            self.sampler = RayGroupManager(
                self.cfg,
                data,
                self.data_keys,
                self.train_uncert_bs,
                self.train_cert_bs,
                uncert_data_idxs=params["trainer"]["data_idxs"],
                cert_data_idxs=None,
            )
            self.update_ray_groups(self.k_val)
            self.sampler.shuffle()

        else:
            params = torch.load(ckpt, map_location=self.device)

            self.global_step = params["trainer"]["global_step"] + 1

            uncert_batch_st = params["trainer"]["uncert_batch_st"]
            cert_batch_st = params["trainer"]["cert_batch_st"]
            uncert_data_idxs = params["trainer"]["uncert_data_idxs"]
            cert_data_idxs = params["trainer"]["cert_data_idxs"]

            data = self.train_dataset.all_data

            self.sampler = RayGroupManager(
                self.cfg,
                data,
                self.data_keys,
                self.train_uncert_bs,
                self.train_cert_bs,
                uncert_batch_st,
                cert_batch_st,
                uncert_data_idxs=uncert_data_idxs,
                cert_data_idxs=cert_data_idxs,
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
            num_voxels = params["renderer"]["num_voxels"]

            self.renderer = ESRNeRF(
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
                num_voxels,
            ).to(self.device)
            self.renderer.load_state_dict(params["renderer"]["params"])
            self.optimizer = create_optimizer_or_freeze_model(self.renderer, **self.lrs)
            self.optimizer.load_state_dict(params["trainer"]["optimizer"])
            self.lr_scheduler = CosineLR(self.cfg, self.global_step)
            rich.print(self.optimizer)

            rich.print(
                f"resume training starting from the global step: {self.global_step}"
            )

    def load_eval_model(self):
        if self.eval_ckpt is None:
            hydra_conf = HydraConfig.get()
            cn = Path(hydra_conf.job.config_name)  # type: ignore
            self.eval_ckpt = str(cn.parent / "checkpoints" / "last.ckpt")
            print(f"ckpt is None, use the last ckpt in the {cn.parent} dir.")

        assert os.path.exists(
            self.eval_ckpt
        ), f"wrong ckpt path: {self.eval_ckpt}. [bold red]Unable to load weights!"

        params = torch.load(self.eval_ckpt, map_location=self.device)

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
        num_voxels = params["renderer"]["num_voxels"]

        self.renderer = ESRNeRF(
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
            num_voxels,
        ).to(self.device)
        self.renderer.load_state_dict(params["renderer"]["params"])

        uncert_data_idxs = params["trainer"]["uncert_data_idxs"]
        cert_data_idxs = params["trainer"]["cert_data_idxs"]

        data = self.train_dataset.all_data

        self.sampler = RayGroupManager(
            self.cfg,
            data,
            self.data_keys,
            self.eval_uncert_bs,
            self.eval_cert_bs,
            uncert_data_idxs=uncert_data_idxs,
            cert_data_idxs=cert_data_idxs,
        )

        rich.print(
            f"Loaded the checkpoints from {self.eval_ckpt}. \n global step: {self.global_step}."
        )

    def process(self):
        if self.phase == "train":
            self.learn()
        else:
            self.evaluate()

    def learn(self):
        self.renderer.train()
        self.renderer.pdra_mode = True

        pbar = tqdm_safe(range(self.global_step, self.n_iters), colour="green")

        ckpt_dir = os.path.join(self.cfg.log.dir, "checkpoints")
        if not os.path.exists(ckpt_dir):
            os.symlink(
                os.path.abspath(self.cfg.log.ckpt_dir),
                ckpt_dir,
                target_is_directory=True,
            )
        ckpt_path = os.path.join(ckpt_dir, "last.ckpt")

        logs: Dict[str, List] = {
            "srgb/MSE": [],
            "srgb/PSNR": [],
            "lin/MSE": [],
            "lin/PSNR": [],
            "lin/pbr/off_MSE": [],
            "lin/pbr/emo_MSE": [],
        }

        for self.global_step in pbar:
            if self.global_step % self.group_interval == self.group_interval - 1:
                self.update_ray_groups(self.k_val)

            batch = self.sampler.sample()
            s_val = self.s_val

            self.optimizer.zero_grad(set_to_none=True)
            results = self.renderer(
                s_val=s_val, normal_eps=self.normal_eps, emit_eps=self.emit_eps, **batch
            )
            rgbs = batch["rgbs"]

            white_bg = results["etc/white_bg"] * self.white_bg

            results["srgb/rgb"] = (results["srgb/rgb"] + white_bg).clamp(
                min=0.0, max=1.0
            )
            results["lin/rgb"] = (results["lin/rgb"] + white_bg).clamp(min=0.0)

            loss = F.mse_loss(results["srgb/rgb"], rgbs)
            render_loss = loss.detach().item()

            lin_render_loss = F.mse_loss(
                apply_gamma_curve(
                    torch.where(
                        rgbs >= 1,
                        results["lin/rgb"].clamp(max=1.0),
                        results["lin/rgb"],
                    )
                ),
                rgbs,
            )
            loss += self.weight_linear * lin_render_loss
            lin_render_loss = lin_render_loss.detach().item()

            pbr_off_loss = F.l1_loss(results["lin/pbr/off"], results["lin/pbr/off_hat"])
            loss += self.weight_lts * pbr_off_loss
            pbr_off_loss = pbr_off_loss.detach().item()

            pbr_emo_loss_l = F.l1_loss(
                results["lin/pbr/emo"].detach(), results["lin/pbr/emo_hat"]
            )
            pbr_emo_loss_r = F.l1_loss(
                results["lin/pbr/emo"], results["lin/pbr/emo_hat"].detach()
            )
            loss += self.weight_lts * (
                self.weight_lts_l * pbr_emo_loss_l + self.weight_lts_r * pbr_emo_loss_r
            )
            pbr_emo_loss = pbr_emo_loss_l.detach().item()

            emu = results["etc/emit_uncert"]
            emc = results["etc/emit_cert"]
            if len(emc):
                em_loss = torch.pow(emc, 2).mean()
                loss += self.weight_emit_supp * em_loss
                em_loss = em_loss.detach().item()
            else:
                em_loss = 0

            if self.global_step % self.em_every == self.em_every - 1:
                rich.print(f"step {self.global_step}")
                with torch.no_grad():
                    if len(emc):
                        rich.print(f"em_supp_loss: {em_loss}")
                        # rich.print(f"emc(max, min): {emc.max(), emc.min()}")
                        rich.print(
                            f"certain quantile: \n {emc.quantile(self.quantile, dim=0)}"
                        )
                    if len(emu):
                        # rich.print(f"emu(max, min): {emu.max(), emu.min()}")
                        rich.print(
                            f"uncertain quantile: \n {emu.quantile(self.quantile, dim=0)}"
                        )

            pout = results["etc/alphainv_cum"][..., -1].clamp(1e-6, 1 - 1e-6)
            entropy_last_loss = -(
                pout * torch.log(pout) + (1 - pout) * torch.log(1 - pout)
            ).mean()
            loss += self.weight_entropy_last * entropy_last_loss

            loss += self.weight_normal_smooth * F.l1_loss(
                results["etc/normal"], results["etc/normal_eps"]
            )

            loss += self.weight_emit_smooth * F.l1_loss(
                results["etc/emit"], results["etc/emit_eps"]
            )

            do_tv = (
                self.global_step > self.tv_from
                and self.global_step < self.tv_end
                and self.global_step % self.tv_every == 0
            )
            if do_tv:
                loss += self.weight_tv_density * self.renderer.density_total_variation(
                    sdf_tv=0,
                    smooth_grad_tv=self.tvs["smooth_grad"],
                )

            loss.backward()

            if do_tv:
                self.renderer.sdf_total_variation_add_grad(
                    self.weight_tv_density * self.tvs["sdf"] / len(rgbs),
                    self.global_step < self.tv_dense_before,
                )

            self.optimizer.step()

            logs["srgb/MSE"].append(render_loss)
            logs["srgb/PSNR"].append(loss2psnr(render_loss))
            logs["lin/MSE"].append(lin_render_loss)
            logs["lin/PSNR"].append(loss2psnr(lin_render_loss))
            logs["lin/pbr/off_MSE"].append(pbr_off_loss)
            logs["lin/pbr/emo_MSE"].append(pbr_emo_loss)

            # update lr
            decay_factor = self.lr_scheduler.decay_factor
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

            if self.global_step % self.cfg.system.tqdm_iters == 0:
                mse = float(np.mean(logs["srgb/MSE"]))
                psnr = float(np.mean(logs["srgb/PSNR"]))
                lin_mse = float(np.mean(logs["lin/MSE"]))
                lin_psnr = float(np.mean(logs["lin/PSNR"]))
                pbr_off_mse = float(np.mean(logs["lin/pbr/off_MSE"]))
                pbr_emo_mse = float(np.mean(logs["lin/pbr/emo_MSE"]))
                logs["srgb/MSE"] = []
                logs["srgb/PSNR"] = []
                logs["lin/MSE"] = []
                logs["lin/PSNR"] = []
                logs["lin/pbr/off_MSE"] = []
                logs["lin/pbr/emo_MSE"] = []
                if not self.cfg.system.debug:
                    pbar.set_description(
                        f"Iter {self.global_step:05d}"
                        + " (s)"
                        + f" psnr = {psnr:.2f}"
                        # + f" mse = {mse:.6f}"
                        + " (l)"
                        + f" psnr = {lin_psnr:.2f}"
                        # + f" mse = {lin_mse:.6f}"
                        + " (p)"
                        + f" env_mse = {pbr_off_mse:.6f}"
                        + f" em_mse = {pbr_emo_mse:.6f}"
                    )
                wandb.log(
                    {
                        "train/metric/srgb/MSE": mse,
                        "train/metric/srgb/PSNR": psnr,
                        "train/metric/lin/MSE": lin_mse,
                        "train/metric/lin/PSNR": lin_psnr,
                        "train/metric/lin/pbr/off_MSE": pbr_off_mse,
                        "train/metric/lin/pbr/emo_MSE": pbr_emo_mse,
                    },
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
                            "num_voxels": self.renderer.num_voxels,
                            "params": self.renderer.state_dict(),
                        },
                        "trainer": {
                            "global_step": self.global_step,
                            "uncert_batch_st": self.sampler.uncert_batch_st,
                            "cert_batch_st": self.sampler.cert_batch_st,
                            "uncert_data_idxs": self.sampler.uncert_data_idxs,
                            "cert_data_idxs": self.sampler.cert_data_idxs,
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
            "lin/MSE": [],
            "lin/PSNR": [],
            "lin/SSIM": [],
            "lin/LPIPS_ALEX": [],
        }

        if self.test_dataset.phase == "test_nv":
            metrics.update(
                {
                    **{f"lin/MSE_EXR_{mode}": [] for mode in ["off", "on"]},
                    "srgb/MSE": [],
                    "srgb/PSNR": [],
                    "srgb/SSIM": [],
                    "srgb/LPIPS_ALEX": [],
                    # just for cumulating the I and U to reduce the memory usage
                    "etc/IoU_I": [],
                    "etc/IoU_U": [],
                }
            )

        targets = []
        renders: Dict[str, List[torch.Tensor]] = {}

        self.renderer.eval()
        for i in tqdm_safe(img_idxes, desc="eval", leave=False):
            data = self.test_dataset[i]
            if self.test_dataset.phase != "test_nv":
                self.finetune_radiance(data)

            rgbs = data["rgbs"]
            batch = {
                "rays_o": data["rays_o"].to(self.device),
                "rays_d": data["rays_d"].to(self.device),
                "viewdirs": data["viewdirs"].to(self.device),
                "em_modes": data["em_modes"][..., 0].to(self.device),
                "pos_rt": data["poses"][:3, :3].to(self.device),
                "render_pbr": self.render_pbr,
                "chunk_sz": self.chunk_sz,
            }

            if self.test_dataset.phase != "test_nv":
                batch["em_modes"] = 1

            results = {}
            for chunk_idx in tqdm_safe(
                torch.arange(len(data["rgbs"])).split(self.eval_bs),
                desc="pixel chunks",
                leave=False,
            ):
                chunk = {
                    "rays_o": batch["rays_o"][chunk_idx],
                    "rays_d": batch["rays_d"][chunk_idx],
                    "viewdirs": batch["viewdirs"][chunk_idx],
                    "em_modes": batch["em_modes"],
                    "pos_rt": batch["pos_rt"],
                    "render_pbr": self.render_pbr,
                    "chunk_sz": self.chunk_sz,
                }
                for k, v in self.renderer(**chunk).items():
                    if k not in results:
                        results[k] = []
                    results[k].append(v)

            for k, v in list(results.items()):
                results[k] = (
                    torch.cat(v, dim=0).reshape(height, width, -1).squeeze(-1).cpu()
                )

            masks = torch.any(results["lin/emit"] > self.k_val, dim=-1, keepdim=True)
            results["lin/emit"] = results["lin/emit"] * masks
            masks.squeeze_(-1)

            # background
            for k, v in list(results.items()):
                if k != "etc/white_bg":
                    wbg = results["etc/white_bg"] * self.white_bg
                    if v.dim() == 3:
                        wbg = wbg.unsqueeze(-1)

                    if k.startswith("lin/"):
                        results[f"{k}_gamma"] = apply_gamma_curve(
                            (v + wbg).clamp(min=0.0, max=1.0)
                        )
                        results[k] = (v + wbg).clamp(min=0.0)
                    else:
                        results[k] = (v + wbg).clamp(min=0.0, max=1.0)
                else:
                    results[k] = v.clamp(min=0.0, max=1.0)

            rgbs = rgbs.reshape(height, width, 3)
            pred = results["srgb/rgb"]
            lin_pred_org = results["lin/rgb"]
            lin_pred_gamma = results["lin/rgb_gamma"]

            if self.test_dataset.phase == "test_nv":
                hdrs = data["hdrs"]
                hdrs = hdrs.reshape(height, width, 3)

                for mode in ["off", "on"]:
                    if LightDict[mode] == batch["em_modes"].item():  # type: ignore
                        metrics[f"lin/MSE_EXR_{mode}"].append(
                            F.mse_loss(lin_pred_org, hdrs).item()
                        )
                    else:
                        metrics[f"lin/MSE_EXR_{mode}"].append(None)  # type: ignore

            if self.test_dataset.phase == "test_nv":
                areas = data["areas"]
                areas = areas.reshape(height, width)

                iou = IoU(masks, areas)
                metrics["etc/IoU_I"].append(iou[1])
                metrics["etc/IoU_U"].append(iou[2])

                metrics["srgb/MSE"].append(F.mse_loss(pred, rgbs).item())
                metrics["srgb/PSNR"].append(loss2psnr(metrics["srgb/MSE"][-1]))
                metrics["srgb/SSIM"].append(rgb_ssim(pred, rgbs, 1))
                metrics["srgb/LPIPS_ALEX"].append(
                    rgb_lpips(rgbs.numpy(), pred.numpy(), "alex", self.device)
                )

            metrics["lin/MSE"].append(F.mse_loss(lin_pred_gamma, rgbs).item())
            metrics["lin/PSNR"].append(loss2psnr(metrics["lin/MSE"][-1]))
            metrics["lin/SSIM"].append(rgb_ssim(lin_pred_gamma, rgbs, 1))
            metrics["lin/LPIPS_ALEX"].append(
                rgb_lpips(rgbs.numpy(), lin_pred_gamma.numpy(), "alex", self.device)
            )

            targets.append((rgbs.numpy() * 255).astype("uint8"))
            for k, v in results.items():
                if k not in renders:
                    renders[k] = []
                renders[k].append(
                    (v.clamp(min=0.0, max=1.0).numpy() * 255).astype("uint8")
                )

        # scene results
        scn_metrics: Dict[str, float] = {}
        scn_results: Dict[str, torch.Tensor] = {}

        if self.test_dataset.phase == "test_nv":
            # compute iou
            scn_metrics["etc/IoU"] = np.sum(metrics["etc/IoU_I"]) / max(
                1, np.sum(metrics["etc/IoU_U"])
            )
            del metrics["etc/IoU_I"]  # delete to avoid redundancy
            del metrics["etc/IoU_U"]

            # envmap extraction
            envmap = (
                self.renderer.render_envmap(self.envmap_height, self.envmap_width)
                .clamp(min=0.0, max=1.0)
                .cpu()
            )
            scn_results["etc/envmap_gamma"] = apply_gamma_curve(envmap)
            scn_results["etc/envmap"] = envmap
            for k, v in scn_results.items():
                scn_results[k] = (v.numpy() * 255).astype("uint8")

            # mesh extraction
            vertices, triangles = self.renderer.extract_geometry(
                batch_size=self.eval_bs
            )
            scale_mat = self.test_dataset.scale_mat.cpu().numpy()
            vertices = vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]
            mesh = trimesh.Trimesh(vertices, triangles)

            # save mesh
            mesh.export(os.path.join(mesh_dir, "mesh.ply"))
            for k, v in scn_results.items():
                _k = k.split("/")
                result_dir = os.path.join(image_dir, *_k[:-1])
                os.makedirs(result_dir, exist_ok=True)
                imageio.imwrite(os.path.join(result_dir, f"{_k[-1]}.png"), v)

        # save images
        for k, v in renders.items():
            result_dir = os.path.join(image_dir, *k.split("/"))
            os.makedirs(result_dir, exist_ok=True)
            for i, _v in enumerate(v):
                imageio.imwrite(os.path.join(result_dir, f"{i:03d}.png"), _v)

        # save videos
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

        # save metrics
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
        for k, v in scn_results.items():
            logs[prefix + "image/" + k] = wandb.Image(v)
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

    @property
    def s_val(self) -> float:
        return (
            min(self.global_step, self.step_end) - self.step_start
        ) / self.s_inv_ratio + self.s_start

    @property
    def k_val(self) -> float:
        return (
            min(self.global_step, self.prog_end_step) * self.prog_slope
            + self.prog_start
        )

    @torch.no_grad()
    def update_ray_groups(self, k_val: float):
        rich.print("[PREV]")
        self.sampler.print_stats()

        rich.print(f"curr k_val: {k_val}")
        rays_o = self.sampler.uncert_data["rays_o"]
        rays_d = self.sampler.uncert_data["rays_d"]
        viewdirs = self.sampler.uncert_data["viewdirs"]
        emission = torch.zeros_like(rays_o)

        self.renderer.eval()
        for idx in tqdm_safe(
            torch.arange(len(emission), device=self.device).split(self.eval_uncert_bs),
            desc="eval emission",
            leave=False,
        ):
            ro = rays_o[idx].to(self.device)
            rd = rays_d[idx].to(self.device)
            vd = viewdirs[idx].to(self.device)

            batch = {
                "rays_o": ro,
                "rays_d": rd,
                "viewdirs": vd,
            }

            emission[idx] = self.renderer.eval_emit(**batch)

        uncert_masks = torch.max(emission, dim=-1)[0] > k_val
        if emission.numel():
            rich.print("emission(max, min):", emission.max(), emission.min())
        if uncert_masks.sum():
            rich.print(
                "emission_uncert_masks(max, min):",
                emission[uncert_masks].max(),
                emission[uncert_masks].min(),
            )
        if uncert_masks.bitwise_not().sum():
            rich.print(
                "emission_cert_masks(max, min):",
                emission[uncert_masks.bitwise_not()].max(),
                emission[uncert_masks.bitwise_not()].min(),
            )

        self.sampler.filter(uncert_masks)

        self.renderer.train()

        rich.print("[bold cyan][NEW MASK UPDATE][/bold cyan]")
        self.sampler.print_stats()

    def filter_edit_rays(
        self, sampler: RayGroupManager, test_data: Dict[str, torch.Tensor]
    ):
        # cam
        w, h = self.train_dataset.image_size
        f = self.train_dataset.focal_length
        w2c = torch.inverse(test_data["poses"]).to(self.device)
        K = torch.tensor(
            [[-f, 0.0, w / 2.0 - 0.5], [0.0, f, h / 2.0 - 0.5], [0.0, 0.0, 1.0]]
        ).to(self.device, dtype=torch.float32)

        # dilation
        em_masks = (
            rearrange(test_data["em_masks"].view(-1, h, w), "n h w -> h w n")
            .cpu()
            .numpy()
        )
        em_masks = rearrange(
            torch.from_numpy(
                cv2.dilate(  # type: ignore
                    em_masks,
                    np.ones((self.mask_dilation_ks, self.mask_dilation_ks)),
                    iterations=1,
                )
            )
            .to(self.device)
            .view(h, w, -1),
            "h w n -> n h w",
        ).view(-1, 1, h, w)  # num conds, 1, h, w

        rays_o = sampler.uncert_data["rays_o"]
        rays_d = sampler.uncert_data["rays_d"]
        viewdirs = sampler.uncert_data["viewdirs"]
        filter = torch.zeros_like(rays_o[..., 0], dtype=torch.bool)
        modes = torch.ones_like(rays_o[..., 0], dtype=torch.long)
        colors = torch.zeros_like(rays_o[..., :2])
        intensities = torch.zeros_like(rays_o[..., 0])

        self.renderer.eval()
        for idx in tqdm_safe(
            torch.arange(len(filter), device=self.device).split(self.eval_bs),
            desc="filter edit rays",
            leave=False,
        ):
            ro = rays_o[idx].to(self.device)
            rd = rays_d[idx].to(self.device)
            vd = viewdirs[idx].to(self.device)

            batch = {
                "rays_o": ro,
                "rays_d": rd,
                "viewdirs": vd,
            }

            esp = self.renderer.eval_esp(**batch)
            esp = torch.concat([esp, torch.ones_like(esp[..., :1])], dim=-1).T

            # projection
            xyz = w2c @ esp
            cam_coord = xyz[:3] / xyz[-1:]
            xyz = K @ cam_coord
            img_coord = (xyz[:2] / xyz[-1:]).T

            out_bound = (img_coord < 0) | (img_coord > (h - 1)) | (img_coord > (w - 1))
            out_bound = out_bound[..., 0] | out_bound[..., 1]
            in_bound = out_bound.bitwise_not()

            img_coord = img_coord[in_bound]
            img_coord[..., 0] = img_coord[..., 0] / (w - 1) * 2 - 1
            img_coord[..., 1] = img_coord[..., 1] / (h - 1) * 2 - 1
            img_coord = img_coord.view(1, 1, -1, 2).repeat(
                len(em_masks), 1, 1, 1
            )  # num conds, 1, num selected rays, 2

            m = (
                F.grid_sample(em_masks, img_coord, align_corners=True, mode="bilinear")
                > 0
            ).view(len(em_masks), -1)
            filter[idx[in_bound]] = torch.sum(m, dim=0) > 0

            for i in range(len(em_masks)):
                _m = m[i]
                _mode = test_data["em_modes"][i]
                modes[idx[in_bound][_m]] = _mode

                if _mode == LightDict["off"]:
                    intensities[idx[in_bound][_m]] = 0
                if _mode in [LightDict["i_change"], LightDict["ic_change"]]:
                    intensities[idx[in_bound][_m]] = test_data["em_intensities"][i].to(
                        self.device
                    )
                if _mode in [LightDict["c_change"], LightDict["ic_change"]]:
                    colors[idx[in_bound][_m]] = test_data["em_colors"][i][:2].to(
                        self.device
                    )

        sampler.uncert_data["em_modes"] = modes
        sampler.uncert_data["em_colors"] = colors
        sampler.uncert_data["em_intensities"] = intensities

        sampler.cert_data["em_modes"] = torch.zeros_like(
            sampler.cert_data["rays_o"][..., 0], dtype=torch.long
        )
        sampler.cert_data["em_colors"] = torch.zeros_like(
            sampler.cert_data["rays_o"][..., :2]
        )
        sampler.cert_data["em_intensities"] = torch.zeros_like(
            sampler.cert_data["rays_o"][..., 0]
        )
        sampler.keys.extend(["em_colors", "em_intensities"])
        sampler.filter(filter)
        return sampler

    @torch.enable_grad()
    def finetune_radiance(self, test_data: Dict[str, torch.Tensor]):
        # refresh model weights and prepare sampler
        def _init_():
            params = torch.load(self.eval_ckpt, map_location=self.device)
            self.renderer.load_state_dict(params["renderer"]["params"], strict=False)

            sampler = RayGroupManager(
                self.cfg,
                self.train_dataset.all_data,
                deepcopy(self.data_keys),
                self.eval_uncert_bs,
                self.eval_cert_bs,
                uncert_data_idxs=params["trainer"]["uncert_data_idxs"],
                cert_data_idxs=params["trainer"]["cert_data_idxs"],
            )

            sampler = self.filter_edit_rays(sampler, test_data)

            for p in self.renderer.parameters():
                p.requires_grad_(False)
            for p in self.renderer.emo_color.parameters():
                p.requires_grad_(True)
            for p in self.renderer.emo_rgbnet.parameters():
                p.requires_grad_(True)

            optimizer = create_optimizer_or_freeze_model(self.renderer, **self.eval_lrs)

            return sampler, optimizer

        sampler, optimizer = _init_()
        sampler.print_stats()
        # turn on finetune mode
        self.renderer.train(finetune=True)
        pbar = tqdm_safe(range(self.eval_niters), colour="green")

        logs: Dict[str, List] = {"lin/pbr/emo_MSE": []}
        for global_step in pbar:
            batch = sampler.sample()

            optimizer.zero_grad(set_to_none=True)
            results = self.renderer(**batch)

            loss = self.eval_weight_lts * F.mse_loss(
                results["lin/pbr/emo"], results["lin/pbr/emo_hat"]
            )
            loss.backward()
            optimizer.step()

            logs["lin/pbr/emo_MSE"].append(loss.item())

            if self.global_step % self.cfg.system.tqdm_iters == 0:
                pbr_emo_mse = float(np.mean(logs["lin/pbr/emo_MSE"]))
                logs["lin/pbr/emo_MSE"] = []
                if not self.cfg.system.debug:
                    pbar.set_description(
                        f"Iter {global_step:05d}"
                        + " (p)"
                        + f" em_mse = {pbr_emo_mse:.6f}"
                    )

        # return to eval mode
        self.renderer.eval()
