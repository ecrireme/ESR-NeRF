from typing import Dict, List, Tuple

import mcubes
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn import functional as F
from torch_scatter import segment_coo

from app.utils.base.functions import (
    differentiable_grid_sample,
    extract_fields,
    neus_alpha_from_sdf_scatter_grad,
    neus_alpha_from_sdf_scatter_interp,
    render_utils_cuda,
    total_variation,
)
from app.utils.base.module import (
    Alphas2Weights,
    DenseGrid,
    Gaussian3DConv,
    GradientConv,
    MaskCache,
)
from app.utils.pbr.functions import (
    diffuse_scattering,
    diffuse_scattering_fib,
    disney_reflection,
    hsv_to_rgb,
    rgb_to_hsv,
)
from app.utils.pbr.module import (
    BRDFNet,
    EmissionNet,
    RadianceNet,
    SphericalGaussian,
    TonemapNet,
)
from utils2.utils import tqdm_safe


class ESRNeRF(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        near: float,
        far: float,
        xyz_min: torch.Tensor,
        xyz_max: torch.Tensor,
        mask_xyz_min: torch.Tensor,
        mask_xyz_max: torch.Tensor,
        mask_alpha_init: float,
        mask_density: torch.Tensor,
        s_val: float,
        num_voxles: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.system.device

        # dynamic variables
        self.near = near
        self.far = far
        self.xyz_min = xyz_min.to(self.device)
        self.xyz_max = xyz_max.to(self.device)
        self.mask_xyz_min = mask_xyz_min.to(self.device)
        self.mask_xyz_max = mask_xyz_max.to(self.device)
        self.mask_alpha_init = mask_alpha_init
        self.mask_density = mask_density.to(self.device)
        self.s_val = s_val
        self.num_voxels = num_voxles

        # static variables
        self.mask_ks = cfg.app.model.mask_ks
        self.maskcache_thres = cfg.app.model.maskcache_thres
        self.fastcolor_thres = cfg.app.model.fastcolor_thres

        self.stepsize = cfg.app.model.stepsize

        self.color_dim = cfg.app.model.color_dim
        self.rgbnet_width = cfg.app.model.rgbnet_width
        self.rgbnet_depth = cfg.app.model.rgbnet_depth
        self.tonemap_width = cfg.app.model.tonemap_width
        self.tonemap_depth = cfg.app.model.tonemap_depth
        self.brdfnet_width = cfg.app.model.brdfnet_width
        self.brdfnet_depth = cfg.app.model.brdfnet_depth
        self.env_sg = cfg.app.model.env_sg
        self.env_activation = cfg.app.model.env_activation

        self.posbase_pe = cfg.app.model.posbase_pe
        self.viewbase_pe = cfg.app.model.viewbase_pe
        self.colorbase_pe = cfg.app.model.colorbase_pe
        self.grad_feat = torch.tensor(cfg.app.model.grad_feat, device=self.device)

        self.neus_alpha = cfg.app.model.neus_alpha

        self.ray_sampling = cfg.app.model.ray_sampling
        self.num_2ndrays = cfg.app.model.num_2ndrays
        self.num_ltspts = cfg.app.model.num_ltspts
        self.lts_near = cfg.app.model.lts_near

        self.set_grid_resolution(self.num_voxels)

        # init sdf
        self.sdf = DenseGrid(
            channels=1,
            world_size=self.world_size,
            xyz_min=self.xyz_min,
            xyz_max=self.xyz_max,
        )
        x, y, z = np.mgrid[
            -1.0 : 1.0 : self.world_size[0].item() * 1j,
            -1.0 : 1.0 : self.world_size[1].item() * 1j,
            -1.0 : 1.0 : self.world_size[2].item() * 1j,
        ]
        self.sdf.grid.data = (
            torch.from_numpy((x**2 + y**2 + z**2) ** 0.5 - 1)
            .float()[None, None, ...]
            .to(self.device)
        )

        # grad conv to calculate gradient
        self.tv_smooth_conv = GradientConv()

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        self.mask_cache = MaskCache(
            self.mask_xyz_min,
            self.mask_xyz_max,
            self.mask_density,
            self.mask_alpha_init,
            self.maskcache_thres,
            self.mask_ks,
        )
        self.set_nonempty_mask()

        self.posfreq = torch.FloatTensor([(2**i) for i in range(self.posbase_pe)]).to(
            self.device
        )
        self.viewfreq = torch.FloatTensor([(2**i) for i in range(self.viewbase_pe)]).to(
            self.device
        )
        self.colorfreq = torch.FloatTensor(
            [(2**i) for i in range(self.colorbase_pe)]
        ).to(self.device)

        # init color
        self.off_color = DenseGrid(
            channels=self.color_dim,
            world_size=self.world_size,
            xyz_min=self.xyz_min,
            xyz_max=self.xyz_max,
        )

        dim0 = (
            (3 + 3 * self.posbase_pe * 2) + (3 * self.viewbase_pe * 3) + self.color_dim
        )
        dim0 += len(self.grad_feat) * 3
        dim0 += len(self.grad_feat) * 6
        dim0 += 1
        self.off_rgbnet = RadianceNet(dim0, self.rgbnet_width, self.rgbnet_depth)

        self.emo_color = DenseGrid(
            channels=self.color_dim,
            world_size=self.world_size,
            xyz_min=self.xyz_min,
            xyz_max=self.xyz_max,
        )
        self.emo_rgbnet = RadianceNet(dim0, self.rgbnet_width, self.rgbnet_depth)

        dim0 = 3 + 3 * self.colorbase_pe * 2
        self.tonemapper = TonemapNet(dim0, self.tonemap_width, self.tonemap_depth)

        # init BRDF
        self.brdf = DenseGrid(
            channels=self.color_dim,
            world_size=self.world_size,
            xyz_min=self.xyz_min,
            xyz_max=self.xyz_max,
        )
        dim0 = (3 + 3 * self.posbase_pe * 2) + self.color_dim
        dim0 += len(self.grad_feat) * 3
        dim0 += len(self.grad_feat) * 6
        dim0 += 1
        self.emitnet = EmissionNet(dim0, self.brdfnet_width, self.brdfnet_depth)
        self.brdfnet = BRDFNet(dim0, self.brdfnet_width, self.brdfnet_depth, self.brdf)

        match self.ray_sampling.lower():
            case "random" | "rand":
                self.scattering = diffuse_scattering
            case "fib" | "fibo" | "fibonacci":
                self.scattering = diffuse_scattering_fib

        # init envmap
        self.envmap = SphericalGaussian(self.env_sg, self.env_activation)

        if self.neus_alpha == "grad":
            self.neus_alpha_from_sdf_scatter = neus_alpha_from_sdf_scatter_grad
        else:
            self.neus_alpha_from_sdf_scatter = neus_alpha_from_sdf_scatter_interp

        # some useful vars
        self.normal_flipper = torch.tensor([1.0, -1.0, -1.0], device=self.device)

        self.grid_size = self.sdf.grid.size()[-3:]
        self.size_factor_zyx = torch.tensor(
            [self.grid_size[2], self.grid_size[1], self.grid_size[0]],
            device=self.device,
        )
        self.sdf_offset = torch.tensor(
            [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]],
            device=self.device,
        )
        self.sdf_displace = torch.tensor([1.0], device=self.device)

        self.pdra_mode = False

    def train(self, mode=True, finetune=False):
        if mode:
            if not finetune:
                self.forward = self.forward_training
                if hasattr(self, "emit_color"):
                    del self.emit_color
            else:
                self.forward = self.forward_finetune
                self.emit_color = DenseGrid(
                    channels=self.color_dim,
                    world_size=self.world_size,
                    xyz_min=self.xyz_min,
                    xyz_max=self.xyz_max,
                ).to(self.device)
                self.emit_color.load_state_dict(self.emo_color.state_dict())
                for p in self.emit_color.parameters():
                    p.requires_grad_(False)
        else:
            self.forward = self.forward_evaluate
            if not hasattr(self, "emit_color"):
                self.emit_color = self.emo_color
        return super().train(mode)

    @torch.no_grad()
    def forward_finetune(self, **kwargs):
        def light_transport_segment(
            ray_pts: torch.Tensor,
            viewdirs: torch.Tensor,
            em_modes: torch.Tensor,
            em_intensities: torch.Tensor,
            em_colors: torch.Tensor,
        ):
            def __prop(
                xyz_emb: torch.Tensor,
                viewdirs: torch.Tensor,
                viewdirs_rand: torch.Tensor,
                sdf: torch.Tensor,
                all_feat: torch.Tensor,
                all_normal: torch.Tensor,
            ):
                viewdirs_emb = (
                    torch.cat([viewdirs, viewdirs_rand], 0).unsqueeze(-1)
                    * self.viewfreq
                ).flatten(-2)

                rgb_feat = torch.cat(
                    [
                        xyz_emb.repeat([2, 1]),
                        viewdirs_emb,
                        viewdirs_emb.sin(),
                        viewdirs_emb.cos(),
                        sdf[:, None].repeat([2, 1]),
                        all_feat.repeat([2, 1]),
                        all_normal.repeat([2, 1]),
                    ],
                    dim=-1,
                )

                with torch.enable_grad():
                    emo_linear_rgb: torch.Tensor = self.emo_rgbnet(
                        torch.cat(
                            [self.emo_color(ray_pts).repeat([2, 1]), rgb_feat], -1
                        )
                    )

                brdf_feat = torch.cat(
                    [xyz_emb, sdf[:, None], all_feat, all_normal], dim=-1
                )
                basecolor, roughness, metallic = self.brdfnet(
                    torch.cat([self.brdf(ray_pts), brdf_feat], dim=-1)
                )
                emit = self.emitnet(
                    torch.cat([self.emit_color(ray_pts), brdf_feat], dim=-1)
                )

                return emo_linear_rgb, basecolor, roughness, metallic, emit

            ret_dict: Dict[str, torch.Tensor] = {}

            sdf, exp_grad = self.sample_sdf_expgrad(ray_pts)
            sdf = sdf.detach()
            exp_grad = exp_grad.detach()
            normal = F.normalize(exp_grad, dim=-1)

            dirs = self.scattering(normal, self.num_2ndrays + 1)
            viewdirs_rand = -dirs[:, -1]
            dirs = dirs[:, :-1]

            # radiance at points
            all_feat, _, all_normal = self.sample_sdfeat_grad_normal(
                ray_pts, displace=self.grad_feat
            )

            rays_xyz = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
            xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
            xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], -1)

            ret_dict["emo"], basecolor, roughness, metallic, emit = __prop(
                xyz_emb, viewdirs, viewdirs_rand, sdf, all_feat, all_normal
            )

            shape = dirs.shape
            ray_pts = ray_pts.view(-1, 1, 3).expand(shape).flatten(0, 1)
            viewdirs = viewdirs.view(-1, 1, 3).expand(shape).flatten(0, 1)
            viewdirs_rand = viewdirs_rand.view(-1, 1, 3).expand(shape).flatten(0, 1)
            normal = normal.view(-1, 1, 3).expand(shape).flatten(0, 1)
            basecolor = basecolor.view(-1, 1, 3).expand(shape).flatten(0, 1)
            roughness = (
                roughness.view(-1, 1, 1).expand(shape[0], shape[1], 1).flatten(0, 1)
            )
            metallic = (
                metallic.view(-1, 1, 1).expand(shape[0], shape[1], 1).flatten(0, 1)
            )
            dirs = dirs.flatten(0, 1)

            R = disney_reflection(
                basecolor.repeat([2, 1]),
                roughness.repeat([2, 1]),
                metallic.repeat([2, 1]),
                normal.repeat([2, 1]),
                dirs.repeat([2, 1]),
                torch.cat([-viewdirs, -viewdirs_rand], dim=0),
            )

            # get incoming radiance
            rays_o = ray_pts
            viewdirs = dirs

            N = len(rays_o)
            ray_pts, ray_id, _ = self.sample_ray(
                rays_o=rays_o, rays_d=viewdirs, near=self.lts_near
            )

            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]

            sdf, grad = self.sample_sdf_grad(ray_pts)

            dist = self.stepsize * self.voxel_size
            alpha = self.neus_alpha_from_sdf_scatter(
                viewdirs, ray_id, dist, sdf, grad, self.s_val
            )

            mask = alpha > self.fastcolor_thres
            alpha = alpha[mask]
            ray_id = ray_id[mask]
            ray_pts = ray_pts[mask]
            sdf = sdf[mask]

            weights, _ = Alphas2Weights.apply(alpha, ray_id, N)  # type: ignore
            mask = weights > self.fastcolor_thres
            weights = weights[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            sdf = sdf[mask]

            # rgb feature
            all_feat, _, all_normal = self.sample_sdfeat_grad_normal(
                ray_pts, displace=self.grad_feat
            )

            rays_xyz = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
            xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)[ray_id]
            rgb_feat = torch.cat(
                [
                    rays_xyz,
                    xyz_emb.sin(),
                    xyz_emb.cos(),
                    viewdirs_emb,
                    viewdirs_emb.sin(),
                    viewdirs_emb.cos(),
                    sdf[:, None],
                    all_feat,
                    all_normal,
                ],
                dim=-1,
            )

            lin_emo_rgb = self.emo_rgbnet(
                torch.cat([self.emo_color(ray_pts), rgb_feat], -1)
            )

            # Ray marching
            weights = weights.unsqueeze(-1)

            lin_emo_rgb_marched = segment_coo(
                src=(weights * lin_emo_rgb),
                index=ray_id,
                out=torch.zeros([N, 3], device=self.device),
                reduce="sum",
            )

            # em out decomposition
            off_mask = em_modes == 0
            i_mask = (em_modes == 2) | (em_modes == 4)
            c_mask = (em_modes == 3) | (em_modes == 4)

            emit[off_mask] = 0
            emit[i_mask] = emit[i_mask] * em_intensities[i_mask][..., None]
            hsv = rgb_to_hsv(emit[c_mask])
            hsv[..., :-1] = em_colors[c_mask]
            emit[c_mask] = hsv_to_rgb(hsv)

            reflect = (
                (lin_emo_rgb_marched.repeat([2, 1]) * R)
                .view(-1, self.num_2ndrays, 3)
                .mean(-2)
            )
            ret_dict["emo_hat"] = emit.repeat([2, 1]) + reflect

            return ret_dict

        rays_o = kwargs["rays_o"]
        rays_d = kwargs["rays_d"]
        viewdirs = kwargs["viewdirs"]
        em_modes = kwargs["em_modes"]
        em_intensities = kwargs["em_intensities"]
        em_colors = kwargs["em_colors"]

        N = len(rays_o)

        ray_pts, ray_id, _ = self.sample_ray(
            rays_o=rays_o, rays_d=rays_d, near=self.near
        )

        # skip known free space
        mask = self.mask_cache(ray_pts)
        ray_pts = ray_pts[mask]
        ray_id = ray_id[mask]

        sdf, grad = self.sample_sdf_grad(ray_pts)

        dist = self.stepsize * self.voxel_size
        alpha = self.neus_alpha_from_sdf_scatter(
            viewdirs, ray_id, dist, sdf, grad, self.s_val
        )

        # app mask 0
        mask = alpha > self.fastcolor_thres
        alpha = alpha[mask]
        ray_id = ray_id[mask]
        ray_pts = ray_pts[mask]

        # app mask 1
        weights, _ = Alphas2Weights.apply(alpha, ray_id, N)  # type: ignore
        mask = weights > self.fastcolor_thres
        ray_pts = ray_pts[mask]
        ray_id = ray_id[mask]

        # rgb feature
        idx = np.random.choice(
            len(ray_pts), min(self.num_ltspts, len(ray_pts)), replace=False
        )
        lts = light_transport_segment(
            ray_pts[idx],
            viewdirs[ray_id][idx],
            em_modes[ray_id][idx],
            em_intensities[ray_id][idx],
            em_colors[ray_id][idx],
        )

        return {
            "lin/pbr/emo": lts["emo"],
            "lin/pbr/emo_hat": lts["emo_hat"],
        }

    def forward_training(self, **kwargs):
        def light_transport_segment(
            ray_pts: torch.Tensor,
            viewdirs: torch.Tensor,
            normal: torch.Tensor,
            sdf: torch.Tensor,
            basecolor: torch.Tensor,
            roughness: torch.Tensor,
            metallic: torch.Tensor,
            emission: torch.Tensor,
            uncert_masks: torch.Tensor,
        ):
            def __radiance(
                xyz_emb: torch.Tensor,
                viewdirs: torch.Tensor,
                viewdirs_rand: torch.Tensor,
                sdf: torch.Tensor,
                all_feat: torch.Tensor,
                all_normal: torch.Tensor,
            ):
                viewdirs_emb = (
                    torch.cat([viewdirs, viewdirs_rand], 0).unsqueeze(-1)
                    * self.viewfreq
                ).flatten(-2)

                rgb_feat = torch.cat(
                    [
                        xyz_emb.repeat([2, 1]),
                        viewdirs_emb,
                        viewdirs_emb.sin(),
                        viewdirs_emb.cos(),
                        sdf[:, None].repeat([2, 1]),
                        all_feat.repeat([2, 1]),
                        all_normal.repeat([2, 1]),
                    ],
                    dim=-1,
                )

                off_linear_rgb: torch.Tensor = self.off_rgbnet(
                    torch.cat([self.off_color(ray_pts).repeat([2, 1]), rgb_feat], -1)
                )
                emo_linear_rgb: torch.Tensor = self.emo_rgbnet(
                    torch.cat([self.emo_color(ray_pts).repeat([2, 1]), rgb_feat], -1)
                )

                return off_linear_rgb, emo_linear_rgb

            ret_dict: Dict[str, torch.Tensor] = {}
            dirs = self.scattering(normal, self.num_2ndrays + 1)
            viewdirs_rand = -dirs[:, -1]
            dirs = dirs[:, :-1]

            # radiance at points
            all_feat, _, all_normal = self.sample_sdfeat_grad_normal(
                ray_pts, displace=self.grad_feat
            )

            rays_xyz = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
            xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
            xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], -1)

            ret_dict["off"], ret_dict["emo"] = __radiance(
                xyz_emb, viewdirs, viewdirs_rand, sdf, all_feat, all_normal
            )

            shape = dirs.shape
            ray_pts = ray_pts.view(-1, 1, 3).expand(shape).flatten(0, 1)
            viewdirs = viewdirs.view(-1, 1, 3).expand(shape).flatten(0, 1)
            viewdirs_rand = viewdirs_rand.view(-1, 1, 3).expand(shape).flatten(0, 1)
            normal = normal.view(-1, 1, 3).expand(shape).flatten(0, 1)
            basecolor = basecolor.view(-1, 1, 3).expand(shape).flatten(0, 1)
            roughness = (
                roughness.view(-1, 1, 1).expand(shape[0], shape[1], 1).flatten(0, 1)
            )
            metallic = (
                metallic.view(-1, 1, 1).expand(shape[0], shape[1], 1).flatten(0, 1)
            )
            dirs = dirs.flatten(0, 1)

            R = disney_reflection(
                basecolor.repeat([2, 1]),
                roughness.repeat([2, 1]),
                metallic.repeat([2, 1]),
                normal.repeat([2, 1]),
                dirs.repeat([2, 1]),
                torch.cat([-viewdirs, -viewdirs_rand], dim=0),
            )

            # get incoming radiance
            rays_o = ray_pts
            viewdirs = dirs

            N = len(rays_o)
            ray_pts, ray_id, _ = self.sample_ray(
                rays_o=rays_o, rays_d=viewdirs, near=self.lts_near
            )

            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]

            sdf, grad = self.sample_sdf_grad(ray_pts)

            dist = self.stepsize * self.voxel_size
            alpha = self.neus_alpha_from_sdf_scatter(
                viewdirs, ray_id, dist, sdf, grad, self.s_val
            )

            mask = alpha > self.fastcolor_thres
            alpha = alpha[mask]
            ray_id = ray_id[mask]
            ray_pts = ray_pts[mask]
            sdf = sdf[mask]

            weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)  # type: ignore
            mask = weights > self.fastcolor_thres
            weights = weights[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            sdf = sdf[mask]

            # rgb feature
            all_feat, _, all_normal = self.sample_sdfeat_grad_normal(
                ray_pts, displace=self.grad_feat
            )

            rays_xyz = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
            xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)[ray_id]
            rgb_feat = torch.cat(
                [
                    rays_xyz,
                    xyz_emb.sin(),
                    xyz_emb.cos(),
                    viewdirs_emb,
                    viewdirs_emb.sin(),
                    viewdirs_emb.cos(),
                    sdf[:, None],
                    all_feat,
                    all_normal,
                ],
                dim=-1,
            )

            lin_off_rgb = self.off_rgbnet(
                torch.cat([self.off_color(ray_pts), rgb_feat], -1)
            )
            lin_emo_rgb = self.emo_rgbnet(
                torch.cat([self.emo_color(ray_pts), rgb_feat], -1)
            )

            # Ray marching
            weights = weights.unsqueeze(-1)
            lin_off_rgb_marched = segment_coo(
                src=(weights * lin_off_rgb),
                index=ray_id,
                out=torch.zeros([N, 3], device=self.device),
                reduce="sum",
            )

            lin_emo_rgb_marched = segment_coo(
                src=(weights * lin_emo_rgb),
                index=ray_id,
                out=torch.zeros([N, 3], device=self.device),
                reduce="sum",
            )

            # env out decomposition
            envmap = self.envmap(dirs) * alphainv_last.unsqueeze(-1)

            ret_dict["off_hat"] = (
                ((lin_off_rgb_marched + envmap).repeat([2, 1]) * R)
                .view(-1, self.num_2ndrays, 3)
                .mean(-2)
            )

            # em out decomposition
            reflect = (
                (lin_emo_rgb_marched.repeat([2, 1]) * R)
                .view(-1, self.num_2ndrays, 3)
                .mean(-2)
            )
            if self.pdra_mode:
                umasks = uncert_masks.repeat([2])

                ret_dict["emo_hat"] = torch.empty_like(ret_dict["emo"])
                ret_dict["emo_hat"][umasks] = (
                    emission.repeat([2, 1])[umasks] + reflect.detach()[umasks]
                )
                ret_dict["emo_hat"][~umasks] = reflect[~umasks]
            else:
                ret_dict["emo_hat"] = emission.repeat([2, 1]) + reflect

            return ret_dict

        rays_o = kwargs["rays_o"]
        rays_d = kwargs["rays_d"]
        viewdirs = kwargs["viewdirs"]
        em_modes = kwargs["em_modes"]
        uncert_masks = kwargs["uncert_masks"]
        self.s_val = kwargs["s_val"]
        normal_eps = kwargs["normal_eps"]
        emit_eps = kwargs["emit_eps"]

        N = len(rays_o)

        ray_pts, ray_id, _ = self.sample_ray(
            rays_o=rays_o, rays_d=rays_d, near=self.near
        )

        # skip known free space
        mask = self.mask_cache(ray_pts)
        ray_pts = ray_pts[mask]
        ray_id = ray_id[mask]

        sdf, exp_grad, grad, _ = self.sample_sdf_expgrad_grad_normal(ray_pts)

        dist = self.stepsize * self.voxel_size
        alpha = self.neus_alpha_from_sdf_scatter(
            viewdirs, ray_id, dist, sdf, grad, self.s_val
        )

        # app mask 0
        mask = alpha > self.fastcolor_thres
        alpha = alpha[mask]
        ray_id = ray_id[mask]
        ray_pts = ray_pts[mask]
        exp_grad = exp_grad[mask]
        sdf = sdf[mask]

        # app mask 1
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)  # type: ignore
        mask = weights > self.fastcolor_thres
        weights = weights[mask]
        ray_pts = ray_pts[mask]
        ray_id = ray_id[mask]
        exp_grad = exp_grad[mask]
        sdf = sdf[mask]

        # rgb feature
        on_mask = em_modes[ray_id] == 1

        all_feat, _, all_normal = self.sample_sdfeat_grad_normal(
            ray_pts, displace=self.grad_feat
        )

        self.gradient = self.neus_sdf_gradient()

        rays_xyz = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
        xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], dim=-1)
        viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
        rgb_feat = torch.cat(
            [
                xyz_emb,
                viewdirs_emb[ray_id],
                viewdirs_emb.sin()[ray_id],
                viewdirs_emb.cos()[ray_id],
                sdf[:, None],
                all_feat,
                all_normal,
            ],
            dim=-1,
        )

        lin_rgb = torch.zeros_like(ray_pts)
        lin_rgb[on_mask] = self.emo_rgbnet(
            torch.cat([self.emo_color(ray_pts[on_mask]), rgb_feat[on_mask]], dim=-1)
        )
        lin_rgb = lin_rgb + self.off_rgbnet(
            torch.cat([self.off_color(ray_pts), rgb_feat], -1)
        )

        rgb = self.apply_tonemapper(lin_rgb)

        brdf_feat = torch.cat([xyz_emb, sdf[:, None], all_feat, all_normal], dim=-1)
        basecolor, roughness, metallic = self.brdfnet(
            torch.cat([self.brdf(ray_pts), brdf_feat], dim=-1)
        )
        emit = self.emitnet(torch.cat([self.emo_color(ray_pts), brdf_feat], dim=-1))

        # Ray marching
        weights_ = weights.unsqueeze(-1)
        rgb_marched = segment_coo(
            src=(weights_ * rgb),
            index=ray_id,
            out=torch.zeros([N, 3], device=self.device),
            reduce="sum",
        )

        lin_rgb_marched = segment_coo(
            src=(weights_ * lin_rgb),
            index=ray_id,
            out=torch.zeros([N, 3], device=self.device),
            reduce="sum",
        )

        lin_emit_marched = segment_coo(
            src=(weights_ * emit),
            index=ray_id,
            out=torch.zeros([N, 3], device=self.device),
            reduce="sum",
        )

        normal = F.normalize(exp_grad.detach(), dim=-1)

        idx = np.random.choice(
            len(ray_pts), min(self.num_ltspts, len(ray_pts)), replace=False
        )
        lts = light_transport_segment(
            ray_pts[idx],
            viewdirs[ray_id][idx],
            normal[idx],
            sdf[idx],
            basecolor[idx],
            roughness[idx],
            metallic[idx],
            emit[idx],
            uncert_masks[ray_id][idx],
        )

        # grad eps
        _, exp_grad_eps = self.sample_sdf_expgrad(
            ray_pts + torch.randn_like(ray_pts) * normal_eps
        )

        # emit_eps
        ray_pts = ray_pts + torch.randn_like(ray_pts) * emit_eps
        rays_xyz = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
        xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], dim=-1)

        # sdf, _, _, _ = self.sample_sdf_expgrad_grad_normal(ray_pts)
        sdf, _ = self.sample_sdf_grad(ray_pts)
        all_feat, _, all_normal = self.sample_sdfeat_grad_normal(
            ray_pts, displace=self.grad_feat
        )
        brdf_feat = torch.cat([xyz_emb, sdf[:, None], all_feat, all_normal], dim=-1)

        emit_eps = self.emitnet(
            torch.cat([self.emo_color(ray_pts), brdf_feat], dim=-1)
        )  #
        basecolor_eps, roughness_eps, metallic_eps = self.brdfnet(
            torch.cat([self.brdf(ray_pts), brdf_feat], dim=-1)
        )

        return {
            "etc/alphainv_cum": alphainv_last,
            "etc/white_bg": alphainv_last[..., None],
            "srgb/rgb": rgb_marched,
            "lin/rgb": lin_rgb_marched,
            "lin/pbr/off": lts["off"],
            "lin/pbr/off_hat": lts["off_hat"],
            "lin/pbr/emo": lts["emo"],
            "lin/pbr/emo_hat": lts["emo_hat"],
            "etc/emit_uncert": lin_emit_marched[uncert_masks],
            "etc/emit_cert": lin_emit_marched[~uncert_masks],
            "etc/normal": exp_grad,
            "etc/normal_eps": exp_grad_eps,
            "etc/emit": emit,
            "etc/emit_eps": emit_eps,
            "etc/brdf": torch.concat([basecolor, roughness, metallic], dim=-1),
            "etc/brdf_eps": torch.concat(
                [basecolor_eps, roughness_eps, metallic_eps], dim=-1
            ),
        }

    def forward_evaluate(self, **kwargs):
        def light_transport_segment(
            ray_pts: torch.Tensor,
            viewdirs: torch.Tensor,
            normal: torch.Tensor,
            sdf: torch.Tensor,
            basecolor: torch.Tensor,
            roughness: torch.Tensor,
            metallic: torch.Tensor,
            emit: torch.Tensor,
        ):
            ret_dict: Dict[str, torch.Tensor] = {}
            # physically based rendering
            if len(ray_pts) == 0:
                ret_dict["lin/env_dir"] = ray_pts  # 0 dim tensor
                ret_dict["lin/env_indir"] = ray_pts
                ret_dict["lin/env_effects"] = ray_pts
                ret_dict["lin/emit_(in)dir"] = ray_pts
                ret_dict["lin/emit_effects"] = ray_pts
                return ret_dict
            else:
                dirs = self.scattering(normal, self.num_2ndrays)

                shape = dirs.shape
                ray_pts = ray_pts.view(-1, 1, 3).expand(shape).flatten(0, 1)
                viewdirs = viewdirs.view(-1, 1, 3).expand(shape).flatten(0, 1)
                normal = normal.view(-1, 1, 3).expand(shape).flatten(0, 1)
                basecolor = basecolor.view(-1, 1, 3).expand(shape).flatten(0, 1)
                roughness = (
                    roughness.view(-1, 1, 1).expand(shape[0], shape[1], 1).flatten(0, 1)
                )
                metallic = (
                    metallic.view(-1, 1, 1).expand(shape[0], shape[1], 1).flatten(0, 1)
                )
                dirs = dirs.flatten(0, 1)

                R = disney_reflection(
                    basecolor, roughness, metallic, normal, dirs, -viewdirs
                )

                # get incoming radiance
                rays_o = ray_pts
                viewdirs = dirs

                N = len(rays_o)
                ray_pts, ray_id, _ = self.sample_ray(
                    rays_o=rays_o, rays_d=viewdirs, near=self.lts_near
                )

                mask = self.mask_cache(ray_pts)
                ray_pts = ray_pts[mask]
                ray_id = ray_id[mask]

                sdf, grad = self.sample_sdf_grad(ray_pts)

                dist = self.stepsize * self.voxel_size
                alpha = self.neus_alpha_from_sdf_scatter(
                    viewdirs, ray_id, dist, sdf, grad, self.s_val
                )

                mask = alpha > self.fastcolor_thres
                alpha = alpha[mask]
                ray_id = ray_id[mask]
                ray_pts = ray_pts[mask]
                sdf = sdf[mask]

                if alpha.dim() != 1:
                    rgb_marched = torch.zeros_like(rays_o)
                    ret_dict["lin/env_dir"] = rgb_marched
                    ret_dict["lin/env_indir"] = rgb_marched
                    ret_dict["lin/env_effects"] = rgb_marched
                    ret_dict["lin/emit_(in)dir"] = rgb_marched
                    ret_dict["lin/emit_effects"] = rgb_marched
                    return ret_dict

                weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)  # type: ignore
                mask = weights > self.fastcolor_thres
                weights = weights[mask]
                ray_pts = ray_pts[mask]
                ray_id = ray_id[mask]
                sdf = sdf[mask]

                # rgb feature
                all_feat, _, all_normal = self.sample_sdfeat_grad_normal(
                    ray_pts, displace=self.grad_feat
                )

                rays_xyz = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
                xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
                viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)[
                    ray_id
                ]
                rgb_feat = torch.cat(
                    [
                        rays_xyz,
                        xyz_emb.sin(),
                        xyz_emb.cos(),
                        viewdirs_emb,
                        viewdirs_emb.sin(),
                        viewdirs_emb.cos(),
                        sdf[:, None],
                        all_feat,
                        all_normal,
                    ],
                    dim=-1,
                )

                lin_off_rgb = self.off_rgbnet(
                    torch.cat([self.off_color(ray_pts), rgb_feat], -1)
                )
                lin_emo_rgb = self.emo_rgbnet(
                    torch.cat([self.emo_color(ray_pts), rgb_feat], -1)
                )

                # Ray marching
                weights = weights.unsqueeze(-1)
                lin_off_rgb_marched = segment_coo(
                    src=(weights * lin_off_rgb),
                    index=ray_id,
                    out=torch.zeros([N, 3], device=self.device),
                    reduce="sum",
                )

                lin_emo_rgb_marched = segment_coo(
                    src=(weights * lin_emo_rgb),
                    index=ray_id,
                    out=torch.zeros([N, 3], device=self.device),
                    reduce="sum",
                )

                # env out decomposition
                envmap = self.envmap(dirs) * alphainv_last.unsqueeze(-1)

                ret_dict["lin/env_dir"] = (
                    (envmap * R).view(-1, self.num_2ndrays, 3).mean(-2)
                )
                ret_dict["lin/env_indir"] = (
                    (lin_off_rgb_marched * R).view(-1, self.num_2ndrays, 3).mean(-2)
                )
                ret_dict["lin/env_effects"] = (
                    ret_dict["lin/env_dir"] + ret_dict["lin/env_indir"]
                )

                # em out decomposition
                ret_dict["lin/emit_(in)dir"] = (
                    (lin_emo_rgb_marched * R).view(-1, self.num_2ndrays, 3).mean(-2)
                )
                ret_dict["lin/emit_effects"] = emit + ret_dict["lin/emit_(in)dir"]
                return ret_dict

        rays_o = kwargs["rays_o"]
        rays_d = kwargs["rays_d"]
        viewdirs = kwargs["viewdirs"]
        em_modes = kwargs["em_modes"]
        pos_rt = kwargs["pos_rt"]
        render_pbr = kwargs["render_pbr"]
        chunk_sz = kwargs["chunk_sz"]

        N = len(rays_o)

        ray_pts, ray_id, step_id = self.sample_ray(
            rays_o=rays_o, rays_d=rays_d, near=self.near
        )

        # skip known free space
        mask = self.mask_cache(ray_pts)
        ray_pts = ray_pts[mask]
        ray_id = ray_id[mask]
        step_id = step_id[mask]

        sdf, exp_grad, grad, _ = self.sample_sdf_expgrad_grad_normal(ray_pts)

        dist = self.stepsize * self.voxel_size
        alpha = self.neus_alpha_from_sdf_scatter(
            viewdirs, ray_id, dist, sdf, grad, self.s_val
        )

        # app mask 0
        mask = alpha > self.fastcolor_thres
        alpha = alpha[mask].squeeze()
        ray_id = ray_id[mask].squeeze()
        ray_pts = ray_pts[mask].squeeze()
        step_id = step_id[mask].squeeze()
        grad = grad[mask].squeeze()
        exp_grad = exp_grad[mask]
        sdf = sdf[mask].squeeze()

        # app mask 1
        if alpha.dim() != 1:
            rgb_marched = torch.zeros_like(rays_o)
            depth = rgb_marched[..., 0]
            disp = 1 / (depth + self.far)
            if render_pbr:
                lts = light_transport_segment(
                    rgb_marched[:0],  # dummy zero-dim tensor
                    rgb_marched[:0],
                    rgb_marched[:0],
                    rgb_marched[:0],
                    rgb_marched[:0],
                    rgb_marched[:0],
                    rgb_marched[:0],
                    rgb_marched[:0],
                )
                for k, v in lts.items():
                    lts[k] = rgb_marched
            else:
                lts = {}
            return {
                "etc/depth": depth,
                "etc/disp": disp,
                "etc/normal": rgb_marched,
                "etc/white_bg": torch.ones_like(rgb_marched[..., :1]),
                "srgb/off_rgb": rgb_marched,
                "lin/off_rgb": rgb_marched,
                "srgb/on_rgb": rgb_marched,
                "lin/on_rgb": rgb_marched,
                "srgb/emo_rgb": rgb_marched,
                "lin/emo_rgb": rgb_marched,
                "srgb/rgb": rgb_marched,
                "lin/rgb": rgb_marched,
                "lin/emit": rgb_marched,
                "lin/basecolor": rgb_marched,
                "lin/roughness": depth,
                "lin/metallic": depth,
                **lts,
            }

        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)  # type: ignore
        mask = weights > self.fastcolor_thres
        weights = weights[mask]
        alpha = alpha[mask]
        ray_pts = ray_pts[mask]
        ray_id = ray_id[mask]
        step_id = step_id[mask]
        grad = grad[mask]
        exp_grad = exp_grad[mask]
        sdf = sdf[mask]

        # rgb feature
        all_feat, _, all_normal = self.sample_sdfeat_grad_normal(
            ray_pts, displace=self.grad_feat
        )

        rays_xyz = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
        xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], dim=-1)
        viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
        rgb_feat = torch.cat(
            [
                xyz_emb,
                viewdirs_emb[ray_id],
                viewdirs_emb.sin()[ray_id],
                viewdirs_emb.cos()[ray_id],
                sdf[:, None],
                all_feat,
                all_normal,
            ],
            dim=-1,
        )

        lin_off_rgb = self.off_rgbnet(
            torch.cat([self.off_color(ray_pts), rgb_feat], -1)
        )
        off_rgb = self.apply_tonemapper(lin_off_rgb)
        lin_emo_rgb = self.emo_rgbnet(
            torch.cat([self.emo_color(ray_pts), rgb_feat], -1)
        )
        emo_rgb = self.apply_tonemapper(lin_emo_rgb)
        lin_on_rgb = lin_off_rgb + lin_emo_rgb
        on_rgb = self.apply_tonemapper(lin_on_rgb)

        brdf_feat = torch.cat(
            [
                xyz_emb,
                sdf[:, None],
                all_feat,
                all_normal,
            ],
            dim=-1,
        )

        basecolor, roughness, metallic = self.brdfnet(
            torch.cat([self.brdf(ray_pts), brdf_feat], dim=-1)
        )
        emit = self.emitnet(torch.cat([self.emit_color(ray_pts), brdf_feat], dim=-1))

        # Ray marching
        weights_ = weights.unsqueeze(-1)
        off_rgb_marched = segment_coo(
            src=(weights_ * off_rgb),
            index=ray_id,
            out=torch.zeros([N, 3], device=self.device),
            reduce="sum",
        )

        lin_off_rgb_marched = segment_coo(
            src=(weights_ * lin_off_rgb),
            index=ray_id,
            out=torch.zeros([N, 3], device=self.device),
            reduce="sum",
        )

        on_rgb_marched = segment_coo(
            src=(weights_ * on_rgb),
            index=ray_id,
            out=torch.zeros([N, 3], device=self.device),
            reduce="sum",
        )

        lin_on_rgb_marched = segment_coo(
            src=(weights_ * lin_on_rgb),
            index=ray_id,
            out=torch.zeros([N, 3], device=self.device),
            reduce="sum",
        )

        emo_rgb_marched = segment_coo(
            src=(weights_ * emo_rgb),
            index=ray_id,
            out=torch.zeros([N, 3], device=self.device),
            reduce="sum",
        )

        lin_emo_rgb_marched = segment_coo(
            src=(weights_ * lin_emo_rgb),
            index=ray_id,
            out=torch.zeros([N, 3], device=self.device),
            reduce="sum",
        )

        lin_basecolor = segment_coo(
            src=(weights_ * basecolor),
            index=ray_id,
            out=torch.zeros([N, 3], device=self.device),
            reduce="sum",
        )
        lin_roughness = segment_coo(
            src=(weights * roughness.squeeze(-1)),
            index=ray_id,
            out=torch.zeros([N], device=self.device),
            reduce="sum",
        )
        lin_metallic = segment_coo(
            src=(weights * metallic.squeeze(-1)),
            index=ray_id,
            out=torch.zeros([N], device=self.device),
            reduce="sum",
        )

        lin_emit_marched = segment_coo(
            src=(weights_ * emit),
            index=ray_id,
            out=torch.zeros([N, 3], device=self.device),
            reduce="sum",
        )

        normal = F.normalize(grad, dim=-1)
        normal = normal @ pos_rt
        normal = normal * self.normal_flipper
        normal = (normal + 1.0) / 2.0
        normal_marched = segment_coo(
            src=(weights_ * normal),
            index=ray_id,
            out=torch.zeros([N, 3], device=self.device),
            reduce="sum",
        )

        depth = segment_coo(
            src=(weights * step_id * dist),
            index=ray_id,
            out=torch.zeros([N], device=self.device),
            reduce="sum",
        )

        disp = 1 / (depth + alphainv_last * self.far)

        lts: Dict[str, torch.Tensor] = {}
        if render_pbr:
            viewdirs = viewdirs[ray_id]
            normal = F.normalize(exp_grad.detach(), dim=-1)
            _lts: Dict[str, List[torch.Tensor]] = {}
            for idx in tqdm_safe(
                torch.arange(len(ray_pts), device=self.device).split(chunk_sz),
                desc="LTS chunks",
                leave=False,
            ):
                ret = light_transport_segment(
                    ray_pts[idx],
                    viewdirs[idx],
                    normal[idx],
                    sdf[idx],
                    basecolor[idx],
                    roughness[idx],
                    metallic[idx],
                    emit[idx],
                )
                for k, v in ret.items():
                    if k not in _lts:
                        _lts[k] = []
                    _lts[k].append(v)

            for k, v in _lts.items():
                v = torch.cat(v, dim=0)

                if v.shape[-1] == 3:
                    lts[k] = segment_coo(
                        src=(weights_ * v),
                        index=ray_id,
                        out=torch.zeros([N, 3], device=self.device),
                        reduce="sum",
                    )
                else:
                    lts[k] = segment_coo(
                        src=(weights * v.squeeze(-1)),
                        index=ray_id,
                        out=torch.zeros([N], device=self.device),
                        reduce="sum",
                    )

        if em_modes == 0:
            rgb_marched = off_rgb_marched
            lin_rgb_marched = lin_off_rgb_marched
        else:
            rgb_marched = on_rgb_marched
            lin_rgb_marched = lin_on_rgb_marched

        return {
            "etc/depth": depth,
            "etc/disp": disp,
            "etc/normal": normal_marched,
            "etc/white_bg": alphainv_last.unsqueeze(-1),
            "srgb/off_rgb": off_rgb_marched,
            "lin/off_rgb": lin_off_rgb_marched,
            "srgb/on_rgb": on_rgb_marched,
            "lin/on_rgb": lin_on_rgb_marched,
            "srgb/emo_rgb": emo_rgb_marched,
            "lin/emo_rgb": lin_emo_rgb_marched,
            "srgb/rgb": rgb_marched,
            "lin/rgb": lin_rgb_marched,
            "lin/emit": lin_emit_marched,
            "lin/basecolor": lin_basecolor,
            "lin/roughness": lin_roughness,
            "lin/metallic": lin_metallic,
            **lts,
        }

    def eval_emit(self, **kwargs):
        rays_o = kwargs["rays_o"]
        rays_d = kwargs["rays_d"]
        viewdirs = kwargs["viewdirs"]

        N = len(rays_o)

        ray_pts, ray_id, _ = self.sample_ray(
            rays_o=rays_o, rays_d=rays_d, near=self.near
        )

        # skip known free space
        mask = self.mask_cache(ray_pts)
        ray_pts = ray_pts[mask]
        ray_id = ray_id[mask]

        sdf, grad = self.sample_sdf_grad(ray_pts)

        dist = self.stepsize * self.voxel_size
        alpha = self.neus_alpha_from_sdf_scatter(
            viewdirs, ray_id, dist, sdf, grad, self.s_val
        )

        # app mask 0
        mask = alpha > self.fastcolor_thres
        alpha = alpha[mask].squeeze()
        ray_id = ray_id[mask].squeeze()
        ray_pts = ray_pts[mask].squeeze()
        sdf = sdf[mask].squeeze()

        # app mask 1
        if alpha.dim() != 1:
            return torch.zeros_like(rays_o)

        weights, _ = Alphas2Weights.apply(alpha, ray_id, N)  # type: ignore
        mask = weights > self.fastcolor_thres
        weights = weights[mask]
        ray_pts = ray_pts[mask]
        ray_id = ray_id[mask]
        sdf = sdf[mask]

        # rgb feature
        all_feat, _, all_normal = self.sample_sdfeat_grad_normal(
            ray_pts, displace=self.grad_feat
        )

        rays_xyz = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
        xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], dim=-1)
        brdf_feat = torch.cat([xyz_emb, sdf[:, None], all_feat, all_normal], dim=-1)

        emit = self.emitnet(torch.cat([self.emit_color(ray_pts), brdf_feat], dim=-1))

        # Ray marching
        return segment_coo(
            src=(weights.unsqueeze(-1) * emit),
            index=ray_id,
            out=torch.zeros([N, 3], device=self.device),
            reduce="sum",
        )

    def eval_esp(self, **kwargs):
        rays_o = kwargs["rays_o"]
        rays_d = kwargs["rays_d"]
        viewdirs = kwargs["viewdirs"]

        N = len(rays_o)

        ray_pts, ray_id, _ = self.sample_ray(
            rays_o=rays_o, rays_d=rays_d, near=self.near
        )

        # skip known free space
        mask = self.mask_cache(ray_pts)
        ray_pts = ray_pts[mask]
        ray_id = ray_id[mask]

        sdf, grad = self.sample_sdf_grad(ray_pts)

        dist = self.stepsize * self.voxel_size
        alpha = self.neus_alpha_from_sdf_scatter(
            viewdirs, ray_id, dist, sdf, grad, self.s_val
        )

        # app mask 0
        mask = alpha > self.fastcolor_thres
        alpha = alpha[mask].squeeze()
        ray_id = ray_id[mask].squeeze()
        ray_pts = ray_pts[mask].squeeze()
        sdf = sdf[mask].squeeze()

        # app mask 1
        if alpha.dim() != 1:
            return torch.zeros_like(rays_o)

        weights, _ = Alphas2Weights.apply(alpha, ray_id, N)  # type: ignore
        mask = weights > self.fastcolor_thres
        weights = weights[mask]
        ray_pts = ray_pts[mask]
        ray_id = ray_id[mask]
        sdf = sdf[mask]

        # Ray marching
        return segment_coo(
            src=(weights.unsqueeze(-1) * ray_pts),
            index=ray_id,
            out=torch.zeros([N, 3], device=self.device),
            reduce="sum",
        )

    def set_grid_resolution(self, num_voxels: int):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1 / 3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        print("voxel_size       {}".format(self.voxel_size))
        print("world_size       {}".format(self.world_size))

    @torch.no_grad()
    def set_nonempty_mask(self):
        # Find grid points that is inside nonempty (occupied) space
        self_grid_xyz = torch.stack(
            torch.meshgrid(
                torch.linspace(
                    self.xyz_min[0],  # type: ignore
                    self.xyz_max[0],  # type: ignore
                    self.sdf.grid.shape[2],
                    device=self.device,
                ),
                torch.linspace(
                    self.xyz_min[1],  # type: ignore
                    self.xyz_max[1],  # type: ignore
                    self.sdf.grid.shape[3],
                    device=self.device,
                ),
                torch.linspace(
                    self.xyz_min[2],  # type: ignore
                    self.xyz_max[2],  # type: ignore
                    self.sdf.grid.shape[4],
                    device=self.device,
                ),
            ),
            -1,
        )
        nonempty_mask: torch.Tensor = self.mask_cache(self_grid_xyz)[
            None, None
        ].contiguous()
        self.nonempty_mask = nonempty_mask
        self.sdf.grid[~self.nonempty_mask] = 1

    def density_total_variation(self, sdf_tv: float = 0, smooth_grad_tv: float = 0):
        tv = 0
        if sdf_tv > 0:
            tv += (
                total_variation(self.sdf.grid, self.nonempty_mask)
                / 2
                / self.voxel_size
                * sdf_tv
            )
        if smooth_grad_tv > 0:
            smooth_tv_error = self.tv_smooth_conv(
                self.gradient.permute(1, 0, 2, 3, 4)
            ).detach() - self.gradient.permute(1, 0, 2, 3, 4)
            smooth_tv_error = (
                smooth_tv_error[self.nonempty_mask.repeat(3, 1, 1, 1, 1)] ** 2
            )
            tv += smooth_tv_error.mean() * smooth_grad_tv
        return tv

    def sdf_total_variation_add_grad(self, weight: float, dense_mode: bool):
        w = weight * self.world_size.max() / 128
        self.sdf.total_variation_add_grad(w, w, w, dense_mode)

    def sample_ray(self, rays_o: torch.Tensor, rays_d: torch.Tensor, near: float):
        """Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        """
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = self.stepsize * self.voxel_size
        (
            ray_pts,
            mask_outbbox,
            ray_id,
            step_id,
            _,
            _,
            _,
        ) = render_utils_cuda.sample_pts_on_rays(  # type: ignore
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist
        )
        # correct the cuda output, which could have a bias of 1 randomly
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]

        return ray_pts, ray_id, step_id

    def grid_sampler(self, xyz: torch.Tensor, grid: torch.Tensor):
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip(
            (-1,)
        ) * 2 - 1
        ret = (
            F.grid_sample(grid, ind_norm, mode="bilinear", align_corners=True)
            .reshape(grid.shape[1], -1)
            .T.reshape(*shape, grid.shape[1])
            .squeeze(-1)
        )
        return ret

    def sample_sdf_grad(self, xyz: torch.Tensor):
        ret = self.grid_sampler(xyz, self.sdf.grid)

        _, grad, _ = self.sample_sdfeat_grad_normal(xyz, displace=self.sdf_displace)
        grad = torch.cat([grad[:, [2]], grad[:, [1]], grad[:, [0]]], dim=-1)

        return ret, grad

    def sample_sdfeat_grad_normal(
        self,
        xyz: torch.Tensor,
        displace: torch.Tensor,
        mode: str = "bilinear",
        align_corners: bool = True,
    ):
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)

        # ind from xyz to zyx !!!!!
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip(
            (-1,)
        ) * 2 - 1
        ind = ((ind_norm + 1) / 2) * (self.size_factor_zyx - 1)
        offset = self.sdf_offset[:, None, :] * displace[None, :, None]

        all_ind = ind.unsqueeze(-2) + offset.view(-1, 3)
        all_ind = all_ind.view(1, 1, 1, -1, 3)
        all_ind[..., 0] = all_ind[..., 0].clamp(min=0, max=self.grid_size[2] - 1)
        all_ind[..., 1] = all_ind[..., 1].clamp(min=0, max=self.grid_size[1] - 1)
        all_ind[..., 2] = all_ind[..., 2].clamp(min=0, max=self.grid_size[0] - 1)

        all_ind_norm = (all_ind / (self.size_factor_zyx - 1)) * 2 - 1
        feat = F.grid_sample(
            self.sdf.grid, all_ind_norm, mode=mode, align_corners=align_corners
        )

        all_ind = all_ind.view(1, 1, 1, -1, 6, len(displace), 3)
        diff = all_ind[:, :, :, :, 1::2, :, :] - all_ind[:, :, :, :, 0::2, :, :]
        diff, _ = diff.max(dim=-1)
        feat_ = feat.view(1, 1, 1, -1, 6, len(displace))
        feat_diff = feat_[:, :, :, :, 1::2, :] - feat_[:, :, :, :, 0::2, :]
        grad = feat_diff / (diff + 1e-12) / self.voxel_size

        feat = feat.view(shape[-1], 6, len(displace))
        grad = grad.view(shape[-1], 3, len(displace))
        normal = F.normalize(grad, dim=1)

        feat = feat.view(shape[-1], 6 * len(displace))
        grad = grad.view(shape[-1], 3 * len(displace))
        normal = normal.view(shape[-1], 3 * len(displace))

        return feat, grad, normal

    def sample_sdf_expgrad(self, xyz: torch.Tensor):
        def grid_sampler(xyz, grid):
            shape = xyz.shape[:-1]
            xyz = xyz.reshape(1, 1, 1, -1, 3)
            ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip(
                (-1,)
            ) * 2 - 1
            sdf = (
                differentiable_grid_sample(grid, ind_norm)
                .reshape(grid.shape[1], -1)
                .T.reshape(*shape, grid.shape[1])
                .squeeze(-1)
            )
            return sdf

        with torch.enable_grad():
            xyz.requires_grad_(True)
            sdf = grid_sampler(xyz, self.sdf.grid)

            grad_grad = torch.autograd.grad(
                sdf.sum(), xyz, retain_graph=self.training, create_graph=self.training
            )[0]
            xyz.requires_grad_(False)

        return sdf, grad_grad

    def sample_sdf_expgrad_grad_normal(self, xyz: torch.Tensor):
        sdf, expgrad = self.sample_sdf_expgrad(xyz)
        _, grad, normal = self.sample_sdfeat_grad_normal(
            xyz, displace=self.sdf_displace
        )
        grad = torch.cat([grad[:, [2]], grad[:, [1]], grad[:, [0]]], dim=-1)
        normal = torch.cat([normal[:, [2]], normal[:, [1]], normal[:, [0]]], dim=-1)
        return sdf, expgrad, grad, normal

    def neus_sdf_gradient(self):
        gradient = torch.zeros(
            [1, 3] + [*self.sdf.grid.shape[-3:]], device=self.sdf.grid.device
        )
        gradient[:, 0, 1:-1, :, :] = (
            (self.sdf.grid[:, 0, 2:, :, :] - self.sdf.grid[:, 0, :-2, :, :])
            / 2
            / self.voxel_size
        )
        gradient[:, 1, :, 1:-1, :] = (
            (self.sdf.grid[:, 0, :, 2:, :] - self.sdf.grid[:, 0, :, :-2, :])
            / 2
            / self.voxel_size
        )
        gradient[:, 2, :, :, 1:-1] = (
            (self.sdf.grid[:, 0, :, :, 2:] - self.sdf.grid[:, 0, :, :, :-2])
            / 2
            / self.voxel_size
        )
        return gradient

    @torch.no_grad()
    def extract_geometry(
        self,
        resolution: int = 512,
        threshold: float = 0.0,
        batch_size: int = 64,
        smooth: bool = True,
        sigma: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        bound_min = self.xyz_min.float()
        bound_max = self.xyz_max.float()
        if smooth:
            conv = Gaussian3DConv(sigma=sigma).to(self.device)
            sdf_grid = conv(self.sdf.grid)
        else:
            sdf_grid = self.sdf.grid

        def query_func(pts):
            return self.grid_sampler(pts, -sdf_grid)

        if resolution is None:
            resolution = self.world_size[0]  # type: ignore

        u = (
            extract_fields(bound_min, bound_max, resolution, query_func, batch_size)
            .cpu()
            .numpy()
        )
        vertices, triangles = mcubes.marching_cubes(u, threshold)
        b_max_np = bound_max.cpu().numpy()
        b_min_np = bound_min.cpu().numpy()

        vertices = (
            vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :]
            + b_min_np[None, :]
        )

        return vertices, triangles

    def apply_tonemapper(self, lin_rgb):
        lin_rgb_emb = (lin_rgb.unsqueeze(-1) * self.colorfreq).flatten(-2)
        lin_rgb_feat = torch.cat(
            [lin_rgb, lin_rgb_emb.sin(), lin_rgb_emb.cos()], dim=-1
        )
        return self.tonemapper(lin_rgb_feat)

    @torch.no_grad()
    def render_envmap(self, H, W):
        phi, theta = torch.meshgrid(
            [
                torch.linspace(0.0, np.pi, H, device=self.device),
                torch.linspace(1.0 * np.pi, -1.0 * np.pi, W, device=self.device),
            ]
        )
        dirs = torch.stack(
            [
                torch.cos(theta) * torch.sin(phi),
                torch.sin(theta) * torch.sin(phi),
                torch.cos(phi),
            ],
            dim=-1,
        ).view(-1, 3)
        return self.envmap(dirs).view(H, W, 3)
