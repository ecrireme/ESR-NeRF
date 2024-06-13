import time
from typing import Tuple

import mcubes
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn import functional as F
from torch_scatter import segment_coo

from app.utils.base.functions import (
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
from app.utils.pbr.module import RadianceNet, TonemapNet


class VoxurfF(nn.Module):
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
        self.posbase_pe = cfg.app.model.posbase_pe
        self.viewbase_pe = cfg.app.model.viewbase_pe
        self.colorbase_pe = cfg.app.model.colorbase_pe
        self.grad_feat = torch.tensor(cfg.app.model.grad_feat, device=self.device)

        self.neus_alpha = cfg.app.model.neus_alpha

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
        self.sdf_random_init = True

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

    def train(self, mode=True):
        if mode:
            self.forward = self.forward_training
        else:
            self.forward = self.forward_evaluate
        return super().train(mode)

    def forward_training(self, **kwargs):
        rays_o = kwargs["rays_o"]
        rays_d = kwargs["rays_d"]
        viewdirs = kwargs["viewdirs"]
        em_modes = kwargs["em_modes"]
        self.s_val = kwargs["s_val"]

        N = len(rays_o)

        ray_pts, ray_id, _ = self.sample_ray(rays_o=rays_o, rays_d=rays_d)

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
        sdf = sdf[mask]

        # app mask 1
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)  # type: ignore
        mask = weights > self.fastcolor_thres
        weights = weights[mask]
        ray_pts = ray_pts[mask]
        ray_id = ray_id[mask]
        sdf = sdf[mask]

        # rgb feature
        on_mask = em_modes[ray_id] == 1
        off_mask = ~on_mask

        all_feat, _, all_normal = self.sample_sdfeat_grad_normal(
            ray_pts, displace=self.grad_feat
        )

        self.gradient = self.neus_sdf_gradient()

        rays_xyz = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
        viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
        rgb_feat = torch.cat(
            [
                rays_xyz,
                xyz_emb.sin(),
                xyz_emb.cos(),
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
        lin_rgb[on_mask] = (
            self.emo_rgbnet(
                torch.cat([self.emo_color(ray_pts[on_mask]), rgb_feat[on_mask]], dim=-1)
            )
            + self.off_rgbnet(
                torch.cat([self.off_color(ray_pts[on_mask]), rgb_feat[on_mask]], dim=-1)
            ).detach()
        )
        lin_rgb[off_mask] = self.off_rgbnet(
            torch.cat([self.off_color(ray_pts[off_mask]), rgb_feat[off_mask]], dim=-1)
        )

        rgb = self.apply_tonemapper(lin_rgb)

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
        return {
            "etc/alphainv_cum": alphainv_last,
            "etc/white_bg": alphainv_last[..., None],
            "srgb/rgb": rgb_marched,
            "lin/rgb": lin_rgb_marched,
        }

    def forward_evaluate(self, **kwargs):
        rays_o = kwargs["rays_o"]
        rays_d = kwargs["rays_d"]
        viewdirs = kwargs["viewdirs"]
        em_modes = kwargs["em_modes"]
        pos_rt = kwargs["pos_rt"]

        N = len(rays_o)

        ray_pts, ray_id, step_id = self.sample_ray(rays_o=rays_o, rays_d=rays_d)

        # skip known free space
        mask = self.mask_cache(ray_pts)
        ray_pts = ray_pts[mask]
        ray_id = ray_id[mask]
        step_id = step_id[mask]

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
        step_id = step_id[mask].squeeze()
        grad = grad[mask].squeeze()
        sdf = sdf[mask].squeeze()

        # app mask 1
        if alpha.dim() != 1:
            rgb_marched = torch.zeros_like(rays_o)
            depth = rgb_marched[..., 0]
            disp = 1 / (depth + self.far)
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
            }

        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)  # type: ignore
        mask = weights > self.fastcolor_thres
        weights = weights[mask]
        alpha = alpha[mask]
        ray_pts = ray_pts[mask]
        ray_id = ray_id[mask]
        step_id = step_id[mask]
        grad = grad[mask]
        sdf = sdf[mask]

        # rgb feature
        all_feat, _, all_normal = self.sample_sdfeat_grad_normal(
            ray_pts, displace=self.grad_feat
        )

        rays_xyz = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
        viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
        rgb_feat = torch.cat(
            [
                rays_xyz,
                xyz_emb.sin(),
                xyz_emb.cos(),
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
        }

    @torch.no_grad()
    def filter_training_rays_in_maskcache_sampling(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor, chunk_size: int
    ):
        print("get_training_rays_in_maskcache_sampling: start")
        eps_time = time.time()
        mask = torch.ones(len(rays_o), device=self.device, dtype=torch.bool)

        if self.sdf_random_init:
            for idx in torch.arange(len(rays_o), device=self.device).split(
                chunk_size, dim=0
            ):
                ray_pts, mask_outbbox, _ = self.sample_ray_ori(rays_o[idx], rays_d[idx])
                mask_outbbox[~mask_outbbox] |= ~self.mask_cache(ray_pts[~mask_outbbox])
                mask[idx] &= (~mask_outbbox).any(-1).to(self.device)

        else:
            for idx in torch.arange(len(rays_o), device=self.device).split(
                chunk_size, dim=0
            ):
                """Check whether the rays hit the solved coarse geometry or not"""
                far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
                ro = rays_o[idx].contiguous()
                rd = rays_d[idx].contiguous()
                stepdist = self.stepsize * self.voxel_size
                ray_pts, mask_outbbox, ray_id = render_utils_cuda.sample_pts_on_rays(  # type: ignore
                    ro, rd, self.xyz_min, self.xyz_max, self.near, far, stepdist
                )[:3]
                mask_inbbox = ~mask_outbbox
                hit = torch.zeros([len(ro)], dtype=torch.bool, device=self.device)
                hit[ray_id[mask_inbbox][self.mask_cache(ray_pts[mask_inbbox])]] = 1
                mask[idx] = hit

        eps_time = time.time() - eps_time
        print(
            f"get_training_rays_in_maskcache_sampling: ratio {mask.sum() / len(rays_o)}"
            + "\n"
            + f"get_training_rays_in_maskcache_sampling: finish (eps time: {eps_time} sec)"
        )
        return mask

    def sample_ray_ori(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor, is_train: bool = False
    ):
        """Sample query points on rays"""
        # 1. determine the maximum number of query points to cover all possible rays
        N_samples = (
            int(np.linalg.norm(np.array(self.sdf.grid.shape[2:]) + 1) / self.stepsize)
            + 1
        )
        # 2. determine the two end-points of ray bbox intersection
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.xyz_max - rays_o) / vec
        rate_b = (self.xyz_min - rays_o) / vec
        t_min = (
            torch.minimum(rate_a, rate_b).amax(-1).clamp(min=self.near, max=self.far)
        )
        t_max = (
            torch.maximum(rate_a, rate_b).amin(-1).clamp(min=self.near, max=self.far)
        )
        # 3. check wheter a raw intersect the bbox or not
        mask_outbbox = t_max <= t_min
        # 4. sample points on each ray
        rng = torch.arange(N_samples, device=self.device)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])
        step = self.stepsize * self.voxel_size * rng
        interpx = t_min[..., None] + step / rays_d.norm(dim=-1, keepdim=True)
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        # 5. update mask for query points outside bbox
        mask_outbbox = mask_outbbox[..., None] | (
            (self.xyz_min > rays_pts) | (rays_pts > self.xyz_max)
        ).any(dim=-1)
        return rays_pts, mask_outbbox, step

    def set_grid_resolution(self, num_voxels: int):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1 / 3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        print("voxel_size       {}".format(self.voxel_size))
        print("world_size       {}".format(self.world_size))

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print("fine: scale_volume_grid start")
        ori_world_size = self.world_size
        self.set_grid_resolution(num_voxels)
        print(
            f"fine: scale_volume_grid scale world_size from {ori_world_size} to {self.world_size}"
        )

        self.sdf.scale_volume_grid(self.world_size)
        self.off_color.scale_volume_grid(self.world_size)
        self.emo_color.scale_volume_grid(self.world_size)
        self.set_nonempty_mask()
        print("fine: scale_volume_grid finish")

        self.grid_size = self.sdf.grid.size()[-3:]
        self.size_factor_zyx = torch.tensor(
            [self.grid_size[2], self.grid_size[1], self.grid_size[0]],
            device=self.device,
        )

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

    def sample_ray(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
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
            rays_o, rays_d, self.xyz_min, self.xyz_max, self.near, far, stepdist
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
        grad = feat_diff / diff / self.voxel_size

        feat = feat.view(shape[-1], 6, len(displace))
        grad = grad.view(shape[-1], 3, len(displace))
        normal = F.normalize(grad, dim=1)

        feat = feat.view(shape[-1], 6 * len(displace))
        grad = grad.view(shape[-1], 3 * len(displace))
        normal = normal.view(shape[-1], 3 * len(displace))

        return feat, grad, normal

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
