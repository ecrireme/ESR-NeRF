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


class VoxurfC(nn.Module):
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

        # static variables
        self.mask_ks = cfg.app.model.mask_ks
        self.maskcache_thres = cfg.app.model.maskcache_thres
        self.fastcolor_thres = cfg.app.model.fastcolor_thres

        self.stepsize = cfg.app.model.stepsize

        self.num_voxels = cfg.app.model.num_voxels

        self.color_dim = cfg.app.model.color_dim
        self.rgbnet_width = cfg.app.model.rgbnet_width
        self.rgbnet_depth = cfg.app.model.rgbnet_depth
        self.posbase_pe = cfg.app.model.posbase_pe
        self.viewbase_pe = cfg.app.model.viewbase_pe

        self.smooth_ksize = cfg.app.model.smooth_ksize
        self.smooth_sigma = cfg.app.model.smooth_sigma

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

        self.smooth_conv = Gaussian3DConv(self.smooth_ksize, self.smooth_sigma)
        print(
            "- " * 10
            + "init smooth conv with ksize={} and sigma={}".format(
                self.smooth_ksize, self.smooth_sigma
            )
            + " -" * 10
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

        # init color
        self.off_color = DenseGrid(
            channels=self.color_dim,
            world_size=self.world_size,
            xyz_min=self.xyz_min,
            xyz_max=self.xyz_max,
        )

        dim0 = (3 + 3 * self.posbase_pe * 2) + (3 * self.viewbase_pe * 3)
        dim0 += self.color_dim
        dim0 += 3
        self.off_rgbnet = nn.Sequential(
            nn.Linear(dim0, self.rgbnet_width),
            nn.ReLU(inplace=True),
            *[
                nn.Sequential(
                    nn.Linear(self.rgbnet_width, self.rgbnet_width),
                    nn.ReLU(inplace=True),
                )
                for _ in range(self.rgbnet_depth - 2)
            ],
            nn.Linear(self.rgbnet_width, 3),
        )
        nn.init.constant_(self.off_rgbnet[-1].bias, 0)  # type: ignore

        self.emo_color = DenseGrid(
            channels=self.color_dim,
            world_size=self.world_size,
            xyz_min=self.xyz_min,
            xyz_max=self.xyz_max,
        )
        self.emo_rgbnet = nn.Sequential(
            nn.Linear(dim0, self.rgbnet_width),
            nn.ReLU(inplace=True),
            *[
                nn.Sequential(
                    nn.Linear(self.rgbnet_width, self.rgbnet_width),
                    nn.ReLU(inplace=True),
                )
                for _ in range(self.rgbnet_depth - 2)
            ],
            nn.Linear(self.rgbnet_width, 3),
        )
        nn.init.constant_(self.emo_rgbnet[-1].bias, 0)  # type: ignore

        if self.neus_alpha == "grad":
            self.neus_alpha_from_sdf_scatter = neus_alpha_from_sdf_scatter_grad
        else:
            self.neus_alpha_from_sdf_scatter = neus_alpha_from_sdf_scatter_interp

        # useful variable
        self.normal_flipper = torch.tensor([1.0, -1.0, -1.0], device=self.device)

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

        sdf_grid = self.smooth_conv(self.sdf.grid)

        sdf = self.grid_sampler(ray_pts, sdf_grid)
        self.gradient = self.neus_sdf_gradient()
        gradient = self.grid_sampler(ray_pts, self.gradient)
        dist = self.stepsize * self.voxel_size
        alpha = self.neus_alpha_from_sdf_scatter(
            viewdirs, ray_id, dist, sdf, gradient, self.s_val
        )
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)  # type: ignore

        # app mask
        mask = weights > self.fastcolor_thres
        ray_pts = ray_pts[mask]
        ray_id = ray_id[mask]
        alpha = alpha[mask]
        gradient = gradient[mask]
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)  # type: ignore

        # rgb_feature
        on_mask = em_modes[ray_id] == 1

        rays_xyz = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
        viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
        normal = gradient / (gradient.norm(dim=-1, keepdim=True) + 1e-5)
        rgb_feat = torch.cat(
            [
                rays_xyz,
                xyz_emb.sin(),
                xyz_emb.cos(),
                viewdirs_emb[ray_id],
                viewdirs_emb.sin()[ray_id],
                viewdirs_emb.cos()[ray_id],
                normal,
            ],
            -1,
        )

        rgb = torch.zeros_like(ray_pts)
        rgb[on_mask] = torch.sigmoid(
            self.emo_rgbnet(
                torch.cat([self.emo_color(ray_pts[on_mask]), rgb_feat[on_mask]], -1)
            )
        )
        rgb = rgb + torch.sigmoid(
            self.off_rgbnet(torch.cat([self.off_color(ray_pts), rgb_feat], -1))
        )

        # Ray marching
        weights_ = weights.unsqueeze(-1)
        rgb_marched = segment_coo(
            src=(weights_ * rgb),
            index=ray_id,
            out=torch.zeros([N, 3], device=self.device),
            reduce="sum",
        )

        cum_weights = segment_coo(
            src=(weights_),
            index=ray_id,
            out=torch.zeros([N, 1], device=self.device),
            reduce="sum",
        )

        return {
            "etc/alphainv_cum": alphainv_last,
            "etc/white_bg": 1 - cum_weights,
            "srgb/rgb": rgb_marched,
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

        sdf_grid = self.smooth_conv(self.sdf.grid)

        sdf = self.grid_sampler(ray_pts, sdf_grid)
        self.gradient = self.neus_sdf_gradient()
        gradient = self.grid_sampler(ray_pts, self.gradient)
        dist = self.stepsize * self.voxel_size
        alpha = self.neus_alpha_from_sdf_scatter(
            viewdirs, ray_id, dist, sdf, gradient, self.s_val
        )

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
                "srgb/emo_rgb": rgb_marched,
                "srgb/on_rgb": rgb_marched,
                "srgb/rgb": rgb_marched,
            }

        weights, _ = Alphas2Weights.apply(alpha, ray_id, N)  # type: ignore

        # app_mask
        mask = weights > self.fastcolor_thres
        ray_pts = ray_pts[mask]
        ray_id = ray_id[mask]
        step_id = step_id[mask]
        alpha = alpha[mask]
        gradient = gradient[mask]

        if mask.sum() <= 1:
            rgb_marched = torch.zeros_like(rays_o)
            depth = rgb_marched[..., 0]
            disp = 1 / (depth + self.far)
            return {
                "etc/depth": depth,
                "etc/disp": disp,
                "etc/normal": rgb_marched,
                "etc/white_bg": torch.ones_like(rgb_marched[..., :1]),
                "srgb/off_rgb": rgb_marched,
                "srgb/emo_rgb": rgb_marched,
                "srgb/on_rgb": rgb_marched,
                "srgb/rgb": rgb_marched,
            }

        weights, _ = Alphas2Weights.apply(alpha, ray_id, N)  # type: ignore

        rays_xyz = (ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min)
        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
        viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
        normal = gradient / (gradient.norm(dim=-1, keepdim=True) + 1e-5)
        rgb_feat = [
            rays_xyz,
            xyz_emb.sin(),
            xyz_emb.cos(),
            viewdirs_emb[ray_id],
            viewdirs_emb.sin()[ray_id],
            viewdirs_emb.cos()[ray_id],
            normal,
        ]

        off_rgb = torch.sigmoid(
            self.off_rgbnet(torch.cat([self.off_color(ray_pts), *rgb_feat], -1))
        )
        emo_rgb = torch.sigmoid(
            self.emo_rgbnet(torch.cat([self.emo_color(ray_pts), *rgb_feat], -1))
        )
        on_rgb = off_rgb + emo_rgb

        # Ray marching
        weights_ = weights.unsqueeze(-1)
        off_rgb_marched = segment_coo(
            src=(weights_ * off_rgb),
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
        on_rgb_marched = segment_coo(
            src=(weights_ * on_rgb),
            index=ray_id,
            out=torch.zeros([N, 3], device=self.device),
            reduce="sum",
        )
        cum_weights = segment_coo(
            src=(weights_),
            index=ray_id,
            out=torch.zeros([N, 1], device=self.device),
            reduce="sum",
        )
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

        bg = 1 - cum_weights
        disp = 1 / (depth + bg[..., -1] * self.far)

        if em_modes == 0:
            rgb_marched = off_rgb_marched
        else:
            rgb_marched = on_rgb_marched

        return {
            "etc/depth": depth,
            "etc/disp": disp,
            "etc/normal": normal_marched,
            "etc/white_bg": bg,
            "srgb/off_rgb": off_rgb_marched,
            "srgb/emo_rgb": emo_rgb_marched,
            "srgb/on_rgb": on_rgb_marched,
            "srgb/rgb": rgb_marched,
        }

    @torch.no_grad()
    def filter_training_rays_in_maskcache_sampling(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor, chunk_size: int
    ):
        print("get_training_rays_in_maskcache_sampling: start")
        eps_time = time.time()
        mask = torch.ones(len(rays_o), device=self.device, dtype=torch.bool)

        for idx in torch.arange(len(rays_o), device=self.device).split(
            chunk_size, dim=0
        ):
            ray_pts, mask_outbbox, _ = self.sample_ray_ori(rays_o[idx], rays_d[idx])
            mask_outbbox[~mask_outbbox] |= ~self.mask_cache(ray_pts[~mask_outbbox])
            mask[idx] &= (~mask_outbbox).any(-1).to(self.device)

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

    def color_total_variation(self):
        v1 = self.off_color.grid
        v2 = self.emo_color.grid
        tv = total_variation(
            v1, self.nonempty_mask.repeat(1, v1.shape[1], 1, 1, 1)
        ) + total_variation(v2, self.nonempty_mask.repeat(1, v2.shape[1], 1, 1, 1))
        return tv

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
            N_steps,
            _,
            _,
        ) = render_utils_cuda.sample_pts_on_rays(  # type: ignore
            rays_o, rays_d, self.xyz_min, self.xyz_max, self.near, far, stepdist
        )
        # correct the cuda output N_steps, which could have a bias of 1 randomly
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
