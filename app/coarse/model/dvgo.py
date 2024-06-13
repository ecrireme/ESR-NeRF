import time

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F


class DVGO(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        near: float,
        far: float,
        xyz_min: torch.Tensor,
        xyz_max: torch.Tensor,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.system.device

        # dynamic variables
        self.near = near
        self.far = far
        self.xyz_min = xyz_min
        self.xyz_max = xyz_max

        # static variables
        self.num_voxels = cfg.app.model.num_voxels
        self.alpha_init = cfg.app.model.alpha_init
        self.stepsize = cfg.app.model.stepsize

        self.set_grid_resolution(self.num_voxels)

        # determine the density bias shift
        self.act_shift = np.log(1 / (1 - self.alpha_init) - 1)
        print("dvgo: set density bias shift to {}".format(self.act_shift))

        # init density voxel grid
        self.density = nn.Parameter(torch.zeros([1, 1, *self.world_size]))

        # color voxel grid  (coarse stage)
        self.off_color = nn.Parameter(torch.zeros([1, 3, *self.world_size]))
        self.emo_color = nn.Parameter(torch.zeros([1, 3, *self.world_size]))

        self.N_samples = (
            int(np.linalg.norm(np.array(self.density.shape[2:]) + 1) / self.stepsize)
            + 1
        )

    def train(self, mode=True):
        if mode:
            self.forward = self.forward_training
        else:
            self.forward = self.forward_evaluate
        return super().train(mode)

    def voxel_count_views(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor, chunk_size: int
    ):
        print("dvgo: voxel_count_views start")
        eps_time = time.time()
        N_samples = (
            int(np.linalg.norm(np.array(self.density.shape[2:]) + 1) / self.stepsize)
            + 1
        )
        rng = torch.arange(N_samples, device=self.device)[None].float()
        count = torch.zeros_like(self.density.detach())
        for idx in range(len(rays_o)):
            ones = torch.ones_like(self.density).requires_grad_()

            for ro, rd in zip(
                rays_o[idx].split(chunk_size, dim=0),
                rays_d[idx].split(chunk_size, dim=0),
            ):
                vec = torch.where(rd == 0, torch.full_like(rd, 1e-6), rd)
                rate_a = (self.xyz_max - ro) / vec
                rate_b = (self.xyz_min - ro) / vec
                t_min = (
                    torch.minimum(rate_a, rate_b)
                    .amax(-1)
                    .clamp(min=self.near, max=self.far)
                )
                step = self.stepsize * self.voxel_size * rng
                interpx = t_min[..., None] + step / rd.norm(dim=-1, keepdim=True)
                rays_pts = ro[..., None, :] + rd[..., None, :] * interpx[..., None]
                self.grid_sampler(rays_pts, ones).sum().backward()
            with torch.no_grad():
                count += ones.grad > 1  # type: ignore
        eps_time = time.time() - eps_time
        print("dvgo: voxel_count_views finish (eps time: {} sec)".format(eps_time))
        return count

    def set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1 / 3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        print("voxel_size       {}".format(self.voxel_size))
        print("world_size       {}".format(self.world_size))

    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o):
        self_grid_xyz = torch.stack(
            torch.meshgrid(
                torch.linspace(
                    self.xyz_min[0],  # type: ignore
                    self.xyz_max[0],  # type: ignore
                    self.density.shape[2],
                    device=self.device,
                ),  # type: ignore
                torch.linspace(
                    self.xyz_min[1],  # type: ignore
                    self.xyz_max[1],  # type: ignore
                    self.density.shape[3],
                    device=self.device,
                ),  # type: ignore
                torch.linspace(
                    self.xyz_min[2],  # type: ignore
                    self.xyz_max[2],  # type: ignore
                    self.density.shape[4],
                    device=self.device,
                ),  # type: ignore
            ),
            -1,
        )

        nearest_dist = torch.stack(
            [
                (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
                for co in cam_o.split(100)  # for memory saving
            ]
        ).amin(0)
        self.density[nearest_dist[None, None] <= self.near] = -100

    def activate_density(self, density, interval=1):
        return 1 - torch.exp(-F.softplus(density + self.act_shift) * interval)

    def sample_ray(self, rays_o, rays_d, is_train=False):
        """Sample query points on rays"""
        # 1. determine the maximum number of query points to cover all possible rays
        N_samples = self.N_samples
        # 2. determine the two end-points of ray bbox intersection
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.xyz_max - rays_o) / vec
        rate_b = (self.xyz_min - rays_o) / vec
        # import ipdb; ipdb.set_trace()
        t_min = (
            torch.minimum(rate_a, rate_b).amax(-1).clamp(min=self.near, max=self.far)
        )
        t_max = (
            torch.maximum(rate_a, rate_b).amin(-1).clamp(min=self.near, max=self.far)
        )
        # 3. check wheter a raw intersect the bbox or not
        mask_outbbox = t_max <= t_min
        # 4. sample points on each ray
        rng = (
            torch.arange(N_samples, device=self.device)[None]
            .float()
            .repeat(rays_d.shape[-2], 1)
        )
        rng += torch.rand_like(rng[:, [0]]) * is_train
        step = self.stepsize * self.voxel_size * rng
        interpx = t_min[..., None] + step / rays_d.norm(dim=-1, keepdim=True)
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        # 5. update mask for query points outside bbox
        mask_outbbox = mask_outbbox[..., None] | (
            (self.xyz_min > rays_pts) | (rays_pts > self.xyz_max)
        ).any(dim=-1)
        # import ipdb; ipdb.set_trace()
        return rays_pts, mask_outbbox

    def forward_training(self, **kwargs):
        rays_o = kwargs["rays_o"]
        rays_d = kwargs["rays_d"]
        em_modes = kwargs["em_modes"]

        # sample points on rays
        rays_pts, mask_outbbox = self.sample_ray(
            rays_o=rays_o, rays_d=rays_d, is_train=True
        )
        interval = self.stepsize

        # query for alpha
        alpha = torch.zeros_like(rays_pts[..., 0])
        # post-activation
        density = self.grid_sampler(rays_pts[~mask_outbbox], self.density)
        alpha[~mask_outbbox] = self.activate_density(density, interval)

        # compute accumulated transmittance
        weights, alphainv_cum = get_ray_marching_ray(alpha)

        # query for color
        on_mask = em_modes == 1

        rgb = torch.zeros_like(rays_pts)
        rgb[on_mask] = torch.sigmoid(
            self.grid_sampler(rays_pts[on_mask], self.emo_color)
        )
        rgb = rgb + torch.sigmoid(self.grid_sampler(rays_pts, self.off_color))

        # Ray marching
        weights_ = weights.unsqueeze(-1)

        rgb_marched = (weights_ * rgb).sum(-2)

        return {
            "etc/alphainv_cum": alphainv_cum,
            "etc/weights": weights,
            "etc/white_bg": alphainv_cum[..., [-1]],
            "srgb/raw_rgb": rgb,
            "srgb/rgb": rgb_marched,
        }

    def forward_evaluate(self, **kwargs):
        rays_o = kwargs["rays_o"]
        rays_d = kwargs["rays_d"]
        em_modes = kwargs["em_modes"]
        # sample points on rays
        rays_pts, mask_outbbox = self.sample_ray(
            rays_o=rays_o, rays_d=rays_d, is_train=False
        )
        interval = self.stepsize

        # query for alpha
        alpha = torch.zeros_like(rays_pts[..., 0])
        # post-activation
        density = self.grid_sampler(rays_pts[~mask_outbbox], self.density)
        alpha[~mask_outbbox] = self.activate_density(density, interval)

        # compute accumulated transmittance
        weights, alphainv_cum = get_ray_marching_ray(alpha)

        # query for color
        off_rgb = torch.sigmoid(self.grid_sampler(rays_pts, self.off_color))
        emo_rgb = torch.sigmoid(self.grid_sampler(rays_pts, self.emo_color))
        on_rgb = off_rgb + emo_rgb

        # Ray marching
        weights_ = weights.unsqueeze(-1)

        off_rgb_marched = (weights_ * off_rgb).sum(-2)
        emo_rgb_marched = (weights_ * emo_rgb).sum(-2)
        on_rgb_marched = (weights_ * on_rgb).sum(-2)
        depth = (rays_o[..., None, :] - rays_pts).norm(dim=-1)
        depth = (weights * depth).sum(-1)
        disp = 1 / (depth + alphainv_cum[..., -1] * self.far)

        if em_modes == 0:
            rgb_marched = off_rgb_marched
        else:
            rgb_marched = on_rgb_marched

        return {
            "etc/depth": depth,
            "etc/disp": disp,
            "etc/white_bg": alphainv_cum[..., [-1]],
            "srgb/off_rgb": off_rgb_marched,
            "srgb/on_rgb": on_rgb_marched,
            "srgb/emo_rgb": emo_rgb_marched,
            "srgb/rgb": rgb_marched,
        }

    def grid_sampler(self, xyz, grid):
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


def get_ray_marching_ray(alpha):
    alphainv_cum = cumprod_exclusive(1 - alpha)
    weights = alpha * alphainv_cum[..., :-1]
    return weights, alphainv_cum


def cumprod_exclusive(p):
    # Not sure why: it will be slow at the end of training if clamping at 1e-10 is not applied
    return torch.cat([torch.ones_like(p[..., [0]]), p.clamp_min(1e-10).cumprod(-1)], -1)
