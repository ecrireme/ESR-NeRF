import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from app.utils.base.functions import render_utils_cuda, total_variation_cuda


class DenseGrid(nn.Module):
    def __init__(
        self,
        channels: int,
        world_size: torch.Tensor,
        xyz_min: torch.Tensor,
        xyz_max: torch.Tensor,
    ):
        super().__init__()
        self.channels = channels
        self.world_size = world_size
        self.xyz_min = xyz_min
        self.xyz_max = xyz_max
        self.grid = nn.Parameter(torch.zeros([1, channels, *world_size]))  # type: ignore

    def forward(self, xyz):
        """xyz: global coordinates to query"""
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip(
            (-1,)
        ) * 2 - 1
        out = F.grid_sample(self.grid, ind_norm, mode="bilinear", align_corners=True)
        out = out.reshape(self.channels, -1).T.reshape(*shape, self.channels)
        if self.channels == 1:
            out = out.squeeze(-1)
        return out

    def scale_volume_grid(self, new_world_size):
        self.world_size = new_world_size
        if self.channels == 0:
            self.grid = nn.Parameter(torch.zeros([1, self.channels, *new_world_size]))
        else:
            self.grid = nn.Parameter(
                F.interpolate(
                    self.grid.data,
                    size=tuple(new_world_size),
                    mode="trilinear",
                    align_corners=True,
                )
            )

    def total_variation_add_grad(self, wx, wy, wz, dense_mode, mask=None):
        """Add gradients by total variation loss in-place"""
        if mask is None:
            total_variation_cuda.total_variation_add_grad(  # type: ignore
                self.grid, self.grid.grad, wx, wy, wz, dense_mode
            )
        else:
            mask = mask.detach()
            if self.grid.size(1) > 1 and mask.size() != self.grid.size():
                mask = mask.repeat(1, self.grid.size(1), 1, 1, 1).contiguous()
            assert mask.size() == self.grid.size()
            total_variation_cuda.total_variation_add_grad_new(  # type: ignore
                self.grid, self.grid.grad, mask.float(), wx, wy, wz, dense_mode
            )

    def get_dense_grid(self):
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f"channels={self.channels}, world_size={self.world_size.tolist()}"


class MaskCache(nn.Module):
    def __init__(
        self,
        xyz_min: torch.Tensor,
        xyz_max: torch.Tensor,
        density: torch.Tensor,
        alpha_init: float,
        cache_thres: float,
        ks: int,
    ):
        super().__init__()
        self.xyz_min = xyz_min
        self.xyz_max = xyz_max

        self.mask_cache_thres = cache_thres
        self.ks = ks

        self.density = F.max_pool3d(
            density,
            kernel_size=self.ks,
            padding=self.ks // 2,
            stride=1,
        )

        self.act_shift = np.log(1 / (1 - alpha_init) - 1)

    @torch.no_grad()
    def forward(self, xyz):
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip(
            (-1,)
        ) * 2 - 1
        density = F.grid_sample(self.density, ind_norm, align_corners=True)
        alpha = 1 - torch.exp(-F.softplus(density + self.act_shift))
        alpha = alpha.reshape(*shape)
        return alpha >= self.mask_cache_thres


class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(  # type: ignore
            alpha, ray_id, N
        )
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(  # type: ignore
            alpha,
            weights,
            T,
            alphainv_last,
            i_start,
            i_end,
            ctx.n_rays,
            grad_weights,
            grad_last,
        )
        return grad, None, None


class Gaussian3DConv(nn.Module):
    def __init__(self, ksize: int = 3, sigma: float = 1.0, channel: int = 1):
        super().__init__()
        self.ksize = ksize
        self.sigma = sigma
        self.channel = channel

        x = np.arange(-(ksize // 2), ksize // 2 + 1, 1)
        y = np.arange(-(ksize // 2), ksize // 2 + 1, 1)
        z = np.arange(-(ksize // 2), ksize // 2 + 1, 1)
        xx, yy, zz = np.meshgrid(x, y, z)
        kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
        kernel = torch.FloatTensor(kernel)
        self.m = nn.Conv3d(
            channel,
            channel,
            ksize,
            stride=1,
            padding=ksize // 2,
            padding_mode="replicate",
            groups=channel,
        )
        self.m.weight.data = (
            torch.cat([kernel[None, None, ...] for _ in range(channel)], dim=0)
            / kernel.sum()
        )
        self.m.bias.data = torch.zeros(channel)  # type: ignore
        for param in self.m.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.m(x)


class GradientConv(nn.Module):
    def __init__(self, sigma: int = 0):
        super().__init__()
        self.sigma = sigma

        kernel = np.asarray(
            [
                [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                [[2, 4, 2], [4, 8, 4], [2, 4, 2]],
                [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
            ]
        )
        # sigma controls the difference between naive [-1,1] and sobel kernel
        distance = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    distance[i, j, k] = (i - 1) ** 2 + (j - 1) ** 2 + (k - 1) ** 2 - 1
        kernel0 = kernel * np.exp(-distance * sigma)

        # smooth conv for TV
        self.m = nn.Conv3d(
            1, 1, (3, 3, 3), stride=1, padding=1, padding_mode="replicate"
        )
        weight = torch.from_numpy(kernel0 / kernel0.sum()).float()
        self.m.weight.data = weight.unsqueeze(0).unsqueeze(0).float()
        self.m.bias.data = torch.zeros(1)  # type: ignore
        for param in self.m.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.m(x)
