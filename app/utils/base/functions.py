import os
import socket
from typing import Callable

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

root_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(
    root_dir, "build", str(socket.gethostname()), str(torch.cuda.get_device_name(0))
)
os.makedirs(build_dir, exist_ok=True)
render_utils_cuda = load(
    name="render_utils_cuda",
    sources=[
        os.path.join(root_dir, "cuda", path)
        for path in ["render_utils.cpp", "render_utils_kernel.cu"]
    ],
    verbose=True,
    build_directory=build_dir,
)
total_variation_cuda = load(
    name="total_variation_cuda",
    sources=[
        os.path.join(root_dir, "cuda", path)
        for path in ["total_variation.cpp", "total_variation_kernel.cu"]
    ],
    verbose=True,
    build_directory=build_dir,
)


def total_variation(v: torch.Tensor, mask=None):
    tv2 = v.diff(dim=2).abs()
    tv3 = v.diff(dim=3).abs()
    tv4 = v.diff(dim=4).abs()
    if mask is not None:
        tv2 = tv2[mask[:, :, :-1] & mask[:, :, 1:]]
        tv3 = tv3[mask[:, :, :, :-1] & mask[:, :, :, 1:]]
        tv4 = tv4[mask[:, :, :, :, :-1] & mask[:, :, :, :, 1:]]
    return (tv2.mean() + tv3.mean() + tv4.mean()) / 3


def neus_alpha_from_sdf_scatter_grad(
    viewdirs: torch.Tensor,
    ray_id: torch.Tensor,
    dist: torch.Tensor,
    sdf: torch.Tensor,
    gradients: torch.Tensor,
    s_val: float,
):
    iter_cos = (
        (viewdirs[ray_id] * gradients).sum(-1, keepdim=True) * dist.reshape(-1, 1) * 0.5
    )

    sdf = sdf.unsqueeze(-1)  # (M, 1)

    # dist is a constant in this impelmentation
    # Estimate signed distances at section points
    estimated_next_sdf = sdf + iter_cos  # (M, 1)
    estimated_prev_sdf = sdf - iter_cos  # (M, 1)

    prev_cdf = torch.sigmoid(estimated_prev_sdf * s_val)
    next_cdf = torch.sigmoid(estimated_next_sdf * s_val)
    p = F.relu(prev_cdf - next_cdf)
    c = prev_cdf
    alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0).squeeze()
    return alpha


def neus_alpha_from_sdf_scatter_interp(
    viewdirs: torch.Tensor,
    ray_id: torch.Tensor,
    dist: torch.Tensor,
    sdf: torch.Tensor,
    gradients: torch.Tensor,
    s_val: float,
):
    same_ray_mask = ray_id[:-1] == ray_id[1:]
    estimated_next_sdf = torch.cat(
        [
            torch.where(
                same_ray_mask, (sdf[..., :-1] + sdf[..., 1:]) * 0.5, sdf[..., :-1]
            ),
            sdf[..., -1:],
        ],
        -1,
    ).reshape(-1, 1)
    estimated_prev_sdf = torch.cat(
        [
            sdf[..., :1],
            torch.where(
                same_ray_mask, (sdf[..., :-1] + sdf[..., 1:]) * 0.5, sdf[..., 1:]
            ),
        ],
        -1,
    ).reshape(-1, 1)

    prev_cdf = torch.sigmoid(estimated_prev_sdf * s_val)
    next_cdf = torch.sigmoid(estimated_next_sdf * s_val)
    p = F.relu(prev_cdf - next_cdf)
    c = prev_cdf
    alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0).squeeze()
    return alpha


def extract_fields(
    bound_min: torch.Tensor,
    bound_max: torch.Tensor,
    resolution: int,
    query_func: Callable,
    N: int = 64,
):
    X = torch.linspace(bound_min[0], bound_max[0], resolution, device=bound_min.device)  # type: ignore
    Y = torch.linspace(bound_min[1], bound_max[1], resolution, device=bound_min.device)  # type: ignore
    Z = torch.linspace(bound_min[2], bound_max[2], resolution, device=bound_min.device)  # type: ignore

    u = torch.zeros([resolution, resolution, resolution], device=bound_min.device)
    with torch.no_grad():
        for xi, xs in enumerate(X.split(N)):
            for yi, ys in enumerate(Y.split(N)):
                for zi, zs in enumerate(Z.split(N)):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat(
                        [
                            xx.reshape(-1, 1),
                            yy.reshape(-1, 1),
                            zz.reshape(-1, 1),
                        ],
                        dim=-1,
                    )
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach()
                    u[
                        xi * N : xi * N + len(xs),
                        yi * N : yi * N + len(ys),
                        zi * N : zi * N + len(zs),
                    ] = val
    return u


def differentiable_grid_sample(
    image: torch.Tensor, optical: torch.Tensor, align_corners=False
):
    # https://github.com/pytorch/pytorch/issues/34704
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    iz = ((iz + 1) / 2) * (ID - 1)
    with torch.no_grad():
        ix_tnw = torch.floor(ix)
        iy_tnw = torch.floor(iy)
        iz_tnw = torch.floor(iz)

        ix_tne = ix_tnw + 1
        iy_tne = iy_tnw
        iz_tne = iz_tnw

        ix_tsw = ix_tnw
        iy_tsw = iy_tnw + 1
        iz_tsw = iz_tnw

        ix_tse = ix_tnw + 1
        iy_tse = iy_tnw + 1
        iz_tse = iz_tnw

        ix_bnw = ix_tnw
        iy_bnw = iy_tnw
        iz_bnw = iz_tnw + 1

        ix_bne = ix_tnw + 1
        iy_bne = iy_tnw
        iz_bne = iz_tnw + 1

        ix_bsw = ix_tnw
        iy_bsw = iy_tnw + 1
        iz_bsw = iz_tnw + 1

        ix_bse = ix_tnw + 1
        iy_bse = iy_tnw + 1
        iz_bse = iz_tnw + 1

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)

    with torch.no_grad():
        torch.clamp(ix_tnw, 0, IW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, IH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, ID - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, IW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, IH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, ID - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, IW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, IH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, ID - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, IW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, IH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, ID - 1, out=iz_tse)

        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

    image = image.view(N, C, ID * IH * IW)

    tnw_val = torch.gather(
        image,
        2,
        (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw)
        .long()
        .view(N, 1, D * H * W)
        .repeat(1, C, 1),
    )
    tne_val = torch.gather(
        image,
        2,
        (iz_tne * IW * IH + iy_tne * IW + ix_tne)
        .long()
        .view(N, 1, D * H * W)
        .repeat(1, C, 1),
    )
    tsw_val = torch.gather(
        image,
        2,
        (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw)
        .long()
        .view(N, 1, D * H * W)
        .repeat(1, C, 1),
    )
    tse_val = torch.gather(
        image,
        2,
        (iz_tse * IW * IH + iy_tse * IW + ix_tse)
        .long()
        .view(N, 1, D * H * W)
        .repeat(1, C, 1),
    )
    bnw_val = torch.gather(
        image,
        2,
        (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw)
        .long()
        .view(N, 1, D * H * W)
        .repeat(1, C, 1),
    )
    bne_val = torch.gather(
        image,
        2,
        (iz_bne * IW * IH + iy_bne * IW + ix_bne)
        .long()
        .view(N, 1, D * H * W)
        .repeat(1, C, 1),
    )
    bsw_val = torch.gather(
        image,
        2,
        (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw)
        .long()
        .view(N, 1, D * H * W)
        .repeat(1, C, 1),
    )
    bse_val = torch.gather(
        image,
        2,
        (iz_bse * IW * IH + iy_bse * IW + ix_bse)
        .long()
        .view(N, 1, D * H * W)
        .repeat(1, C, 1),
    )

    out_val = (
        tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W)
        + tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W)
        + tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W)
        + tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W)
        + bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W)
        + bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W)
        + bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W)
        + bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W)
    )

    return out_val