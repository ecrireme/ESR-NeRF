import multiprocessing as mp
import warnings

import lpips
import numpy as np
import sklearn.neighbors as skln
import torch
import trimesh

from utils2.utils import tqdm_safe

__LPIPS__ = {}


def init_lpips(net_name, device):
    assert net_name in ["alex", "vgg"]
    print(f"init_lpips: lpips_{net_name}")
    return lpips.LPIPS(net=net_name, version="0.1").eval().to(device)


def rgb_lpips(np_gt, np_im, net_name, device):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if net_name not in __LPIPS__:
            __LPIPS__[net_name] = init_lpips(net_name, device)
        gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
        im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
        return __LPIPS__[net_name](gt, im, normalize=True).item()


def rgb_ssim(
    img0,
    img1,
    max_val,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    return_map=False,
):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        import scipy.signal

        return scipy.signal.convolve2d(z, f, mode="valid")

    def filt_fn(z):
        return np.stack(
            [
                convolve2d(convolve2d(z[..., i], filt[:, None]), filt[None, :])
                for i in range(z.shape[-1])
            ],
            -1,
        )

    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0.0, sigma00)
    sigma11 = np.maximum(0.0, sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


def loss2psnr(loss: float):
    return -10 * np.log10(loss)


def IoU(mask1: torch.Tensor, mask2: torch.Tensor):
    _Inter = (mask1 & mask2).sum().item()
    _Union = max(1, (mask1 | mask2).sum().item())
    return _Inter / _Union, _Inter, _Union


def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[: n1 + 1, : n2 + 1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q


def DTU_CD(
    mesh: trimesh.Trimesh,
    ObsMask: np.ndarray,
    BB: np.ndarray,
    Res: np.ndarray,
    stl: trimesh.points.PointCloud,
    ground_plane: np.ndarray,
    max_dist: float = 20.0,
    patch: int = 60,
    thresh: float = 0.2,
):
    prefix = "(CD)"
    pbar = tqdm_safe(range(8), desc=prefix, leave=False)
    pbar.set_description(prefix + " read data mesh")

    # use trimesh
    mesh.remove_unreferenced_vertices()
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.faces)
    tri_vert = vertices[triangles]

    pbar.update(1)
    pbar.set_description(prefix + " sample pcd from mesh")
    v1 = tri_vert[:, 1] - tri_vert[:, 0]
    v2 = tri_vert[:, 2] - tri_vert[:, 0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    non_zero_area = (area2 > 0)[:, 0]
    l1, l2, area2, v1, v2, tri_vert = [
        arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
    ]
    thr = thresh * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr)
    n2 = np.floor(l2 / thr)

    with mp.Pool() as mp_pool:
        new_pts = mp_pool.map(
            sample_single_tri,
            (
                (
                    n1[i, 0],
                    n2[i, 0],
                    v1[i : i + 1],
                    v2[i : i + 1],
                    tri_vert[i : i + 1, 0],
                )
                for i in range(len(n1))
            ),
            chunksize=1024,
        )

    new_pts = np.concatenate(new_pts, axis=0)
    data_pcd = np.concatenate([vertices, new_pts], axis=0)

    pbar.update(1)
    pbar.set_description(prefix + " random shuffle pcd index")
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    pbar.update(1)
    pbar.set_description(prefix + " downsample pcd")
    nn_engine = skln.NearestNeighbors(
        n_neighbors=1, radius=thresh, algorithm="kd_tree", n_jobs=-1
    )
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(
        data_pcd, radius=thresh, return_distance=False
    )
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]

    pbar.update(1)
    pbar.set_description(prefix + "masking data pcd")
    BB = BB.astype(np.float32)

    inbound = ((data_down >= BB[:1] - patch) & (data_down < BB[1:] + patch * 2)).sum(
        axis=-1
    ) == 3
    data_in = data_down[inbound]

    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    grid_inbound = (
        (data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))
    ).sum(axis=-1) == 3
    data_grid_in = data_grid[grid_inbound]
    in_obs = ObsMask[data_grid_in[:, 0], data_grid_in[:, 1], data_grid_in[:, 2]].astype(
        np.bool_
    )
    data_in_obs = data_in[grid_inbound][in_obs]

    # use trimesh
    stl = stl[::1]

    pbar.update(1)
    pbar.set_description(prefix + " compute data2stl")
    nn_engine.fit(stl)  # type: ignore
    dist_d2s, idx_d2s = nn_engine.kneighbors(
        data_in_obs, n_neighbors=1, return_distance=True
    )

    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    pbar.update(1)
    pbar.set_description(prefix + " compute stl2data")

    stl_hom = np.concatenate([stl, np.ones_like(stl[:, :1])], -1)  # type: ignore
    above = (ground_plane.reshape((1, 4)) * stl_hom).sum(-1) > 0
    stl_above = stl[above]

    nn_engine.fit(data_in)
    dist_s2d, idx_s2d = nn_engine.kneighbors(
        stl_above, n_neighbors=1, return_distance=True
    )
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    pbar.update(1)
    pbar.set_description(prefix + " visualize error")
    vis_dist = 1
    R = np.array([[1, 0, 0]], dtype=np.float64)
    G = np.array([[0, 1, 0]], dtype=np.float64)
    B = np.array([[0, 0, 1]], dtype=np.float64)
    W = np.array([[1, 1, 1]], dtype=np.float64)
    data_color = np.tile(B, (data_down.shape[0], 1))
    data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
    data_color[np.where(inbound)[0][grid_inbound][in_obs]] = R * data_alpha + W * (
        1 - data_alpha
    )
    data_color[
        np.where(inbound)[0][grid_inbound][in_obs][dist_d2s[:, 0] >= max_dist]
    ] = G
    stl_color = np.tile(B, (stl.shape[0], 1))
    stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
    stl_color[np.where(above)[0]] = R * stl_alpha + W * (1 - stl_alpha)
    stl_color[np.where(above)[0][dist_s2d[:, 0] >= max_dist]] = G

    pbar.update(1)
    pbar.close()
    over_all = (mean_d2s + mean_s2d) / 2
    return mean_d2s, mean_s2d, over_all
