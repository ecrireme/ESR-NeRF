import torch


def tensor2img(x):
    """convert 0~1 FloatTensor to 0~255 uint8 image"""
    return (255 * torch.clip(x.cpu(), 0, 1)).type(torch.uint8)


def mse2psnr(x):
    """return psnr"""
    return -10.0 * torch.log(x) / torch.log(torch.ones_like(x) * 10.0)


def apply_gamma_curve(image: torch.Tensor) -> torch.Tensor:
    """Apply standard sRGB gamma curve
    Args:
        image: 0-1 normalized tensor
    Returns:
        result: gamma curve applied image
    """
    rst = torch.empty_like(image)
    low_mask = image <= 0.0031308
    high_mask = low_mask.bitwise_not()
    rst[low_mask] = 12.92 * image[low_mask]
    rst[high_mask] = 1.055 * torch.pow(image[high_mask], 1 / 2.4) - 0.055
    return rst


def remove_gamma_curve(image: torch.Tensor) -> torch.Tensor:
    """ """
    rst = torch.empty_like(image)
    low_mask = image < 0.04045
    high_mask = low_mask.bitwise_not()

    rst[low_mask] = image[low_mask] / 12.92
    rst[high_mask] = torch.pow((image[high_mask] + 0.055) / 1.055, 2.4)
    return rst


def rgb_to_hsv(rgb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # https://kornia.readthedocs.io/en/latest/_modules/kornia/color/hsv.html

    max_rgb, argmax_rgb = rgb.max(-1)
    min_rgb, argmin_rgb = rgb.min(-1)
    deltac = max_rgb - min_rgb

    v = max_rgb
    s = deltac / (max_rgb + eps)

    deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-1) - rgb), dim=-1)

    h1 = bc - gc
    h2 = (rc - bc) + 2.0 * deltac
    h3 = (gc - rc) + 4.0 * deltac

    h = torch.stack((h1, h2, h3), dim=-1) / deltac.unsqueeze(-1)
    h = torch.gather(h, dim=-1, index=argmax_rgb.unsqueeze(-1)).squeeze(-1)
    h = (h / 6.0) % 1.0  # 0.0 ~ 1.0

    return torch.stack((h, s, v), dim=-1)


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    h: torch.Tensor = hsv[..., 0]
    s: torch.Tensor = hsv[..., 1]
    v: torch.Tensor = hsv[..., 2]

    hi: torch.Tensor = torch.floor(h * 6) % 6
    f: torch.Tensor = ((h * 6) % 6) - hi
    one: torch.Tensor = torch.tensor(1.0, device=hsv.device, dtype=hsv.dtype)
    p: torch.Tensor = v * (one - s)
    q: torch.Tensor = v * (one - f * s)
    t: torch.Tensor = v * (one - (one - f) * s)

    hi = hi.long()
    indices: torch.Tensor = torch.stack([hi, hi + 6, hi + 12], dim=-1)
    out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-1)
    out = torch.gather(out, -1, indices)

    return out
