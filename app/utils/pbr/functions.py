import numpy as np
import torch
from torch.nn import functional as F


def dot(a: torch.Tensor, b: torch.Tensor):
    return torch.sum(a * b, dim=-1, keepdim=True)


@torch.no_grad()
def diffuse_scattering(normal: torch.Tensor, number: int):
    """ """
    # uniform unit vector
    ret = F.normalize(
        torch.randn(*normal.shape[:-1], number, 3, device=normal.device), dim=-1
    )
    ret[torch.sum(ret * normal.unsqueeze(-2), dim=-1) < 0] *= -1.0
    return ret


@torch.no_grad()
def diffuse_scattering_fib(normal: torch.Tensor, number: int):
    """ """
    # uniform unit vector
    ret = (
        fibonacci_spiral_samples_on_unit_hemisphere(number)
        .expand(*normal.shape[:-1], number, 3)
        .clone()
        .to(normal.device)
    )
    ret[torch.sum(ret * normal.unsqueeze(-2), dim=-1) < 0] *= -1.0
    return ret


def micro_reflection(
    albedo: torch.Tensor,
    roughness: torch.Tensor,
    normal: torch.Tensor,
    win: torch.Tensor,
    wout: torch.Tensor,
):
    F0 = 0.04
    h = win + wout
    h = F.normalize(h, dim=-1)
    k = roughness**4 / 2.0
    # k = torch.pow(roughness + 1, 2) / 8.
    rho = roughness**2

    NoO = dot(normal, wout)
    NoI = dot(normal, win)
    NoH = dot(normal, h)  # n h cos
    HoI = dot(h, win)

    D = torch.pow(rho, 2) / (
        torch.pi * torch.pow(torch.pow(NoH, 2) * (torch.pow(rho, 2) - 1) + 1, 2)
    )
    _F = F0 + (1 - F0) * torch.pow(1 - HoI, 5)
    G = NoI / ((NoO * (1 - k) + k) * (NoI * (1 - k) + k))  # pre divide NoO in G

    R = D * _F * G / 2 * torch.pi + NoI * (1 - _F) * albedo * 2
    return R


def tensoir_reflection(
    albedo: torch.Tensor,
    roughness: torch.Tensor,
    normal: torch.Tensor,
    win: torch.Tensor,
    wout: torch.Tensor,
    fresnel: float = 0.04,
):
    # https://github.com/Haian-Jin/TensoIR/blob/8467f2c2af0d0aa1d01d5ecc2bfc6b32145422c4/models/relight_utils.py#L17
    # https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf

    L = F.normalize(win, dim=-1)  # [n, 3]
    V = F.normalize(wout, dim=-1)  # [n, 3]
    H = F.normalize((L + V) / 2.0, dim=-1)  # [n, 3]
    N = F.normalize(normal, dim=-1)  # [n, 3]

    NoV = torch.sum(V * N, dim=-1, keepdim=True)  # [n, 1]
    # calculate cosine(NoL in "brdf = 4 * np.pi * NoL * brdf") before "N = N * NoV.sign()" ?
    N = N * NoV.sign()  # [n, 3]

    NoL = torch.sum(N * L, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [n, 1]
    NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [n, 1]
    NoH = torch.sum(N * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [n, 1]
    VoH = torch.sum(V * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [n, 1]

    alpha = roughness * roughness  # [n, 1]
    alpha2 = alpha * alpha  # [n, 1]
    k = (alpha + 2 * roughness + 1.0) / 8.0  # [n, 1]
    FMi = ((-5.55473) * VoH - 6.98316) * VoH  # [n, 1]
    frac0 = fresnel + (1 - fresnel) * torch.pow(2.0, FMi)  # [n, 1]

    frac = frac0 * alpha2  # [n, 1]
    nom0 = NoH * NoH * (alpha2 - 1) + 1  # [n, 1]

    nom1 = NoV * (1 - k) + k  # [n, 1]
    nom2 = NoL * (1 - k) + k  # [n, 1]
    nom = (4 * np.pi * nom0 * nom0 * nom1 * nom2).clamp_(1e-6, 4 * np.pi)  # [n, 1]
    spec = frac / nom  # [n, 1]

    brdf = albedo / np.pi + spec  # [n, 3]
    brdf = 2 * np.pi * NoL * brdf  # 4pi?
    return brdf  # [n, 3]


def disney_reflection(
    albedo: torch.Tensor,
    roughness: torch.Tensor,
    metallic: torch.Tensor,
    normal: torch.Tensor,
    win: torch.Tensor,
    wout: torch.Tensor,
):
    """
    [specular] D * F * G / (n * wi) * (n * wo)
    [diffuse] + (1 - m) / pi * albedo

    return (specular + diffuse) * lamertian-law(win * n)


    D = 1 / (pi * r^2) * e^(2/(r^2) * (h * n - 1))
    F = F0 + (1 - F0)(1 - (wo * h)^5) where F0 = 0.04 * (1 - m) + b*m
    G = 2 * wi * n / (wi * n + sqrt(r ** 2 + (1 - r ** 2) * (wi * n) ** 2)) * 2 * wo * n / (wo * n + sqrt(r ** 2 + (1 - r ** 2) * (wo * n) ** 2))
    h is the half vector between the incident direction ω iand the viewing direction ω o
    """
    EPS = 1e-7

    def f_diffuse(a: torch.Tensor, m: torch.Tensor):
        return (1 - m) * a / torch.pi

    def f_specular(
        albedo: torch.Tensor,
        roughness: torch.Tensor,
        metallic: torch.Tensor,
        h_d_n: torch.Tensor,
        h_d_o: torch.Tensor,
        n_d_i: torch.Tensor,
        n_d_o: torch.Tensor,
    ):
        # used in SG, wrongly normalized
        def _d_sg(r: torch.Tensor, cos: torch.Tensor):
            r2 = (r * r).clamp(min=EPS)
            amp = 1 / (r2 * np.pi)
            sharp = 2 / r2
            return amp * torch.exp(sharp * (cos - 1))

        D = _d_sg(roughness, h_d_n)

        # Fresnel term F
        F_0 = 0.04 * (1 - metallic) + albedo * metallic
        F = F_0 + (1.0 - F_0) * ((1.0 - h_d_o) ** 5)

        # geometry term V, we use V = G / (4 * cos * cos) here
        def _v_schlick_ggx(r: torch.Tensor, cos: torch.Tensor):
            r2 = ((1 + r) ** 2) / 8
            return 0.5 / (cos * (1 - r2) + r2).clamp(min=EPS)

        V = _v_schlick_ggx(roughness, n_d_i) * _v_schlick_ggx(roughness, n_d_o)

        return D * F * V

    h = F.normalize(win + wout, dim=-1)
    noh = dot(normal, h).clamp(min=0)
    ooh = dot(wout, h).clamp(min=0)
    ion = dot(win, normal).clamp(min=0)
    oon = dot(wout, normal).clamp(min=0)

    fd = f_diffuse(albedo, metallic)
    fs = f_specular(albedo, roughness, metallic, noh, ooh, ion, oon)

    return (fd + fs) * ion * torch.pi * 2


def fibonacci_spiral_samples_on_unit_hemisphere(
    nb_samples: int, random: bool = False, up: bool = True
):
    # https://github.com/matt77hias/fibpy/blob/master/src/sampling.py
    n = 2 * nb_samples
    rn = torch.arange(nb_samples, n) if up else torch.arange(nb_samples)

    shift = 1.0 if random == 0 else n * np.random.random()

    ga = np.pi * (3.0 - np.sqrt(5.0))
    offset = 1.0 / nb_samples

    phi = ga * ((rn + shift) % n)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    cos_theta = ((rn + 0.5) * offset) - 1.0
    sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta)

    return torch.stack([cos_phi * sin_theta, sin_phi * sin_theta, cos_theta], dim=-1)


def fibonacci_spiral_samples_on_unit_sphere(nb_samples: int, random: bool = False):
    shift = 1.0 if random == 0 else nb_samples * np.random.random()

    ga = np.pi * (3.0 - np.sqrt(5.0))
    offset = 2.0 / nb_samples

    rn = torch.arange(nb_samples)

    phi = ga * ((rn + shift) % nb_samples)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    cos_theta = ((rn + 0.5) * offset) - 1.0
    sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta)

    return torch.stack([cos_phi * sin_theta, sin_phi * sin_theta, cos_theta], dim=-1)


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
