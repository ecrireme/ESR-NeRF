import torch
from torch import nn
from torch.nn import functional as F


class RadianceNet(nn.Module):
    def __init__(self, inputdim: int, width: int, depth: int):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(inputdim, width),
            nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True))
                for _ in range(depth - 2)
            ],
            nn.Linear(width, 3, bias=True),
        )

    def forward(self, x) -> torch.Tensor:
        return F.softplus(self.linear(x))


class TonemapNet(nn.Module):
    def __init__(self, dim0: int, width: int, depth: int):
        super().__init__()

        self.srgb = nn.Sequential(
            nn.Linear(dim0, width),
            nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True))
                for _ in range(depth - 2)
            ],
            nn.Linear(width, 3, bias=True),
        )

    def forward(self, x):
        return torch.sigmoid(self.srgb(x))


class BRDFNet(nn.Module):
    def __init__(self, inputdim, width, depth, mode):
        if mode in ["micro", "tensoir"]:
            outputdim = 4
            self.splitter = lambda x: x.split([3, 1], -1)
        else:
            outputdim = 5
            self.splitter = lambda x: x.split([3, 1, 1], -1)

        super().__init__()
        self.brdfnet = nn.Sequential(
            nn.Linear(inputdim, width),
            nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True))
                for _ in range(depth - 2)
            ],
            nn.Linear(width, outputdim),
        )
        nn.init.constant_(self.brdfnet[-1].bias, 0)  # type: ignore

    def forward(self, x):
        x = torch.sigmoid(self.brdfnet(x))
        return self.splitter(x)


class EmissionNet(nn.Module):
    def __init__(self, inputdim, width, depth):
        super().__init__()
        self.brdfnet = nn.Sequential(
            nn.Linear(inputdim, width),
            nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True))
                for _ in range(depth - 2)
            ],
            nn.Linear(width, 3),
        )
        nn.init.constant_(self.brdfnet[-1].bias, 0)  # type: ignore

    def forward(self, x):
        return F.softplus(self.brdfnet(x))


class SphericalGaussian(nn.Module):
    def __init__(
        self,
        num_sg: int = 48,
        activation: str = "softplus",
    ):
        super().__init__()
        # relu, abs, exp, sigmoid, softplus
        if hasattr(torch, activation):
            self.activation = getattr(torch, activation)
        elif hasattr(F, activation):
            self.activation = getattr(F, activation)
        else:
            raise AttributeError(
                "'{}' not found in torch or torch.nn.functional".format(activation)
            )

        self.mus = torch.randn(num_sg, 3)  # .repeat([1, 3])
        self.lambdas = 10.0 + torch.abs(torch.randn(num_sg, 1) * 20.0)
        self.lobes = torch.randn(num_sg, 3)

        lambdas = torch.abs(self.lambdas)
        energy = (
            self.activation(self.mus)
            * 2.0
            * torch.pi
            / lambdas
            * (1.0 - torch.exp(-2.0 * lambdas))
        )
        normalized_mu = (
            self.activation(self.mus)
            / torch.sum(energy, dim=0, keepdim=True)
            * 2.0
            * torch.pi
            * 0.8
        )
        if self.activation in [torch.abs, torch.relu]:
            self.mus = normalized_mu
        elif self.activation is F.softplus:
            self.mus = torch.log(torch.exp(normalized_mu) - 1.0)
        elif self.activation is torch.exp:
            self.mus = torch.log(normalized_mu)

        self.mus = torch.nn.Parameter(self.mus)
        self.lambdas = torch.nn.Parameter(self.lambdas)
        self.lobes = torch.nn.Parameter(self.lobes)

    def forward(self, dirs):
        lobes = F.normalize(self.lobes, dim=-1)
        lambdas = torch.abs(self.lambdas)
        return self.activation(
            (
                self.mus
                * torch.exp(
                    lambdas * ((dirs.unsqueeze(-2) * lobes).sum(-1, keepdim=True) - 1.0)
                )
            ).sum(-2)
        )
