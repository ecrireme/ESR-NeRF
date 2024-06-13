import json
import math
import os
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import rich
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.nn import functional as F

from data import DataClass
from utils2.utils import LightDict, tqdm_safe


class ESRNeRF(DataClass):
    """ESR-NeRF dataloader"""

    def __init__(self, cfg: DictConfig, phase: str):
        super().__init__(cfg, phase)

        with open(
            os.path.join(
                self.root, self.scene, "transforms", f"transforms_{phase}.json"
            ),
            "r",
        ) as f:
            self.infos = json.load(f)

        sample = self.seek(0)
        self.width, self.height = sample["image"].size

        if self.resize:
            self.width = int(self.width * self.resize)
            self.height = int(self.height * self.resize)

        self.flen = (
            self.width / 2.0 / math.tan(float(self.infos["camera_angle_x"]) / 2.0)
        )

        # for rays
        self.blender2opencv = torch.FloatTensor(
            np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        )

        i, j = torch.meshgrid(
            torch.arange(self.width), torch.arange(self.height), indexing="xy"
        )
        i, j = i + 0.5, j + 0.5
        self.pixelcoord = torch.stack(
            [
                (i - self.width * 0.5) / self.flen,
                (j - self.height * 0.5) / self.flen,
                torch.ones_like(i),
            ],
            dim=-1,
        )

        # for fast loading
        self.cache: Dict[str, torch.Tensor] = {}
        self.preprocess()

    @property
    def image_size(self) -> Tuple[int, int]:
        return (self.width, self.height)

    @property
    def focal_length(self) -> float:
        return self.flen

    @property
    def all_data(self) -> Dict[str, torch.Tensor]:
        return self.cache

    @property
    def near_far(self) -> Tuple[float, float]:
        return 2.0, 6.0

    @property
    def scale_mat(self) -> torch.Tensor:
        return torch.eye(4)

    def __len__(self) -> int:
        return len(self.cache["rgbs"])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {k: v[index] for k, v in self.cache.items()}

    def seek(self, index: int) -> Dict[str, Any]:
        frame = self.infos["frames"][index]
        sample = {}
        sample["pose"] = np.asarray(frame["transform_matrix"])
        dname, fname = frame["file_path"].split("/")
        sample["image"] = Image.open(
            os.path.join(self.root, self.scene, dname, fname + ".png")
        )
        sample["em_mode"] = [light["mode"] for light in frame["lights"]]

        if self.phase == "test_nv":
            sample["area"] = Image.open(
                os.path.join(self.root, self.scene, dname, "emission", fname + ".png")
            )

        if self.phase != "train":
            sample["hdr"] = cv2.imread(  # type: ignore
                os.path.join(self.root, self.scene, dname, "exr", fname + ".exr"),
                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,  # type: ignore
            )
        if self.phase not in ["train", "test_nv"]:
            sample["em_mask"] = [
                Image.open(
                    os.path.join(self.root, self.scene, light["mask_path"] + ".png")
                )
                for light in frame["lights"]
            ]

            sample["em_color"] = [light["color"] for light in frame["lights"]]
            sample["em_intensity"] = [light["intensity"] for light in frame["lights"]]

        return sample

    def preprocess(self):
        rich.print(f"Data preprocessing for {self.phase} phase has been started.")

        cache: Dict[str, Any] = {
            "poses": [],
            "rays_o": [],
            "rays_d": [],
            "viewdirs": [],
            "rgbs": [],
            "em_modes": [],
        }
        if self.phase == "test_nv":
            cache["areas"] = []
            cache["hdrs"] = []

        if self.phase in ["test_nvi", "test_nvic"]:
            cache["em_masks"] = []
            cache["em_intensities"] = []

        if self.phase in ["test_nvc", "test_nvic"]:
            cache["em_masks"] = []
            cache["em_colors"] = []

        for i in tqdm_safe(range(len(self.infos["frames"]))):
            sample = self.seek(i)

            # pose
            sample["pose"] = torch.FloatTensor(sample["pose"])
            cache["poses"].append(sample["pose"])

            # rgb
            if self.resize:
                sample["image"] = sample["image"].resize(
                    (self.width, self.height), Image.LANCZOS
                )
            sample["image"] = torch.FloatTensor(
                np.asarray(sample["image"]) / 255.0
            ).view(self.width * self.height, -1)
            cache["rgbs"].append(sample["image"])

            if self.phase == "train":
                # em_mode (train)
                sample["em_mode"] = torch.LongTensor(
                    [LightDict[sample["em_mode"][0]]]
                ).expand(len(sample["image"]))

                cache["em_modes"].append(sample["em_mode"])
            else:
                # em_mode (eval)
                sample["em_mode"] = torch.LongTensor(
                    [
                        LightDict[sample["em_mode"][i]]
                        for i in range(len(sample["em_mode"]))
                    ]
                )
                cache["em_modes"].append(sample["em_mode"])

                if self.phase == "test_nv":
                    # area
                    if self.resize:
                        sample["area"] = sample["area"].resize(
                            (self.width, self.height), Image.LANCZOS
                        )
                    sample["area"] = (
                        torch.FloatTensor(np.asarray(sample["area"]) / 255.0)[..., 0]
                        > 0.5
                    ).view(-1)
                    cache["areas"].append(sample["area"])

                    # hdr
                    if self.resize:
                        sample["hdr"] = cv2.resize(  # type: ignore
                            sample["hdr"],
                            (self.width, self.height),
                            interpolation=cv2.INTER_LANCZOS4,  # type: ignore
                        )
                    sample["hdr"] = torch.FloatTensor(
                        cv2.cvtColor(sample["hdr"], cv2.COLOR_BGR2RGB)  # type: ignore
                    ).view(self.width * self.height, -1)
                    cache["hdrs"].append(sample["hdr"])

                else:
                    # em_mask
                    if self.resize:
                        sample["em_mask"] = [
                            m.resize((self.width, self.height), Image.LANCZOS)
                            for m in sample["em_mask"]
                        ]
                    sample["em_mask"] = torch.stack(
                        [
                            torch.FloatTensor(np.asarray(m) / 255.0)[..., 0].view(-1)
                            for m in sample["em_mask"]
                        ],
                        dim=0,
                    )
                    cache["em_masks"].append(sample["em_mask"])
                    # em_color
                    if self.phase in ["test_nvc", "test_nvic"]:
                        sample["em_color"] = torch.FloatTensor(
                            np.asarray(sample["em_color"])
                        )
                        cache["em_colors"].append(sample["em_color"])
                    # em_intensity
                    if self.phase in ["test_nvi", "test_nvic"]:
                        sample["em_intensity"] = torch.FloatTensor(
                            np.asarray(sample["em_intensity"])
                        )
                        cache["em_intensities"].append(sample["em_intensity"])

        cache = {k: torch.stack(v, dim=0) for k, v in cache.items() if len(v) > 0}

        mask = cache["rgbs"][..., -1:]
        cache["rgbs"] = cache["rgbs"][..., :3] * mask + (1 - mask) * self.white_bg
        cache["rays_o"], cache["rays_d"] = self.pose2ray(cache["poses"])
        cache["viewdirs"] = F.normalize(cache["rays_d"], dim=-1)

        if self.phase == "test_nv":
            cache["hdrs"] = cache["hdrs"][..., :3] * mask + (1 - mask) * self.white_bg

        if self.phase == "train":
            cache["rgbs"] = cache["rgbs"].reshape(-1, 3)
            cache["rays_o"] = cache["rays_o"].reshape(-1, 3)
            cache["rays_d"] = cache["rays_d"].reshape(-1, 3)
            cache["viewdirs"] = cache["viewdirs"].reshape(-1, 3)
            cache["em_modes"] = cache["em_modes"].reshape(-1)

        self.cache: Dict[str, torch.Tensor] = cache

    def pose2ray(self, pose: torch.Tensor):
        _pose = pose @ self.blender2opencv
        _pixel = self.pixelcoord.view(self.width * self.height, 3)
        rays_o = (
            _pose[..., :3, -1].unsqueeze(-2).expand(*_pose.shape[:-2], len(_pixel), 3)
        )
        rays_d = torch.sum(_pixel[:, None, :] * _pose[..., None, :3, :3], dim=-1)
        return rays_o, rays_d
