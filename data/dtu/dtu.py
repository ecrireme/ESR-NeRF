import os
from glob import glob
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import rich
import torch
import trimesh
from omegaconf import DictConfig
from PIL import Image
from scipy.io import loadmat
from torch.nn import functional as F

from data import DataClass
from utils2.utils import LightDict, tqdm_safe


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(P=None):
    out = cv2.decomposeProjectionMatrix(P)  # type: ignore
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


# modified from Voxurf
class DTU(DataClass):
    def __init__(self, cfg: DictConfig, phase: str):
        super().__init__(cfg, phase)

        self.basedir = os.path.join(self.root, f"dtu_scan{self.scene}")

        self.camera_dict = np.load(os.path.join(self.basedir, "cameras_sphere.npz"))
        self.rgb_paths = sorted(glob(os.path.join(self.basedir, "image", "*.png")))
        self.mask_paths = sorted(glob(os.path.join(self.basedir, "mask", "*png")))

        sample = self.seek(0)
        self.width, self.height = sample["image"].size
        P = (sample["world_mat"] @ sample["scale_mat"])[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(P)

        self.flen = intrinsics[0, 0]
        self.K = intrinsics
        self._scale_mat = torch.FloatTensor(sample["scale_mat"])

        if self.resize:
            self.width = int(self.width * self.resize)
            self.height = int(self.height * self.resize)
            self.flen *= self.resize
            self.K[:2] *= self.resize

        # point cloud
        obs_mask_file = loadmat(f"{self.root}/ObsMask/ObsMask{self.scene}_10.mat")
        ObsMask, BB, Res = [obs_mask_file[attr] for attr in ["ObsMask", "BB", "Res"]]

        stl: trimesh.points.PointCloud = trimesh.load(
            f"{self.root}/Points/stl/stl{self.scene:03}_total.ply"
        )  # type: ignore
        ground_plane = loadmat(f"{self.root}/ObsMask/Plane{self.scene}.mat")["P"]

        self._pcd_info = (ObsMask, BB, Res, stl, ground_plane)

        # for rays
        i, j = torch.meshgrid(
            torch.arange(self.width), torch.arange(self.height), indexing="xy"
        )
        i, j = i + 0.5, j + 0.5
        self.pixelcoord = torch.stack(
            [
                (i - self.K[0][2]) / self.K[0][0],
                (j - self.K[1][2]) / self.K[1][1],
                torch.ones_like(i),
            ],
            dim=-1,
        )
        # for fast loading
        self.cache: Dict[str, torch.Tensor] = {}
        self.preprocess()

    @property
    def pcd(
        self,
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, trimesh.points.PointCloud, np.ndarray
    ]:
        """ObsMask, BB, Res, stl, groud_plane"""
        return self._pcd_info

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
        return self.near, self.far

    @property
    def scale_mat(self) -> torch.Tensor:
        return self._scale_mat

    def __len__(self) -> int:
        return len(self.cache["rgbs"])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {k: v[index] for k, v in self.cache.items()}

    def seek(self, index: int) -> Dict[str, Any]:
        sample = {}
        sample["world_mat"] = self.camera_dict[f"world_mat_{index}"].astype(np.float32)
        sample["scale_mat"] = self.camera_dict[f"scale_mat_{index}"].astype(np.float32)

        sample["image"] = Image.open(self.rgb_paths[index])
        sample["mask"] = Image.open(self.mask_paths[index])

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

        for i in tqdm_safe(range(len(self.rgb_paths))):
            sample = self.seek(i)

            # pose
            world_mat = sample["world_mat"]
            scale_mat = sample["scale_mat"]

            P = (world_mat @ scale_mat)[:3, :4]
            _, pose = load_K_Rt_from_P(P)

            pose = torch.FloatTensor(pose)
            cache["poses"].append(pose)

            # rgb
            if self.resize:
                sample["image"] = sample["image"].resize(
                    (self.width, self.height), Image.LANCZOS
                )
                sample["mask"] = sample["mask"].resize(
                    (self.width, self.height), Image.LANCZOS
                )
            sample["image"] = torch.FloatTensor(
                np.asarray(sample["image"]) / 255.0
            ).view(self.width * self.height, -1)
            sample["mask"] = torch.FloatTensor(np.asarray(sample["mask"]) / 255.0).view(
                self.width * self.height, -1
            )[..., :1]
            cache["rgbs"].append(
                sample["image"] * sample["mask"] + self.white_bg * (1 - sample["mask"])
            )

            # em_mode
            if self.phase == "train":
                cache["em_modes"].append(
                    torch.LongTensor([LightDict["off"]] * len(sample["image"]))
                )
            else:
                cache["em_modes"].append(torch.LongTensor([0]))

        cache = {k: torch.stack(v, dim=0) for k, v in cache.items() if len(v) > 0}

        cam_o = cache["poses"][:, :3, 3]
        self.far = np.linalg.norm(cam_o[:, None] - cam_o, axis=-1).max()
        self.near = self.far * 0.05

        cache["rays_o"], cache["rays_d"] = self.pose2ray(cache["poses"])
        cache["viewdirs"] = F.normalize(cache["rays_d"], dim=-1)

        if self.phase == "train":
            cache["rgbs"] = cache["rgbs"].reshape(-1, 3)
            cache["rays_o"] = cache["rays_o"].reshape(-1, 3)
            cache["rays_d"] = cache["rays_d"].reshape(-1, 3)
            cache["viewdirs"] = cache["viewdirs"].reshape(-1, 3)
            cache["em_modes"] = cache["em_modes"].reshape(-1)
        else:
            cache["hdrs"] = cache["rgbs"]

        self.cache: Dict[str, torch.Tensor] = cache

    def pose2ray(self, pose: torch.Tensor):
        pixel = self.pixelcoord.view(self.width * self.height, 3)
        rays_o = pose[..., :3, -1].unsqueeze(-2).expand(*pose.shape[:-2], len(pixel), 3)
        rays_d = torch.sum(pixel[:, None, :] * pose[..., None, :3, :3], dim=-1)
        return rays_o, rays_d
