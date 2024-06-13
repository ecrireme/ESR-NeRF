from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset


class DataClass(Dataset, metaclass=ABCMeta):
    """BaseDataset defining essential APIs"""

    def __init__(self, cfg: DictConfig, phase: str):
        super().__init__()
        self.cfg = cfg
        self.phase = phase

        self.device = cfg.system.device

        self.root = cfg.data.root
        self.scene = cfg.data.scene
        self.resize = cfg.data.resize
        self.batch_type = cfg.data.batch_type
        self.white_bg = cfg.data.white_bg

        if self.batch_type != "nerf":
            raise NotImplementedError(
                "Unimplemented yet!. gaussian splattign style dataloader will be done ASAP."
            )

    @property
    @abstractmethod
    def image_size(self) -> Tuple[int, int]:
        """(width, height)"""
        raise NotImplementedError

    @property
    @abstractmethod
    def focal_length(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def all_data(self) -> Dict[str, torch.Tensor]:
        """Retrieve all of processed data."""
        raise NotImplementedError

    @property
    @abstractmethod
    def near_far(self) -> Tuple[float, float]:
        """(near, far)"""
        raise NotImplementedError

    @property
    @abstractmethod
    def scale_mat(self) -> torch.Tensor:
        """4x4 mat"""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Retrieve data. This must be faster than the seek method."""
        raise NotImplementedError

    @abstractmethod
    def seek(self, index: int) -> Dict[str, Any]:
        """Retrieve raw data without any processing."""
        raise NotImplementedError

    @abstractmethod
    def preprocess(self):
        raise NotImplementedError
