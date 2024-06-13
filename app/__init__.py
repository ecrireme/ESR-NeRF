from abc import ABCMeta, abstractmethod

from omegaconf import DictConfig


class AppClass(metaclass=ABCMeta):
    """Base model defining essential APIs"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = cfg.system.device
        self.data_preload = cfg.system.data_preload
        self.phase = cfg.app.phase

        self.white_bg = cfg.data.white_bg

    @property
    def global_step(self):
        return self.cfg.global_step

    @global_step.setter
    def global_step(self, iter: int):
        self.cfg.global_step = iter

    @abstractmethod
    def load_dataset(self):
        """Load dataset to run the process."""
        raise NotImplementedError

    @abstractmethod
    def load_model(self):
        """Load checkpoint to resume training, use pre-trained models, or to evaluate models.
        Note that, if checkpoint is not found, this function will init the model."""
        raise NotImplementedError

    @abstractmethod
    def process(self):
        """Run process"""
        raise NotImplementedError
