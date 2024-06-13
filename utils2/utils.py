import sys
from typing import Any, Dict, List, Optional, Union

import rich
import torch
import wandb
from omegaconf import DictConfig
from tqdm.auto import tqdm


def import_class(class_path) -> Any:
    """ """
    class_module, class_name = class_path.rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    _class = getattr(module, class_name)
    return _class


def tqdm_safe(iterator: Any, **tqdm_kwargs) -> Union[Any, tqdm]:
    if wandb.config["system"]["debug"]:
        return iterator
    else:
        return tqdm(
            iterator,
            miniters=wandb.config["system"]["tqdm_iters"],
            file=sys.stdout,
            dynamic_ncols=True,
            **tqdm_kwargs,
        )


LightDict = {
    "off": 0,
    "on": 1,
    "i_change": 2,
    "c_change": 3,
    "ic_change": 4,
}


class BatchSampler:
    def __init__(
        self,
        cfg: DictConfig,
        data: Dict[str, torch.Tensor],
        keys: List[str],
        batch_size: int,
        batch_st: int = 0,
        data_idxs: Optional[torch.Tensor] = None,
    ):
        self.cfg = cfg
        self.device = cfg.system.device
        self.data_preload_to_cpu = cfg.system.data_preload == "cpu"
        assert (
            "cpu" == cfg.system.data_preload
            or "gpu" in cfg.system.data_preload
            or "cuda" in cfg.system.data_preload
        )

        self.data = data
        self.keys = keys
        self.batch_size = batch_size
        self.batch_st = batch_st

        self.data_idxs = (
            torch.arange(len(data[keys[0]])) if data_idxs is None else data_idxs
        )
        self.data_num = len(self.data_idxs)

        if self.data_preload_to_cpu:
            self.data_idxs = torch._pin_memory(self.data_idxs.cpu().contiguous())
            for k in keys:
                data[k] = torch._pin_memory(data[k][self.data_idxs].contiguous())
        else:
            self.data_idxs = self.data_idxs.to(self.device).contiguous()
            for k in keys:
                v = data[k].to(self.device)
                data[k] = v[self.data_idxs].to(self.device).contiguous()

    def shuffle(self):
        if self.data_preload_to_cpu:
            b_ids = torch.randperm(self.data_num, pin_memory=True)
            self.data_idxs = torch._pin_memory(self.data_idxs[b_ids].contiguous())
            for k in self.keys:
                self.data[k] = torch._pin_memory(self.data[k][b_ids].contiguous())
        else:
            b_ids = torch.randperm(self.data_num, device=self.device)
            self.data_idxs = self.data_idxs[b_ids].contiguous()
            for k in self.keys:
                self.data[k] = self.data[k][b_ids].contiguous()
        self.batch_st = 0

    def filter(self, mask):
        if self.data_preload_to_cpu:
            mask = torch._pin_memory(mask.cpu().contiguous())
            for k in self.keys:
                self.data[k] = torch._pin_memory(self.data[k][mask].contiguous())
            self.data_idxs = torch._pin_memory(self.data_idxs[mask].contiguous())
        else:
            mask = mask.to(self.device).contiguous()
            for k in self.keys:
                self.data[k] = self.data[k][mask].contiguous()
            self.data_idxs = self.data_idxs[mask].contiguous()
        self.data_num = len(self.data_idxs)

    def sample(self):
        b_en = self.batch_st + self.batch_size

        if b_en > self.data_num:
            self.shuffle()
            b_en = self.batch_size

        b_st = self.batch_st
        self.batch_st = b_en

        return {
            k: self.data[k][b_st:b_en].to(self.device, non_blocking=True)
            for k in self.keys
        }


class RayGroupManager:
    def __init__(
        self,
        cfg: DictConfig,
        data: Dict[str, torch.Tensor],
        keys: List[str],
        uncert_batch_size: int,
        cert_batch_size: int,
        uncert_batch_st: int = 0,
        cert_batch_st: int = 0,
        uncert_data_idxs: Optional[torch.Tensor] = None,
        cert_data_idxs: Optional[torch.Tensor] = None,
    ):
        self.cfg = cfg
        self.device = cfg.system.device
        self.data_preload_to_cpu = cfg.system.data_preload == "cpu"
        assert (
            "cpu" == cfg.system.data_preload
            or "gpu" in cfg.system.data_preload
            or "cuda" in cfg.system.data_preload
        )
        self.uncert_data: Dict[str, torch.Tensor] = {k: data[k][:0] for k in keys}
        self.cert_data: Dict[str, torch.Tensor] = {k: data[k][:0] for k in keys}

        self.keys = keys
        self.uncert_batch_size = uncert_batch_size
        self.cert_batch_size = cert_batch_size
        self.uncert_batch_st = uncert_batch_st
        self.cert_batch_st = cert_batch_st

        self.uncert_data_idxs = (
            torch.arange(len(data[keys[0]]))
            if uncert_data_idxs is None
            else uncert_data_idxs
        )
        self.cert_data_idxs = (
            torch.arange(0) if cert_data_idxs is None else cert_data_idxs
        )

        if self.data_preload_to_cpu:
            self.uncert_data_idxs = torch._pin_memory(
                self.uncert_data_idxs.cpu().contiguous()
            )
            self.cert_data_idxs = torch._pin_memory(
                self.cert_data_idxs.cpu().contiguous()
            )
            for k in keys:
                v = data[k]
                self.uncert_data[k] = torch._pin_memory(
                    v[self.uncert_data_idxs].contiguous()
                )
                self.cert_data[k] = torch._pin_memory(
                    v[self.cert_data_idxs].contiguous()
                )
        else:
            self.uncert_data_idxs = self.uncert_data_idxs.to(self.device).contiguous()
            self.cert_data_idxs = self.cert_data_idxs.to(self.device).contiguous()
            for k in keys:
                v = data[k].to(self.device)
                self.uncert_data[k] = (
                    v[self.uncert_data_idxs].to(self.device).contiguous()
                )
                self.cert_data[k] = v[self.cert_data_idxs].to(self.device).contiguous()

    @property
    def uncert_data_num(self):
        return len(self.uncert_data_idxs)

    @property
    def cert_data_num(self):
        return len(self.cert_data_idxs)

    def shuffle(self):
        self.shuffle_uncert()
        self.shuffle_cert()

    def shuffle_uncert(self):
        if self.data_preload_to_cpu:
            b_ids = torch.randperm(self.uncert_data_num, pin_memory=True)
            self.uncert_data_idxs = torch._pin_memory(
                self.uncert_data_idxs[b_ids].contiguous()
            )
            for k in self.keys:
                self.uncert_data[k] = torch._pin_memory(
                    self.uncert_data[k][b_ids].contiguous()
                )
        else:
            b_ids = torch.randperm(self.uncert_data_num, device=self.device)
            self.uncert_data_idxs = self.uncert_data_idxs[b_ids].contiguous()
            for k in self.keys:
                self.uncert_data[k] = self.uncert_data[k][b_ids].contiguous()

        self.uncert_batch_st = 0

    def shuffle_cert(self):
        if self.data_preload_to_cpu:
            b_ids = torch.randperm(self.cert_data_num, pin_memory=True)
            self.cert_data_idxs = torch._pin_memory(
                self.cert_data_idxs[b_ids].contiguous()
            )
            for k in self.keys:
                self.cert_data[k] = torch._pin_memory(
                    self.cert_data[k][b_ids].contiguous()
                )
        else:
            b_ids = torch.randperm(self.cert_data_num, device=self.device)
            self.cert_data_idxs = self.cert_data_idxs[b_ids].contiguous()
            for k in self.keys:
                self.cert_data[k] = self.cert_data[k][b_ids].contiguous()

        self.cert_batch_st = 0

    def filter(self, mask: torch.Tensor):
        if self.data_preload_to_cpu:
            mask = torch._pin_memory(mask.cpu().contiguous())
            nmask = ~mask
            for k in self.keys:
                self.cert_data[k] = torch._pin_memory(
                    torch.concat(
                        [self.cert_data[k], self.uncert_data[k][nmask]], dim=0
                    ).contiguous()
                )
                self.uncert_data[k] = torch._pin_memory(
                    self.uncert_data[k][mask].contiguous()
                )

            self.cert_data_idxs = torch._pin_memory(
                torch.concat(
                    [self.cert_data_idxs, self.uncert_data_idxs[nmask]], dim=0
                ).contiguous()
            )
            self.uncert_data_idxs = torch._pin_memory(
                self.uncert_data_idxs[mask].contiguous()
            )
        else:
            mask = mask.to(self.device).contiguous()
            nmask = ~mask
            for k in self.keys:
                self.cert_data[k] = torch.concat(
                    [self.cert_data[k], self.uncert_data[k][nmask]], dim=0
                ).contiguous()
                self.uncert_data[k] = self.uncert_data[k][mask].contiguous()
            self.cert_data_idxs = torch.concat(
                [self.cert_data_idxs, self.uncert_data_idxs[nmask]], dim=0
            ).contiguous()
            self.uncert_data_idxs = self.uncert_data_idxs[mask].contiguous()

    def sample(self):
        uncert_b_en = self.uncert_batch_st + self.uncert_batch_size
        cert_b_en = self.cert_batch_st + self.cert_batch_size

        if uncert_b_en > self.uncert_data_num:
            self.shuffle_uncert()
            uncert_b_en = min(len(self.uncert_data_idxs), self.uncert_batch_size)
        if cert_b_en > self.cert_data_num:
            self.shuffle_cert()
            cert_b_en = min(len(self.cert_data_idxs), self.cert_batch_size)

        uncert_b_st = self.uncert_batch_st
        cert_b_st = self.cert_batch_st
        self.uncert_batch_st = uncert_b_en
        self.cert_batch_st = cert_b_en
        uncert_bs = uncert_b_en - uncert_b_st
        cert_bs = cert_b_en - cert_b_st

        batch = {
            k: torch.concat(
                [
                    self.uncert_data[k][uncert_b_st:uncert_b_en],
                    self.cert_data[k][cert_b_st:cert_b_en],
                ],
                dim=0,
            ).to(self.device, non_blocking=True)
            for k in self.keys
        }
        batch["uncert_masks"] = torch.ones(
            uncert_bs + cert_bs,
            dtype=torch.bool,
            device=self.device,
        )
        batch["uncert_masks"][-cert_bs:] = False
        return batch

    def print_stats(self):
        nuncert = self.uncert_data_num
        ncert = self.cert_data_num
        nall = nuncert + ncert

        rich.print(
            f"uncertain: [yellow]{nuncert}[/yellow]\t certain: [yellow]{ncert}[/yellow]\t uncertain/all: [yellow]{nuncert / nall * 100}[/yellow]"
        )
