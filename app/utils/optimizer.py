import math
from typing import List

import rich
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor


def create_optimizer_or_freeze_model(model: nn.Module, **lrates: float):
    param_states = {}
    param_group = []
    for k, lr in lrates.items():
        if not hasattr(model, k):
            continue

        param = getattr(model, k)
        if param is None:
            print(f"create_optimizer_or_freeze_model: param {k} not exist")
            continue

        if lr > 0:
            # print(f"create_optimizer_or_freeze_model: param {k} lr {lr}")
            if isinstance(param, nn.Module):
                for np in param.named_parameters():
                    param_states[f"{k}.{np[0]}"] = {
                        "lr": lr,
                        "requires_grad": np[1].requires_grad,
                    }
                param = param.parameters()
            else:
                param_states[f"{k}"] = {"lr": lr, "requires_grad": param.requires_grad}
            param_group.append({"params": param, "lr": lr, "name": k})
        else:
            # print(f"create_optimizer_or_freeze_model: param {k} freeze")
            if isinstance(param, nn.Module):
                for np in param.named_parameters():
                    np[1].requires_grad = False
                    param_states[f"{k}.{np[0]}"] = {
                        "requires_grad": np[1].requires_grad
                    }
            else:
                param.requires_grad = False
                param_states[f"{k}"] = {"requires_grad": param.requires_grad}

    for np in model.named_parameters():
        n = np[0]
        if n not in param_states:
            param_states[n] = {"requires_grad": False}

    rich.print("Parameters")
    for n in sorted(param_states.keys()):
        s = param_states[n]
        rich.print(
            f"{n}, requires_grad: {s['requires_grad']}"
            + (f", lr: {s['lr']}" if "lr" in s else "")
        )

    return Adam(param_group, betas=(0.9, 0.99))


class Adam(torch.optim.Optimizer):
    """Extend Adam to support per-voxel learning rate"""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        self.per_lr = None
        super(Adam, self).__init__(params, defaults)
        self.name2pg = {pg["name"]: pg for pg in self.param_groups}

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def set_pervoxel_lr(self, count):
        assert self.param_groups[0]["params"][0].shape == count.shape
        self.per_lr = count.float() / count.max()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            per_lrs = []
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )
                    grads.append(p.grad)
                    if self.per_lr is not None and p.shape == self.per_lr.shape:
                        per_lrs.append(self.per_lr)
                    else:
                        per_lrs.append(None)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        if group["amsgrad"]:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq"] = torch.zeros_like(
                                p, memory_format=torch.preserve_format
                            )

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])

                    if group["amsgrad"]:
                        max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                    # update the steps for each param group update
                    state["step"] += 1
                    # record the step after step update
                    state_steps.append(state["step"])

            adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                per_lrs=per_lrs,
            )
        return loss


def adam(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[int],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    per_lrs,
):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        per_lr = per_lrs[i]

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        if per_lr is not None:
            param.addcdiv_(exp_avg * per_lr, denom, value=-step_size)
        else:
            param.addcdiv_(exp_avg, denom, value=-step_size)


class CosineLR:
    def __init__(self, cfg: DictConfig, cur_step: int = 0):
        self.cfg = cfg
        self.cur_step = cur_step
        self.warm_up_iters = cfg.app.trainer.warm_up_iters
        if self.warm_up_iters == -1:
            self.warm_up_iters = cfg.app.trainer.n_iters
        self.warm_up_min_ratio = cfg.app.trainer.warm_up_min_ratio
        self.n_iters = cfg.app.trainer.n_iters
        self.const_warm_up = cfg.app.trainer.const_warm_up
        self.cos_min_ratio = cfg.app.trainer.cos_min_ratio

        if cur_step == 0:
            self.pre_decay_factor = 1.0
        else:
            self.pre_decay_factor = self.cosine_lr_func(cur_step - 1)
        self.pos_decay_factor = self.cosine_lr_func(cur_step)

    @property
    def decay_factor(self):
        pre_decay_factor = self.pre_decay_factor
        pos_decay_factor = self.cosine_lr_func(self.cur_step)

        self.cur_step += 1
        self.pre_decay_factor = pos_decay_factor
        return pos_decay_factor / pre_decay_factor

    def cosine_lr_func(self, iter: int):
        if iter < self.warm_up_iters:
            if not self.const_warm_up:
                lr = self.warm_up_min_ratio + (1 - self.warm_up_min_ratio) * (
                    iter / self.warm_up_iters
                )
            else:
                lr = self.warm_up_min_ratio
        else:
            lr = (
                1
                + math.cos(
                    (iter - self.warm_up_iters)
                    / (self.n_iters - self.warm_up_iters)
                    * math.pi
                )
            ) * 0.5 * (1 - self.cos_min_ratio) + self.cos_min_ratio
        return lr
