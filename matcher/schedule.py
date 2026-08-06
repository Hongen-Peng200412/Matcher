"""Matcher 训练使用的学习率调度器。"""

from __future__ import annotations

from typing import Any

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WarmupPlateau:
    """先按优化步线性预热，再按完整验证事件降低学习率。"""

    def __init__(
        self,
        optimizer: AdamW,
        *,
        warmup_steps: int,
        warmup_start_factor: float,
        factor: float,
        patience: int,
        threshold: float,
        stop_after_lr_reductions: int,
        cooldown: int = 0,
        min_lr: float = 0.0,
        eps: float = 1.0e-8,
    ) -> None:
        self.optimizer = optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.warmup_steps = warmup_steps
        self.warmup_start_factor = warmup_start_factor
        self.warmup_step = 0
        self.reduction_count = 0
        self.stop_after_lr_reductions = stop_after_lr_reductions
        self.plateau = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode="abs",
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
        )
        if warmup_steps:
            for group, base_lr in zip(optimizer.param_groups, self.base_lrs, strict=True):
                group["lr"] = base_lr * warmup_start_factor

    def step_optimizer(self) -> None:
        """在每次参数更新后推进线性预热。"""

        if self.warmup_step >= self.warmup_steps:
            return
        self.warmup_step += 1
        progress = self.warmup_step / self.warmup_steps
        factor = self.warmup_start_factor + progress * (1.0 - self.warmup_start_factor)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * factor

    def step_validation(self, metric: float) -> None:
        """用一次完整验证的指标推进平台期调度。"""

        if self.warmup_step < self.warmup_steps:
            return
        before = [group["lr"] for group in self.optimizer.param_groups]
        self.plateau.step(metric)
        after = [group["lr"] for group in self.optimizer.param_groups]
        self.reduction_count += int(
            any(new < old - 1.0e-8 for old, new in zip(before, after))
        )

    @property
    def should_stop(self) -> bool:
        """返回学习率降低次数是否已达到停止阈值。"""

        return self.reduction_count >= self.stop_after_lr_reductions

    def state_dict(self) -> dict[str, Any]:
        """返回可写入 checkpoint 的调度状态。"""

        return {
            "base_lrs": self.base_lrs,
            "warmup_steps": self.warmup_steps,
            "warmup_start_factor": self.warmup_start_factor,
            "warmup_step": self.warmup_step,
            "reduction_count": self.reduction_count,
            "stop_after_lr_reductions": self.stop_after_lr_reductions,
            "plateau": self.plateau.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """恢复由 :meth:`state_dict` 保存的调度状态。"""

        self.base_lrs = state["base_lrs"]
        self.warmup_steps = state["warmup_steps"]
        self.warmup_start_factor = state["warmup_start_factor"]
        self.warmup_step = state["warmup_step"]
        self.reduction_count = state["reduction_count"]
        self.stop_after_lr_reductions = state["stop_after_lr_reductions"]
        self.plateau.load_state_dict(state["plateau"])
