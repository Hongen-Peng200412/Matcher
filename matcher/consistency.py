"""O 与 O′ 阈值判断的一致性诊断；本模块不参与反向传播。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class OOPrimeConsistency:
    """两个阈值判断的相等数量、总数和比例。"""

    equal_count: int
    total_count: int
    agreement: float


@torch.no_grad()
def O_O_prime_consistency(
    O_logit: Tensor,
    O_prime: Tensor,
    *,
    O_threshold: float = 0.5,
    O_radius: float = 12.0,
    O_prime_scale: float = 6.0,
) -> OOPrimeConsistency:
    """比较 ``sigmoid(O_logit)>0.5`` 与 ``O_prime>q(12 Å)``。"""

    O_positive = torch.sigmoid(O_logit.float()) > O_threshold
    O_prime_threshold = 1.0 / (1.0 + O_radius / O_prime_scale)
    equal_count = int((O_positive == (O_prime.float() > O_prime_threshold)).sum())
    total_count = O_logit.numel()
    return OOPrimeConsistency(equal_count, total_count, equal_count / total_count)
