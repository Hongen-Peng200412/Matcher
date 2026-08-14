"""统一配置训练与保留评估使用的随机种子和 PyTorch 数值后端。

``configure_random_seeds_and_numeric_backend`` 是唯一入口；它同时执行配置并返回
可直接写入严格 JSON 的 ``NumericBackendRecord``，避免训练与 calibration 使用不同后端。
"""

from __future__ import annotations

import random
from typing import TypedDict

import numpy as np
import torch


class NumericBackendRecord(TypedDict):
    """记录一次正式运行的随机源和数值后端语义。

    ``python_numpy_torch_seed`` 是 Python、NumPy 和 Torch 共同使用的非负整数种子。
    ``cudnn_benchmark`` 表示允许 cuDNN 为固定形状选择高吞吐算法。
    ``cudnn_deterministic`` 表示未强制 cuDNN 使用确定性算法。
    ``deterministic_algorithms`` 表示未要求全部 Torch 算子逐位确定。
    ``float32_matmul_precision`` 是 PyTorch float32 矩阵乘法精度策略。
    """

    python_numpy_torch_seed: int
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    deterministic_algorithms: bool
    float32_matmul_precision: str


def configure_random_seeds_and_numeric_backend(seed: int) -> NumericBackendRecord:
    """配置训练与评估共同数值后端，并返回完整留证字段。"""

    if type(seed) is not int or seed < 0 or seed > 2**32 - 1:
        raise ValueError("随机种子必须是 [0,2**32-1] 内的整数。")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("high")
    return {
        "python_numpy_torch_seed": seed,
        "cudnn_benchmark": True,
        "cudnn_deterministic": False,
        "deterministic_algorithms": False,
        "float32_matmul_precision": "high",
    }
