"""统一设置训练与保留评估共用的随机源和 PyTorch 数值后端。

configure_random_seeds_and_numeric_backend 同时改变进程级随机状态与数值后端开关，并返回
可写入严格 JSON 的事实记录。训练引擎和 calibration 都调用同一入口，但各自仍须保存并
核对返回字段；本模块不声称不同硬件或不同软件版本之间逐位复现。
"""

from __future__ import annotations

import random
from typing import TypedDict

import numpy as np
import torch


class NumericBackendRecord(TypedDict):
    """记录一次调用完成后的随机种子与数值后端设置。

    - python_numpy_torch_seed: [0,2**32-1] 内的 Python 整数；同值写入 Python random、NumPy 旧式全局随机状态和 Torch CPU/CUDA 随机状态。
    - cudnn_benchmark: 布尔值 True；允许 cuDNN 按输入形状试选算法，可能弱化跨运行确定性。
    - cudnn_deterministic: 布尔值 False；不强制 cuDNN 只使用确定性实现。
    - deterministic_algorithms: 布尔值 False；不要求所有 Torch 算子采用确定性实现。
    - float32_matmul_precision: 字符串 high；传给 torch.set_float32_matmul_precision 的 float32 矩阵乘法内部精度策略。
    """

    python_numpy_torch_seed: int
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    deterministic_algorithms: bool
    float32_matmul_precision: str


def configure_random_seeds_and_numeric_backend(seed: int) -> NumericBackendRecord:
    """重置四类随机状态、设置数值后端，并返回对应 JSON 标量记录。

    - seed: [0,2**32-1] 内且 type(...) is int 的整数；布尔值与 NumPy 整数不会被接受。

    - 返回值.python_numpy_torch_seed: 与输入 seed 相同的整数。
    - 返回值.cudnn_benchmark: 固定为 True。
    - 返回值.cudnn_deterministic: 固定为 False。
    - 返回值.deterministic_algorithms: 固定为 False。
    - 返回值.float32_matmul_precision: 固定为 high。

    本函数重置 Python random、NumPy 旧式全局生成器、Torch CPU 默认生成器和所有当前可见
    CUDA 设备生成器；随后修改进程级 cuDNN、确定性算法和 float32 矩阵乘法设置。它不保存
    调用前状态，也不控制 DataLoader 自有生成器、第三方库随机源或未来新增 CUDA 设备。
    """

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
