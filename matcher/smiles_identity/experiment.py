"""从一个正式 YAML 启动单一 SMILES 身份 Matcher 训练. """

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_smiles_experiment
from .engine import run_smiles_training


def main(argv: list[str] | None = None) -> int:
    """解析必需的 --config, 执行训练并打印正常收口后的运行目录. 

    - argv: 可选命令行参数列表; None 表示读取当前进程命令行. 
    - return: 训练达到配置停止条件并写出 summary.json 后返回零. 

    allocation 通过 kill_lock 中断驻留任务时进程不会到达本函数的正常返回, 也不会
    伪造 summary.json; 已写的 run.json、完整验证事件和 W&B 曲线仍保留. 
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args(argv)
    config = load_smiles_experiment(arguments.config)
    print(run_smiles_training(config, arguments.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
