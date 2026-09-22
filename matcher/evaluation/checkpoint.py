"""从训练 run 指向的代码快照恢复 Matcher 模型和 checkpoint.

主要入口 load_matcher_from_training_release 读取 checkpoint 所属运行目录中的 run.json,
再从 source_project_root 加载训练时的模型配置、模型类和 checkpoint 读取器. 这样同名模型
变体在当前工作区发生变化后, 历史 checkpoint 仍由训练时的代码解释.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


def load_matcher_from_training_release(
    checkpoint_path: Path,
    device: Any,
) -> tuple[Any, dict[str, Any], Any]:
    """用训练代码快照恢复一个 Matcher checkpoint.

    输入参数:
        - checkpoint_path: Path, 训练运行中的 BEST.pt、LAST.pt 或 checkpoints 目录下的逐次 checkpoint.
        - device: torch.device 或 torch.load 接受的设备位置.

    返回值:
        - model: 已迁移到 device、严格加载参数并进入评估模式的 ClosedSetMatcher.
        - checkpoint: dict[str, Any], 训练快照 checkpoint 读取器返回的完整字段映射.
        - config: 训练快照中的 MatcherModelConfig 实例.
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    run_directory = (
        checkpoint_path.parent.parent
        if checkpoint_path.parent.name == "checkpoints"
        else checkpoint_path.parent
    )
    run_record = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    project_root = Path(run_record["source_project_root"]).resolve()
    package_directory = project_root / "matcher"
    package_name = "_matcher_training_snapshot_" + re.sub(
        r"\W+", "_", project_root.parent.name
    )
    if package_name not in sys.modules:
        specification = importlib.util.spec_from_file_location(
            package_name,
            package_directory / "__init__.py",
            submodule_search_locations=[str(package_directory)],
        )
        package = importlib.util.module_from_spec(specification)  # type: ignore[arg-type]
        sys.modules[package_name] = package
        specification.loader.exec_module(package)  # type: ignore[union-attr]
        importlib.invalidate_caches()

    checkpointing = importlib.import_module(f"{package_name}.training.checkpointing")
    model_config_module = importlib.import_module(f"{package_name}.model.config")
    matcher_module = importlib.import_module(f"{package_name}.model.ligand_matcher")
    checkpoint = dict(
        checkpointing.load_checkpoint(checkpoint_path, map_location=device)
    )
    config = model_config_module.MatcherModelConfig.from_dict(
        checkpoint["model_config"]
    )
    model = matcher_module.ClosedSetMatcher(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint, config
