"""从训练 run 记录的 release 恢复 Matcher 模型和 BEST checkpoint. """

from __future__ import annotations

import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _checkpoint_run_directory(checkpoint_path: Path) -> Path:
    """
    返回 BEST 或逐次 checkpoint 所属的训练运行目录.

    - checkpoint_path: BEST.pt、LAST.pt 或 checkpoints 目录中的逐次 checkpoint.
    - return: 含 run.json 的训练运行目录.
    """
    checkpoint_path = checkpoint_path.resolve()
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def _load_snapshot_package(project_root: Path) -> tuple[str, ModuleType]:
    """
    把训练 release 中的 matcher 包加载到独立模块命名空间.

    - project_root: run.json 记录的训练 release 项目根目录.
    - return[0]: 本进程内唯一的训练快照包名.
    - return[1]: 已加载的训练快照 matcher 包.

    独立包名使评估器继续使用当前 release 的数据读取与 top-k 逻辑，而模型配置、模型类和
    checkpoint 加载规则始终来自训练该 checkpoint 的 release.
    """
    package_directory = project_root / "matcher"
    package_initializer = package_directory / "__init__.py"
    if not package_initializer.is_file():
        raise FileNotFoundError(f"{package_initializer}: 缺少训练 matcher 包入口。")
    release_name = re.sub(r"\W+", "_", project_root.parent.name)
    package_name = f"_matcher_training_snapshot_{release_name}"
    existing = sys.modules.get(package_name)
    if existing is not None:
        existing_paths = tuple(Path(path).resolve() for path in existing.__path__)
        if existing_paths != (package_directory.resolve(),):
            raise RuntimeError(f"{package_name} 已绑定到另一份训练快照。")
        return package_name, existing
    specification = importlib.util.spec_from_file_location(
        package_name,
        package_initializer,
        submodule_search_locations=[str(package_directory)],
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"无法从 {package_initializer} 建立训练快照包。")
    package = importlib.util.module_from_spec(specification)
    sys.modules[package_name] = package
    try:
        specification.loader.exec_module(package)
    except Exception:
        sys.modules.pop(package_name, None)
        raise
    importlib.invalidate_caches()
    return package_name, package


def load_matcher_from_training_release(
    checkpoint_path: Path,
    device: Any,
) -> tuple[Any, dict[str, Any], Any, dict[str, str]]:
    """
    用训练 release 的模型代码严格恢复一个 Matcher checkpoint.

    - checkpoint_path: 受信任训练 run 中的 BEST.pt、LAST.pt 或逐次 checkpoint.
    - device: torch.device 或 torch.load 接受的设备位置.
    - return[0]: 已迁移到 device、严格加载权重并进入评估模式的模型.
    - return[1]: 训练快照 checkpoint 加载器返回的完整字段映射.
    - return[2]: 训练快照 MatcherModelConfig 实例.
    - return[3]: 供冻结测试产物记录的训练代码来源.

    run.json 必须同时指向现存训练项目根和 release manifest. 本函数不提供使用当前工作区
    模型代码的兼容分支，避免同名变体后来改定义后错误解释历史 checkpoint.
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    run_directory = _checkpoint_run_directory(checkpoint_path)
    run_json_path = run_directory / "run.json"
    run_record = json.loads(run_json_path.read_text(encoding="utf-8"))
    if not isinstance(run_record, dict):
        raise ValueError(f"{run_json_path} 顶层必须是对象。")
    source_project_root = run_record.get("source_project_root")
    release_manifest_path = run_record.get("source_release_manifest_path")
    if not isinstance(source_project_root, str) or not source_project_root:
        raise ValueError(f"{run_json_path} 缺少 source_project_root。")
    if not isinstance(release_manifest_path, str) or not release_manifest_path:
        raise ValueError(f"{run_json_path} 缺少 source_release_manifest_path。")
    project_root = Path(source_project_root).resolve()
    release_manifest = Path(release_manifest_path).resolve()
    if not release_manifest.is_file():
        raise FileNotFoundError(release_manifest)
    if release_manifest.parent != project_root.parent:
        raise ValueError("run.json 的训练项目根与 release manifest 不属于同一 release。")
    package_name, _ = _load_snapshot_package(project_root)
    checkpointing = importlib.import_module(f"{package_name}.training.checkpointing")
    model_config_module = importlib.import_module(f"{package_name}.model.config")
    matcher_module = importlib.import_module(f"{package_name}.model.ligand_matcher")
    checkpoint = dict(checkpointing.load_checkpoint(checkpoint_path, map_location=device))
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("BEST checkpoint 缺少 model_config。")
    config = model_config_module.MatcherModelConfig.from_dict(model_config)
    model = matcher_module.ClosedSetMatcher(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    provenance = {
        "model_code_source": "training_release",
        "training_run_json_path": str(run_json_path),
        "training_project_root": str(project_root),
        "training_release_manifest_path": str(release_manifest),
    }
    return model, checkpoint, config, provenance
