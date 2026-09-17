"""用一个冻结 BEST checkpoint 评估真实受体与 CryoAtom2 test_0. """

from __future__ import annotations

import argparse
from pathlib import Path

from matcher.evaluation.test_set import build_final_test_artifact


def main(argv: list[str] | None = None) -> int:
    """解析冻结 test 命令并调用唯一评估器.

    - argv: 可选命令行参数; None 表示读取当前进程命令行.
    - return: 两种受体条件和可选 test_1 成功写入同一输出 JSON 时返回 0.
"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--cryoatom2-manifest", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--stage1-preview", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--test1-pdb-list", type=Path)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--pp-top-k", type=int, required=True)
    parser.add_argument("--cache-gib-per-worker", type=float, required=True)
    arguments = parser.parse_args(argv)
    print(
        build_final_test_artifact(
            arguments.checkpoint,
            arguments.real_manifest,
            arguments.cryoatom2_manifest,
            arguments.split,
            arguments.stage1_preview,
            arguments.output_path,
            test1_pdb_list_path=arguments.test1_pdb_list,
            batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
            pp_top_k=arguments.pp_top_k,
            cache_gib_per_worker=arguments.cache_gib_per_worker,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
