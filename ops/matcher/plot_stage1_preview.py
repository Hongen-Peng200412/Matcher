"""从冻结 Stage1-only 元数据生成两张明确标记为预览的 top-k 图。"""

from __future__ import annotations

import argparse
from pathlib import Path

from matcher.evaluation.plotting import plot_stage1_only_preview


def main(argv: list[str] | None = None) -> int:
    """解析绘图命令并生成两张 Stage1-only 预览图。

    - argv: 可选命令行参数；None 表示读取当前进程命令行。
    - return: Gaussian 与来源概率均值两张 PNG 均写入成功时返回 0。
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args(argv)
    outputs = (
        plot_stage1_only_preview(
            arguments.metadata_path,
            "gaussian",
            arguments.output_directory
            / "topk_gaussian_stage1_only_preview.png",
        ),
        plot_stage1_only_preview(
            arguments.metadata_path,
            "source_probability_mean",
            arguments.output_directory
            / "topk_source_probability_mean_stage1_only_preview.png",
        ),
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
