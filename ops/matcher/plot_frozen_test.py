"""从冻结 Matcher 测试结果生成一张指标图和两张 top-k 图. """

from __future__ import annotations

import argparse
from pathlib import Path

from matcher.evaluation.plotting import plot_metric_bars, plot_topk_final


def main(argv: list[str] | None = None) -> int:
    """
    解析绘图参数并生成三张正式测试图.

    - argv: 可选命令行参数; None 表示读取当前进程命令行.
    - return: 三张 PNG 全部写入成功时返回 0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args(argv)
    output_directory = arguments.output_directory
    outputs = (
        plot_metric_bars(
            arguments.result_path,
            output_directory / "foreground_and_smiles_metrics.png",
        ),
        plot_topk_final(
            arguments.result_path,
            "gaussian",
            output_directory / "topk_gaussian_final.png",
        ),
        plot_topk_final(
            arguments.result_path,
            "source_probability_mean",
            output_directory / "topk_source_probability_mean_final.png",
        ),
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
