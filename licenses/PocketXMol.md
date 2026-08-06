# Matcher 第三方实现来源

`matcher/graph.py` 中 `BondFFN`、`NodeBlock`、`EdgeBlock` 和 `NodeEdgeLayer` 的计算结构改编自 [PocketXMol](https://github.com/pengxingang/PocketXMol) 的 `models/graph.py`。PocketXMol 以 MIT License 发布，版权归 Peng Xingang（2024）所有；上游许可全文保存在 `matcher/LICENSE.PocketXMol`。本项目删除了位置更新、扩散时间和其他未使用模块，并改用 AdaLigand 的 31 维固定边契约与 PyTorch `index_add_` 聚合。

配体 149 维原子输入沿用本机 Emap2lig `src/emap2lig/data/dataset.py::LigandModelingDataset._process_features()` 的字段顺序；字段本身来自 AdaLigand 已落盘的 `LigandObject`。
