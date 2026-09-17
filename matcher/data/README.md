# data：冻结数据契约与纯转换

本目录定义 Matcher 唯一活动身份、清单读取边界、密度转换和全局 SMILES 图读取。唯一身份键是精确的顶层 `PreparedLigand.smiles`；`object_key`、沉积坐标、`present` 掩码和模型内部字符串均不参与身份。目录内代码只读取已经发布的产物，不扫描 Stage1 文件夹，也不修补缺失记录。

## 推荐阅读顺序

1. `schema.py`：先确认固定类别顺序、数组宽度、空间尺寸和正式来源目录。
2. `manifest.py`：理解训练与评估唯一允许读取的清单字段。
3. `smiles_graph.py`：理解精确 SMILES 如何映射为进程内重原子图。
4. `labels.py`：理解唯一最大覆盖 occurrence 的选择规则。
5. `density.py`：理解 56 个密度通道和单实验密度通道的构造。106 通道完整模型输入在 `matcher/datasets/candidate_data.py` 中由 56 个密度通道与 50 个受体通道连接得到。

## 文件职责

| 文件 | 稳定职责 |
| --- | --- |
| `schema.py` | 固定类别报告顺序、张量宽度、空间尺寸和来源目录。 |
| `manifest.py` | 读取并失败关闭地核验 `matcher_stage1_manifest_v1`，不扫描来源目录。 |
| `smiles_graph.py` | 每个进程完整载入一次 `smiles_graphs_v1.npz`，再按精确 SMILES 返回内存图。 |
| `labels.py` | 用候选和 occurrence 稀疏体素计算唯一最大覆盖目标。 |
| `density.py` | 构造 56 通道密度数组或 1 通道实验密度数组。 |

## manifest 产物布局

清单构建器在一个输出目录中发布一个入口文件和每个划分的两个版本化 JSONL 文件。训练只从 `manifest.json` 找到当前一代文件；未被入口引用的旧代文件或中断后遗留文件不是当前接口。

```text
<manifest_root>/
├── manifest.json                                      # 清单入口和当前代文件名
├── <split>_<generation>_candidates.jsonl             # 一个划分的冻结候选顺序
└── <split>_<generation>_identities.jsonl             # 一个划分的 PDB 内受支持身份轴
```

### `manifest.json`

| 字段 | 数据类型与形状 | 含义 | 示例值 |
| --- | --- | --- | --- |
| `schema` | `str` | 清单版本，当前只能为 `matcher_stage1_manifest_v1`。 | `"matcher_stage1_manifest_v1"` |
| `class_names` | `list[str]`，长度 6 | 固定报告顺序；索引 0 是背景，后五项仅用于真值分层。 | `["background", "ion", "nucleotide_like", "peptide_like", "small_molecule", "sugar"]` |
| `identity_key` | `str` | 唯一身份字段，当前只能为精确顶层 SMILES。 | `"PreparedLigand.smiles"` |
| `language_model` | `str` | 身份语言向量必须声明的公开模型名称。 | `"SMI-TED Light 289M"` |
| `data_root` | `str` | AdaLigand `Ori_Data` 根目录，身份语言向量和监督资产相对它定位。 | `"/storage/penghongen/AdaLigand/Ori_Data"` |
| `stage1_root` | `str` | Find_1 的 `artifacts/Find_1` 根目录，候选 centered 文件相对它定位。 | `"/storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts/Find_1"` |
| `graph_asset_path` | `str` | 全局 `smiles_graphs_v1.npz` 的路径。 | `"/storage/penghongen/AdaLigand/Ori_Data/smiles_assets/smiles_graphs_v1.npz"` |
| `splits` | `dict[str, object]` | 划分名到当前代文件的映射；每项只含 `candidates` 和 `identities` 两个相对文件名。 | `{"validation": {"candidates": "validation_123_candidates.jsonl", "identities": "validation_123_identities.jsonl"}}` |
| `summary` | `dict[str, object]` | 构建器冻结的逐划分规模和 PDB 轴；测试评估以其中的 `pdb_ids` 作为固定 PDB 分母。 | `{"held_out_test_0": {"pdb_ids": ["9abc", "9def"]}}` |

`manifest.json` 是跨文件提交点。构建器先写全新的版本化 JSONL，再原子替换入口；因此入口始终只引用一代完整文件。训练启动时会把入口写成运行目录内可直接读取的 `manifest.snapshot.json`，并把 `splits.*.candidates` 与 `splits.*.identities` 解析为不可变代际 JSONL 的绝对路径。train 与 validation Dataset 都读取该快照；原入口后续发布新代际不会改变本次运行实际读取的记录。快照不复制外部 JSONL 或图包内容。

`summary.<split>` 的字段是：`pdb_ids: list[str]` 保存扫描到的全部小写 PDB 标识，即使其中某个 PDB 没有 centered 候选也保留；`pdb_count: int` 是该列表长度；`candidate_count: int` 是全部候选数；`selected_candidate_count: int` 是 Stage1 已选候选数；`foreground_candidate_count: int` 是几何真前景数；`supported_foreground_candidate_count: int` 和 `unsupported_foreground_candidate_count: int` 分别是真前景中具有或缺少严格图与语言资产的数量；`train_validation_candidate_count: int` 是训练和 validation 身份过滤后保留的全部候选数；`selected_train_validation_candidate_count: int` 是同时应用 Stage1 已选过滤后的数量；`identity_count: int` 是按 PDB 分开的身份槽总数；`unique_smiles_count: int` 是当前划分跨 PDB 去重后的精确 SMILES 数。例如一个只有两个 PDB、10 个候选、8 个已选候选且无缺失身份的构造摘要可写为 `{"pdb_ids":["9abc","9def"],"pdb_count":2,"candidate_count":10,"selected_candidate_count":8,"foreground_candidate_count":6,"supported_foreground_candidate_count":6,"unsupported_foreground_candidate_count":0,"train_validation_candidate_count":10,"selected_train_validation_candidate_count":8,"identity_count":4,"unique_smiles_count":3}`。

### `<split>_<generation>_candidates.jsonl`

每行对应一个 Stage1 centered 候选，文件顺序就是 Dataset 的冻结候选顺序。

| 字段 | 数据类型 | 含义与特殊值 | 示例值 |
| --- | --- | --- | --- |
| `split` | `str` | 必须等于当前划分键。 | `"validation"` |
| `pdb_id` | `str` | 小写 PDB 标识。 | `"9abc"` |
| `centered_box_index` | `int >= 0` | 指向该 PDB `F2_centered.npz` 的稳定候选编号，不等于文件中的数组位置。 | `7` |
| `source_blob_index` | `int >= 0` | 指向 Stage1 源 blob；也是排序同分时的稳定次级键。 | `3` |
| `candidate_selected` | `bool` | Stage1 是否选择该候选；`selected_only=true` 时只读取 `true`。 | `true` |
| `gaussian_score` | 有限 `float` | Find_1 Gaussian 排序分数，较大者靠前。 | `0.731` |
| `source_probability_mean` | 有限 `float` | 源 blob 内 Stage1 概率均值，较大者靠前。 | `0.684` |
| `foreground_target` | `bool` | 是否存在双向覆盖率均不低于 0.30 的 occurrence。 | `true` |
| `target_category` | `str|null` | 真前景的五类分层标签；背景必须为 `null`。 | `"small_molecule"` |
| `matched_occurrence_id` | `int >= 0|null` | 最大化双向覆盖率几何平均、并列时编号最小的 occurrence；背景必须为 `null`。 | `2` |
| `target_smiles` | `str|null` | 目标 occurrence 的精确顶层 SMILES；背景为 `null`，无法支持身份的真前景也允许为 `null`。 | `"CC(=O)O"` |
| `candidate_coverage` | 有限 `float`，范围 `[0,1]` | 目标 occurrence 占据候选体素的比例；背景保存构建器给出的有限值。 | `0.75` |
| `occurrence_coverage` | 有限 `float`，范围 `[0,1]` | 候选覆盖目标 occurrence 体素的比例。 | `0.64` |
| `overlap_score` | 有限 `float`，范围 `[0,1]` | 两项覆盖率的几何平均，用于选择唯一 occurrence。 | `0.6928` |

### `<split>_<generation>_identities.jsonl`

每行对应一个有合法图和 SMI-TED 语言向量的 `(pdb_id, smiles)` 身份。同一 PDB 内记录由构建器按精确 SMILES 字典序发布，Dataset 继续保持该顺序。

| 字段 | 数据类型 | 含义 | 示例值 |
| --- | --- | --- | --- |
| `pdb_id` | 非空 `str` | 小写 PDB 标识。 | `"9abc"` |
| `smiles` | 非空 `str` | 精确顶层 `PreparedLigand.smiles`，不再次规范化或去空白。 | `"CC(=O)O"` |
| `language_relative_path` | 非空 `str` | 相对 `data_root` 的 SMI-TED 归档路径。 | `"stage1_preparation_box_pool_2/ligand_language_models/smi_ted_289m/9abc/2.npz"` |

真前景若没有对应身份记录，训练和 validation 跳过整条候选；test 保留候选与原排名，并令 `identity_target=-1`。背景不需要身份目标，始终可以保留。

## 全局 `smiles_graphs_v1.npz`

图包以精确 SMILES 为唯一逻辑键，不含坐标，也不引入 `graph_id`。设图数为 `K`、全部重原子数为 `N`、全部无向键数为 `E`。

| 字段 | 数据类型与形状 | 含义与对齐 | 示例值 |
| --- | --- | --- | --- |
| `schema_version` | `int64` 标量 | 当前固定为 1。 | `1` |
| `smiles` | Unicode，`(K,)` | 唯一精确 SMILES，按字符串严格递增。 | `["C", "CCO"]` |
| `atom_offsets` | `int64`，`(K+1,)` | 第 `i` 张图用区间 `[atom_offsets[i], atom_offsets[i+1])` 同步切分三个原子数组；首值 0，末值 N。 | `[0, 1, 4]` |
| `element` | `int16`，`(N,)` | 原子序数，与 `charge`、`atom_in_ring` 第一维逐原子对齐。 | `[6, 6, 6, 8]` |
| `charge` | `int8`，`(N,)` | 形式电荷。 | `[0, 0, 0, 0]` |
| `atom_in_ring` | `bool`，`(N,4)` | 四列依次标记 3、4、5、6 元环。 | `[false, false, false, false]` |
| `bond_offsets` | `int64`，`(K+1,)` | 第 `i` 张图用对应区间同步切分三个键数组；首值 0，末值 E。 | `[0, 0, 2]` |
| `bond_index` | `int32`，`(2,E)` | 每张图内部的无向键局部原子下标；两行依次为 RDKit 起点和终点。 | `[[0, 1], [1, 2]]` |
| `bond_type` | `uint8`，`(E,)` | `0..4` 依次为 single、double、triple、dative、aromatic。 | `[0, 0]` |
| `bond_in_ring` | `bool`，`(E,4)` | 与 `bond_index` 第二维逐键对齐的 3、4、5、6 元环标志。 | `[[false, false, false, false], [false, false, false, false]]` |

`SmilesGraphStore.get()` 把每条无向键展开成正反两个方向，得到 `edge_index (2,2E_i)`；对应 `edge_features (2E_i,9)` 的前五列是键型独热编码，后四列是环标志。持久 DataLoader worker 初始化时完整复制图包数组并关闭文件，样本读取只访问内存。

## 密度与最大覆盖目标

`density.py::build_density_channels()` 只构造 56 个密度相关通道。`candidate_data.py` 再连接 50 个受体原子栅格通道，形成 `strongest` 和 `trunk_only` 的 `(106,48,48,48)` 输入。`build_experimental_density_channel()` 只读取 supplied 实验密度，使用 0.1%/99.9% 分位截断和 z-score，返回 `(1,48,48,48)`；它不读取模拟密度、受体 hardmask 或 50 维受体栅格。

最大覆盖目标要求候选覆盖率和 occurrence 覆盖率同时不低于 0.30。多个有效 occurrence 先选择两项覆盖率几何平均最大的记录，再以较小 `occurrence_id` 打破并列；其精确 `PreparedLigand.smiles` 是唯一身份答案。
