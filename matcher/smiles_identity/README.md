# 精确 SMILES 身份管线

`matcher/smiles_identity/` 是 Matcher 当前从头训练入口。它把
`PreparedLigand.smiles` 的精确顶层字符串同时作为监督身份和化学图查询键，不读取
`object_key`，也不读取 LigandObject。原有 object_key 联合身份 v4 代码保留在
`matcher/data/`、`matcher/datasets/`、`matcher/model/`、`matcher/training/` 和
`matcher/evaluation/`，只服务历史实验复现。新管线复用其中不消费 LigandObject 的
`matcher.data.density.build_density_channels`、`matcher.utils.cache.ByteBudgetLru`、
`matcher.model.fusion` 和 `matcher.model.raunet`；它不复用旧图读取器、旧清单或旧身份
打分器。

## 科学契约

一个 PDB 内，同一精确 `PreparedLigand.smiles` 只建立一个身份槽。不同 PDB 的相同
字符串不会进入同一次闭集归一化。顶层字符串不得改写、重新 canonicalize 或替换成语言
归档中的 `model_smiles`。

前景候选只有在命中的精确 SMILES 身份同时具备下列两项时才进入清单：

- `smiles_graphs_v1.npz` 中存在该精确字符串的严格 RDKit 重原子图；
- 同一 PDB、同一精确字符串的至少一个 occurrence 具有固定 SMI-TED Light 289M
  有限 `(768,)` 向量，且归档的 `prepared_smiles` 与顶层字符串逐字相同。

语言向量属于 SMILES 身份，而不是某个 occurrence 的独立身份属性。一个组内只有部分
occurrence 留有语言文件时，身份仍保存全部图合法成员，并固定选择最小可用 occurrence
编号的向量；整个组都没有有效向量时才跳过该组对应的前景候选。背景候选不需要
SMILES，继续保留。新清单不生成 seen/unseen、语言家族、模板键或图编号字段。

## 文件职责

| 文件 | 职责 |
|---|---|
| `build_assets.py` | 从全部 `prepared_smiles.jsonl` 构建全局图包与 Stage3 自同构包。 |
| `graph.py` | 每个持久 DataLoader worker 一次载入全局图包，并按精确字符串返回图切片。 |
| `write_labels.py` | 用 Find_1 blob 与 occurrence 掩码的双向覆盖率生成紧邻 centered 的标签旁路。 |
| `manifest.py` | 连接 Find_1 候选、标签、顶层 SMILES、图和 SMI-TED，发布候选与身份 JSONL。 |
| `dataset.py` | 读取 supplied crop，在内存构造体素、四类 token、区域目标与 PDB 内身份轴。 |
| `batching.py` | 固定洗牌顺序，补齐 token 和身份轴，连接全部 SMILES 原子图。 |
| `candidate_encoder.py`、`model.py` | 产生候选预测，并计算语言、SMILES 图和同 PDB 身份清单三条分数。 |
| `losses.py` | 计算六分类、两个区域和全部五类前景 SMILES 身份损失。 |
| `config.py`、`engine.py`、`experiment.py` | 解析完整 YAML，执行单卡训练、完整验证和 W&B 记录。 |

推荐按 `schema.py` → `contracts.py` → `graph.py` → `labels.py` → `manifest.py` →
`dataset.py` → `batching.py` → `model.py` → `losses.py` → `config.py` → `engine.py` →
`experiment.py` 的顺序阅读。这个顺序先固定字段和张量，再连接数据，最后进入训练流程。

## 产物目录

```text
/storage/penghongen/AdaLigand/Ori_Data/
└── smiles_assets/
    ├── smiles_graphs_v1.npz             # Matcher 与 PocketXMol 共用的重原子化学图。
    ├── smiles_symmetries_v1.npz         # 仅供 PocketXMol 坐标重分配与 RMSD 使用。
    └── smiles_assets_v1_report.json      # 严格解析成功数、失败 SMILES 与来源证据。

/storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts/Find_1/
└── validation/<pdb_id>/centered/
    ├── F2_centered.npz                  # Find_1 已有的 supplied crop 与候选特征。
    └── F2_centered_label.npz            # 本管线新增且与候选轴对齐的监督旁路。

/storage/penghongen/tmp/Matcher/
├── manifests/smiles_identity/
│   ├── manifest.json                    # 固定来源、类别顺序和 split 文件入口。
│   ├── validation_candidates.jsonl      # 每个保留候选的分类、命中与覆盖率。
│   └── validation_identities.jsonl      # 每个 PDB 内精确 SMILES 身份及语言来源。
└── experiments/<name>/<run_stamp>/
    ├── run.json                          # 配置、运行状态以及 W&B run ID 与 URL。
    ├── history.jsonl                     # 初始验证与后续完整验证的标量记录。
    ├── LAST.pt                           # 仅在 save_checkpoints=true 时存在。
    └── summary.json                      # 正常结束时的最终状态与统计。
```

## 图资产

`smiles_graphs_v1.npz` 以同序 `smiles` 和 offsets 连接变长图，只保存可由 SMILES
唯一决定的字段：原子序数、形式电荷、3/4/5/6 元环标志、化学键端点、五类键型和
3/4/5/6 元环标志。它不保存坐标、构象键长、残基编号、present、手性模板字段或新造的
`graph_id`。

`smiles_symmetries_v1.npz` 使用相同精确字符串顺序保存 RDKit 重原子自同构。Matcher
不读取该文件；它只供未来 PocketXMol 的坐标重分配和 RMSD 使用。occurrence 坐标必须
另按 PDB 与 `candidate_id` 保存，不得混入全局化学图包。

### `smiles_graphs_v1.npz`

设图包含 `K` 个 SMILES、总计 `A` 个重原子和 `E` 条无向化学键。

| 字段 | dtype 与形状 | 语义 |
|---|---|---|
| `schema_version` | `int64` 标量 | 固定为 `1`。 |
| `smiles` | Unicode `(K,)` | 精确字符串严格递增，且与自同构包同序。 |
| `atom_offsets` | `int64 (K+1,)` | 第 `i` 张图的原子切片为 `[atom_offsets[i], atom_offsets[i+1])`。 |
| `element` | `int16 (A,)` | 重原子序数。 |
| `charge` | `int8 (A,)` | 形式电荷。 |
| `atom_in_ring` | `bool (A,4)` | 四列依次标记 3、4、5、6 元环。 |
| `bond_offsets` | `int64 (K+1,)` | 第 `i` 张图的键切片为 `[bond_offsets[i], bond_offsets[i+1])`。 |
| `bond_index` | `int32 (2,E)` | 每列为一条无向键的图内局部起点和终点。 |
| `bond_type` | `uint8 (E,)` | `0–4` 依次表示单键、双键、三键、配位键和芳香键。 |
| `bond_in_ring` | `bool (E,4)` | 四列依次标记该键是否属于 3、4、5、6 元环。 |

例如两张图分别含 3 和 1 个原子、2 和 0 条无向键时，`atom_offsets` 为
`[0, 3, 4]`，`bond_offsets` 为 `[0, 2, 2]`。第二张图的原子切片是 `[3, 4)`，
键切片是 `[2, 2)`；空键切片不表示图缺失。

无键的单原子离子具有非空原子切片和空键切片。运行时才把每条无向键展开为两个方向；
这个展开不改变持久化图包。

### `smiles_symmetries_v1.npz`

设全部紧凑自同构矩阵连接后有 `M` 个整数。

| 字段 | dtype 与形状 | 语义 |
|---|---|---|
| `schema_version` | `int64` 标量 | 固定为 `1`。 |
| `smiles` | Unicode `(K,)` | 与图包的 `smiles` 逐项相同。 |
| `atom_count` | `int32 (K,)` | 每个 SMILES 的重原子数。 |
| `matches_shape` | `int32 (K,2)` | 第 `i` 项依次保存自同构数和可变原子列数。 |
| `matches_offsets` | `int64 (K+1,)` | 第 `i` 项在 `matches_iso` 中的切片边界。 |
| `matches_iso` | `int32 (M,)` | 依照 `matches_shape[i]` 还原的紧凑自同构矩阵。 |

例如两张图的紧凑矩阵分别为形状 `(2,2)` 和 `(1,0)` 时，`matches_shape` 为
`[[2,2],[1,0]]`，`matches_offsets` 为 `[0,4,4]`；第一张图读取
`matches_iso[0:4]` 后还原为 `(2,2)`，第二张图读取空切片后还原为 `(1,0)`。

若一个分子没有会被自同构改变的原子，`matches_shape[i,1]` 为 `0`，对应切片为空；这
不是缺失数据。

### `smiles_assets_v1_report.json`

| 字段 | 类型与示例 | 语义 |
|---|---|---|
| `schema_version` | 整数，例如 `1` | 报告契约版本。 |
| `rdkit_version` | 字符串，例如 `2025.03.3` | 实际构图所用 RDKit 版本。 |
| `exact_smiles_count` | 非负整数，例如 `2148` | 扫描得到的不同精确顶层 SMILES 数。 |
| `accepted_smiles_count` | 非负整数，例如 `2146` | 严格解析并进入两个资产包的 SMILES 数。 |
| `excluded_smiles_count` | 非负整数，例如 `2` | 严格解析失败并被全局排除的 SMILES 数。 |
| `excluded` | 对象列表 | 每项保存原始 `smiles`、解析 `error` 和全部 `sources`。 |

`excluded[*].sources[*]` 保存字符串 `pdb_id`、整数 `occurrence_id`、字符串 `path` 和
从 1 开始的整数 `line_number`。严格解析失败只进入这份报告，不进入图包、身份清单或
前景候选。

```json
{
  "smiles": "无法严格解析的原始字符串",
  "error": "RDKit 返回的解析错误摘要",
  "sources": [
    {
      "pdb_id": "example",
      "occurrence_id": 7,
      "path": "/storage/example/prepared_smiles.jsonl",
      "line_number": 8
    }
  ]
}
```

## 标签与清单字段

### `F2_centered_label.npz`

设一个 PDB 的 centered 候选数为 `N`。以下七个数组都与
`F2_centered.npz:centered_box_index` 的候选轴逐项对齐。

| 字段 | dtype 与形状 | 语义 |
|---|---|---|
| `centered_box_index` | `int32 (N,)` | 候选稳定编号，不等同于数组第一维位置。 |
| `source_blob_index` | `int32 (N,)` | 连接 `F2_blobs.npz:blob_index` 的稳定编号。 |
| `class_index` | `int8 (N,)` | 固定六分类下标；未命中或命中 `other` 时为 `0`。 |
| `matched_occurrence_id` | `int32 (N,)` | 命中的 occurrence 编号；未命中时为 `-1`。 |
| `candidate_coverage` | `float32 (N,)` | 交集体素数除以候选体素数。 |
| `occurrence_coverage` | `float32 (N,)` | 交集体素数除以 occurrence 体素数。 |
| `overlap_score` | `float32 (N,)` | 两个覆盖率乘积的平方根。 |

例如 `centered_box_index=[12,20]`、`source_blob_index=[4,9]` 表示标签第一项连接
centered 编号 12 与 blob 编号 4，而不是连接两个数组的第 12 和第 4 个位置。
`matched_occurrence_id=[7,-1]` 表示第一个候选命中 occurrence 7，第二个候选为背景。
覆盖率字段均在 `[0,1]` 内。

### `manifest.json`

入口文件含 `schema`、固定顺序 `class_names`、固定值
`identity_key="PreparedLigand.smiles"`、固定 `language_model`、`data_root`、
`stage1_root`、`graph_asset_path`、`splits` 和 `summary`。`splits[split]` 分别给出
候选与身份 JSONL 的相对文件名；`summary[split]` 含 `pdb_count`、`candidate_count`、
`foreground_candidate_count`、`identity_count`、`unique_smiles_count` 和
`skipped_positive_candidate_count`。

完整结构示例如下；路径值只是字段示例，不替代正式 manifest 中的绝对路径：

```json
{
  "schema": "smiles_identity_v1",
  "class_names": ["background", "ion", "nucleotide_like", "peptide_like", "small_molecule", "sugar"],
  "identity_key": "PreparedLigand.smiles",
  "language_model": "SMI-TED Light 289M",
  "data_root": "/storage/penghongen/AdaLigand/Ori_Data",
  "stage1_root": "/storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts/Find_1",
  "graph_asset_path": "/storage/penghongen/AdaLigand/Ori_Data/smiles_assets/smiles_graphs_v1.npz",
  "splits": {
    "validation": {
      "candidates": "validation_candidates.jsonl",
      "identities": "validation_identities.jsonl"
    }
  },
  "summary": {
    "validation": {
      "pdb_count": 200,
      "candidate_count": 1600,
      "foreground_candidate_count": 400,
      "identity_count": 300,
      "unique_smiles_count": 210,
      "skipped_positive_candidate_count": 2
    }
  }
}
```

### 候选 JSONL

候选记录字段为 `split`、`pdb_id`、`centered_box_index`、`source_blob_index`、
`stage1_selected`、`class_index`、`matched_occurrence_id`、`matched_smiles`、
`candidate_coverage`、`occurrence_coverage` 和 `overlap_score`。背景候选的
`matched_occurrence_id` 与 `matched_smiles` 都为 `null`，例如：

```json
{"split":"validation","pdb_id":"example","centered_box_index":4,"source_blob_index":7,"stage1_selected":true,"class_index":0,"matched_occurrence_id":null,"matched_smiles":null,"candidate_coverage":0.0,"occurrence_coverage":0.0,"overlap_score":0.0}
```

### 身份 JSONL

身份记录字段为 `pdb_id`、精确 `smiles`、`class_index`、同一身份下全部图合法成员组成的
`occurrence_ids`，以及相对 `data_root` 的最小可用 occurrence
`language_relative_path`。一个 PDB 没有合法
前景身份时可以没有身份记录；其背景候选仍可进入 Dataset。身份记录示例如下：

```json
{"pdb_id":"example","smiles":"CCO","class_index":4,"occurrence_ids":[3,8],"language_relative_path":"stage1_preparation_box_pool_2/ligand_language_models/smi_ted_289m/example/candidate_3.npz"}
```

## 运行时输入

每个候选直接读取 Find_1 `F2_centered.npz` 提供的 48 立方体，不另存 100 维以上的
候选体素缓存。主要张量如下：

| 字段 | 形状 | 内容 |
|---|---|---|
| `voxel_input` | `(B,106,48,48,48)` | 56 个密度派生通道和 50 个受体通道。 |
| `a_tokens` | `(B,N_A,230)` | 四层 A 特征、概率和相对 X/Y/Z。 |
| `p_tokens` | `(B,N_P,116)` | 两层 P 特征、概率和相对 X/Y/Z。 |
| `v_tokens` | `(B,N_V,68)` | 64 维 V 特征、概率和相对 X/Y/Z。 |
| `pp_tokens` | `(B,N_PP,4)` | supplied crop 内的概率和相对 X/Y/Z。 |
| `language_embeddings` | `(B,I,768)` | 当前 PDB 的精确 SMILES 身份语言向量。 |

例如一个物理批次补齐到 `N_A=3`、`N_P=2`、`N_V=2`、`N_PP=4`，且两个样本的
A token 数分别为 3 和 1 时，`a_mask` 为
`[[true,true,true],[true,false,false]]`。`p_mask`、`v_mask` 与 `pp_mask` 使用相同
规则：`true` 对应真实 token，`false` 只对应补齐位置。若同批两个 PDB 的身份数分别为
2 和 1，`identity_mask` 为 `[[true,true],[true,false]]`；身份 softmax 不得读取
`false` 位置。

受体的 `receptor_tokens.npz:feat` 固定为 49 维，Dataset 在内存把独立
`is_backbone` 转成第 50 维。`source_blob_index` 连接 centered 候选与 blob，
`occurrence_id` 只定位真实配体实例；二者不得混用。

## 正式入口

构建资产、标签与 validation 清单：

```bash
bash 训练与运行/sh/build_smiles_identity.sh
```

启动 Find_1 validation 驻留训练：

```bash
bash 训练与运行/sh/train_smiles_identity.sh
```

正式命令保持在两个 shell 文件内。哈希核验、短批吞吐、`strace` 和 checkpoint 组件
诊断属于门禁或一次性执行证据，不得写入正式 YAML 或上述 shell 文件。
