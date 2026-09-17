# datasets：manifest 到模型批次

本目录只有一个公开数据入口：

```python
load_manifest_dataset(
    manifest_path,
    split=...,
    selected_only=...,
    mode=...,
    variant=...,
    pp_top_k=...,
    cache_gib_per_worker=...,
)
```

`selected_only=True` 只保留 Stage1 的 `candidate_selected=True` 候选。`mode=train` 和 `mode=validation` 跳过身份资产不完整的真前景；`mode=test` 保留它们，不改变原始候选排名。

## 文件职责

| 文件 | 职责 |
|---|---|
| `candidate_data.py` | 按模型变体只解压实际需要的 centered 字段，构造单个候选 |
| `contracts.py` | 定义 `CandidateMetadata`、`CandidateItem` 和 `CandidateBatch` |
| `batching.py` | 可复现采样、token 补齐、图连接和持久 worker 初始化 |

## 四种候选输入

| 变体 | 实际读取的候选输入 |
|---|---|
| `strongest` | `voxel_input (106,48,48,48)`、A 230、P 116、V 68、PP 4、12 项模态统计和 4 项裁剪统计 |
| `density_blob_only` | `voxel_input (1,48,48,48)`、V 68、V 三项统计和 4 项裁剪统计 |
| `pocket_only` | 原始 `A_feat_L0 (N_A,50)`；不读取 U-Net、V、P、PP 输入 |
| `trunk_only` | 候选侧与 `strongest` 相同；身份侧不读取图包 |

区域目标是 `bool (48,48,48)`。区域目标所需的 occurrence 和受体坐标是监督来源，不会因此进入 `density_blob_only` 的模型输入。

## 身份轴与批处理

同一 PDB 内的受支持精确 SMILES 按字符串排序，形成长度 `I` 的闭集身份轴。语言张量是 `(I,768)`；训练目标是该轴下标。批处理把身份轴补到 `I_max` 并生成 `identity_mask (B,I_max)`。

图分支启用时，每个有效身份连接一张图，`graph_to_sample` 和 `graph_to_identity` 把图表示写回 `(B,I_max)`。`trunk_only` 的图张量为空，参数树中也不存在图与身份清单增量模块。

DataLoader 使用 `spawn` 和持久工作进程。每个 worker 启动时完整载入一次图包；PDB centered、受体、标签和语言向量受每 worker 的字节预算最近使用缓存约束。
