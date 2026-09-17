# data：冻结数据契约与纯转换

本目录定义 Matcher 唯一活动身份：精确的顶层 `PreparedLigand.smiles`。`object_key`、沉积坐标、`present` 掩码和模型内部字符串都不参与身份。目录内代码不扫描 Stage1 文件夹；扫描和发布由 `ops/matcher/` 完成。

## 文件职责

| 文件 | 职责 |
|---|---|
| `schema.py` | 固定类别报告顺序、张量宽度和正式来源目录 |
| `labels.py` | 用候选和 occurrence 稀疏体素计算唯一最大覆盖目标 |
| `density.py` | 构造 106 通道完整密度输入或 1 通道实验密度输入 |
| `smiles_graph.py` | 每个工作进程一次载入 `smiles_graphs_v1.npz`，按精确 SMILES 取图 |
| `manifest.py` | 读取并核验 `matcher_stage1_manifest_v1`，不补建缺失记录 |

## 最大覆盖目标

候选对每个 occurrence 同时计算候选覆盖率和 occurrence 覆盖率。两者均不低于 `0.30` 才是有效前景匹配；多个匹配依次按几何平均较大、`occurrence_id` 较小选出唯一目标。没有有效匹配的候选是背景。

类别名称只用于真值分层：`background`、`ion`、`nucleotide_like`、`peptide_like`、`small_molecule`、`sugar`。模型只预测一个前景 logit，不预测六分类。

## 全局 SMILES 图包

`smiles_graphs_v1.npz` 以精确顶层 SMILES 为唯一逻辑键。它只保存由 SMILES 决定的字段：

- `element (N_atom,)`：原子序数；
- `charge (N_atom,)`：形式电荷；
- `atom_in_ring (N_atom,4)`：3、4、5、6 元环标志；
- `bond_index (2,N_bond)`：RDKit 重原子顺序下的无向键端点；
- `bond_type (N_bond,)`：single、double、triple、dative、aromatic；
- `bond_in_ring (N_bond,4)`：化学键的 3、4、5、6 元环标志。

图包不含坐标，也不引入 `graph_id`。`SmilesGraphStore.initialize()` 把压缩包完整复制到当前进程内存并关闭文件；持久 DataLoader worker 后续样本只访问内存。

## manifest 稳定字段

`manifest.json` 记录来源根、图包路径、划分文件和计数。每个候选 JSONL 记录来源编号、两个 Stage1 排序分数、选择掩码、唯一几何目标与覆盖率。每个身份 JSONL 记录 `pdb_id`、精确 `smiles` 和 SMI-TED 相对路径。

真前景可以因严格 SMILES 图或 SMI-TED 缺失而没有受支持身份。训练和 validation 读取时整条候选跳过；test 保留原候选及排名，并令 `identity_target=-1`。背景候选始终保留。

## 单密度通道

`build_experimental_density_channel()` 只读取 supplied 实验密度，沿用 0.1%/99.9% 分位截断和 z-score，返回 `(1,Z,Y,X)`。它不读取模拟密度、受体 hardmask 或 50 维受体栅格。
