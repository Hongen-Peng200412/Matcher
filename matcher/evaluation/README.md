# evaluation：冻结指标、test 连接与纯绘图

本目录把训练期 validation 与阶段 E 的冻结 test 分开。test 只读取 BEST 中保存的前景阈值，绝不在真实受体或 CryoAtom2 上重新拟合。

## 指标

- 前景 PRAUC：全候选前景二分类 average precision，同分预测共同处理；
- 前景 F1：只使用 BEST 在 validation 上保存的阈值；
- SMILES 匹配准确率：只在真前景且图、SMI-TED 均受支持的候选上统计；
- 五类分层：`ion`、`nucleotide_like`、`peptide_like`、`small_molecule`、`sugar`；零支持返回 `None`，图中显示 `N/A (n=0)`。

`9ian` 与 `9oq7` 的严格图和语言资产均不受支持。它们保留前景二分类真值和 Stage1 原始排名，但不进入 SMILES 准确率分母，也不能产生 Stage1+Matcher 命中。

## Stage1-only 预览

`test_set.py` 从 Find_1 真实受体、Find_1 CryoAtom2 和 Emap2lig 正式只读产物重算 k=1..20 曲线。Stage1-only 命中定义是：top-k 内至少一个候选与任一真实 occurrence 的候选覆盖率和 occurrence 覆盖率均不低于 0.30。

Find_1 只在已冻结 `candidate_selected=True` 集合内分别按 Gaussian `candidate_score` 和 `source_probability_mean` 排序。Emap2lig 保留自己的候选集合，按实例内原生 ligand probability 均值排序；每张图只放一条 density-only、receptor-independent 曲线。两套来源不能解释为逐候选配对。

预览元数据状态固定为 `stage1_only_preview`，图文件名也必须含该字符串。它不含 Matcher 数值，不能称为最终图。

## 阶段 E 冻结 test

同一个 BEST 对真实受体与 CryoAtom2 的 test_0 各前向一次。每个候选用 `(pdb_id, source_blob_index)` 连接排序事实；同一份 Matcher 预测同时生成两种 Find_1 排序。Stage1+Matcher top-k 不使用前景阈值：只要唯一几何目标是受支持真前景，且 argmax SMILES 与 `target_smiles` 逐字相同即命中。

test_1 不重新前向。评估器读取官方 149-PDB 列表，从 test_0 冻结候选预测中筛选并重算补充指标。

`plotting.py` 只读取冻结 JSON：两张 preview、两张最终 top-k 和七栏指标图均不加载模型、不读取 checkpoint、不重新计算科学事实。
