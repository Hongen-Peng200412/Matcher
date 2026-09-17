# Matcher

Matcher 接收 Pocket Plus Find_1 发布的 centered 候选，对每个候选回答两个问题：该候选是否为真实配体前景；若为真实前景，它与当前 PDB 内哪个精确 `PreparedLigand.smiles` 身份匹配。

当前仓库只有一条活动科学主线。精确顶层 SMILES 同时定义监督身份、SMI-TED 语言向量输入和重原子化学图查询键；运行时不读取 LigandObject、`object_key`、seen/unseen、语言家族或另造的图编号。

## 科学契约

- 一个候选与 occurrence 的候选覆盖率和 occurrence 覆盖率必须同时不低于 0.30，才可形成真前景目标。
- 同时命中多个 occurrence 时，选择两侧覆盖率几何平均最大的 occurrence；并列时选择较小的 `occurrence_id`。
- manifest 保存全部 centered 候选及 `candidate_selected`。正式训练默认只读取已选候选；train 和 validation 跳过图或 SMI-TED 身份不受支持的真前景候选，背景候选保留。
- 模型只有一个前景二分类 logit。身份损失是在真前景且身份受支持的候选上直接计算的 SMILES 交叉熵，不乘前景预测概率。
- validation 选模分数为前景 PRAUC 与真实前景候选 SMILES 匹配准确率的算术平均。前景 F1 阈值只在最终 BEST 的 validation 预测上拟合一次并写入 checkpoint。
- 测试任务是 PDB 内给定候选 SMILES 集合的闭集匹配。测试只读取 BEST 保存的前景阈值，不在 test_0 或 test_1 上重新拟合。

## 唯一工作流

1. `ops/matcher/build_smiles_assets.py` 从 PreparedLigand 顶层 SMILES 发布版本化全局图包。
2. `ops/matcher/build_stage1_manifest.py` 扫描一个 Find_1 划分，发布候选 JSONL、PDB 内身份 JSONL 和 manifest。
3. `matcher.datasets.load_manifest_dataset()` 读取冻结 manifest，并按模型变体物化 supplied crop、token、监督目标、语言向量和 SMILES 图。
4. `matcher.training.experiment` 读取一份正式 YAML，从零训练并保存 BEST、LAST、history、summary 和 W&B online 记录。
5. `ops/matcher/evaluate_frozen_test.py` 在阶段 E 使用真实 BEST 生成冻结测试结果；`matcher/evaluation/plotting.py` 只读取冻结 JSON 绘图。

正式 shell 位于 `训练与运行/sh/`。每个 shell 的文件头给出唯一提交命令；release、launch、SHA-256 取证和锁操作不写入正式脚本。

## 四个正式变体

- `strongest`：106 通道体素输入、A/P/V/PP、mean/max 与候选条件注意力摘要拼接、语言主干、SMILES 图增量和同 PDB 身份清单增量。
- `density_blob_only`：U-Net 只读取一个 `exp_clipnorm_nopost` 实验密度通道；保留 V/blob token、V 的 count/mean/max 和四项裁剪统计，不读取模拟密度、受体 hardmask、A/P/PP。
- `pocket_only`：只读取 A token 的原始 50 维 L0 特征，并做掩码 mean/max；不实例化 U-Net 或区域头。
- `trunk_only`：候选侧与 strongest 相同；身份侧只保留 SMI-TED 投影和原生 PairScorer，参数树中没有 SMILES 图、身份清单或对应门参数。

四份活动 YAML 位于 `configs/experiments/final_smiles/`，四个从头训练 shell 位于 `训练与运行/sh/final_smiles/`。除预注册结构边界及 pocket-only 没有区域头所需的区域损失系数外，训练科学字段保持一致。

## 目录

- `matcher/data/`：稳定 schema、密度通道、最大覆盖目标、SMILES 图包和 manifest 读取。
- `matcher/datasets/`：唯一 CandidateDataset、样本与批次契约、批量合并和 DataLoader。
- `matcher/model/`：RAUNet、四变体候选编码、语言/图/身份清单表示和闭集打分。
- `matcher/training/`：配置、损失、训练循环、BEST/LAST、阈值与 W&B。
- `matcher/evaluation/`：validation 指标、Stage1-only 元数据、冻结 test 和纯绘图。
- `matcher/utils/`：不决定科学语义的缓存与 JSON 写入帮助函数。
- `ops/matcher/`：扫描只读来源并发布正式派生产物的薄入口。
- `matcher/tests/`：单一主线的机械回归；测试代码不做学习型细粒度注释。
- `talk/refactor/`：逐文件迁移、删除和最终实施差异清单。
- `文档/exec_plan/`：来源、正式命令、门禁、审查与结果证据。
- `文档/mapping/`：规划、当前实现和历史证据之间的映射。

各正式代码目录和科学产物目录均有本目录级 README；冷读字段形状和特殊值时，应从对应 README 进入，而不是从历史实验记录推断活动契约。

## 本地验证

```powershell
python -m compileall -q matcher ops
python -m pytest -q
git diff --check
```

`pytest.ini` 只收集 `matcher/tests`。本地测试不代替服务器真实 Find_1 batch、CUDA BF16 前后向、完整 validation 或显存门禁。

## 当前阶段边界

阶段 A、B、D 的训练前实现及阶段 C 的 Stage1-only 范围已经落地。当前只生成明确标记为 `stage1_only_preview` 的两张 top-k 预览图；它们不含 Matcher 数值。四项正式训练、Matcher test_0/test_1、第一张七栏指标图和两张最终 top-k 图必须等新版真实 BEST checkpoint 完成后进入阶段 E。

第 6 片 Stage1 推理由独立任务 `codex://threads/01a099cd-293c-7200-9df3-31ebde48b468` 负责。当前代码验收不得接管该任务，也不得把未完成分片写入正式训练 manifest。

## 服务器边界

- 正式源码：`/home/penghongen/My_Project/Matcher`
- release、launch、Slurm 日志和锁：`/home/penghongen/Feedback/Matcher`
- manifest、checkpoint、指标和评估派生产物：`/storage/penghongen/Matcher/`
- 只读 AdaLigand 与 Stage1 输入：`/storage/penghongen/AdaLigand/` 和 `/storage/penghongen/AdaLigand_stage1_inference/`

安全同步不会删除远端 Matcher 目录；删除式同步只允许人类手动运行。正式任务通过 `训练与运行/submit_task.sh` 申请资源，并遵守 `pre_lock`、`try_lock`、`kill_lock`、`after_lock` 控制协议。不得读取 calibration。
