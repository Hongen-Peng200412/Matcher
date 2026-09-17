# Handoff: Matcher 单一 SMILES 主线训练前收口

Date: 2026-09-17

## Current State

Matcher 已把精确顶层 `PreparedLigand.smiles` 固定为唯一身份，阶段 A、B、D 与阶段 C 中不依赖新版 BEST checkpoint 的范围已经完成。活动代码只保留一套 manifest、Dataset、模型、训练和冻结评估管线；旧 `object_key` 身份、六分类、类别概率耦合、身份损失变体、seen/unseen、calibration 和平行 `matcher/smiles_identity` 已从当前树删除，但仍可由 Git 历史恢复。

实现分支 `codex/matcher-single-pipeline-refactor` 保存真实实施历史，`Learn/CUMULATIVE` 保存按人类理解顺序重建的学习历史。两个端点的完整树已核验等价，学习端点完整回归为 23 passed；隔离工作树和临时学习分支已删除，当前只保留 `C:/Users/15919/Desktop/Matcher` 一个工作树。用户修订的 `talk_new/反馈/re-9-16.md` 始终保留为未暂存工作区改动，分支切换前后字节哈希不变，也没有进入任何提交。

## Completed

- validation manifest 的原始候选数为 7,413，已选候选数为 5,554；跳过 10 个不受支持真前景后，训练/验证可消费候选分别为 7,403 和 5,544。
- 全局 SMILES 图包含 3,497 个精确字符串。每个持久 DataLoader worker 只完整载入一次图包，预热后不重复打开图文件。
- 四个正式变体是 `strongest`、`trunk_only`、`density_blob_only` 和 `pocket_only`。模型只有前景二分类头；身份损失是在受支持真前景上的直接 SMILES 交叉熵。
- `density_blob_only` 的 U-Net 只读取一个归一化实验密度通道；`pocket_only` 不读取体素；`trunk_only` 的参数树不含图、清单或门参数。
- manifest 先核对既有固定契约，再发布唯一生成名 JSONL，最后原子替换入口。训练运行目录会保存解析配置和可由 Dataset 直接读取的 manifest 快照。
- 真实受体与 CryoAtom2 test_0 均以 179 个 PDB 为固定分母。`9ian` 与 `9oq7` 的不受支持身份保留原 Stage1 排名和前景评价，但从 SMILES 匹配准确率分母排除，且不能形成 Stage1+Matcher 命中。
- Stage1-only 元数据已写入 `/storage/penghongen/Matcher/evaluation/stage1_only_preview/stage1_only_preview.json`。Emap2lig 原生概率均值排序复现 top-3/4/5 命中数 87/89/90，并在 Gaussian 与概率均值两张预览图中使用同一条单曲线。
- 两张预览图明确命名为 `topk_gaussian_stage1_only_preview.png` 和 `topk_source_probability_mean_stage1_only_preview.png`，没有 Matcher 数值，也不是最终图。
- 本机完整回归为 23 passed，30 个非测试模块导入、compileall、注释标点门禁和 `git diff --check` 均通过。科学逻辑、函数布局/Git、代码注释/YAML 三条独立审查最终均为 PASS。

## GPU Gate Decision

一次性 A100 门禁曾提交为 Job 385011，但始终处于 `PENDING (Priority)`，没有获得节点、没有创建 release、没有执行 Python。用户要求先取消并延后门禁，Job 385011 已精确取消且从队列消失。

阶段 E 正式运行前，应以届时最终代码重新生成一次性 GPU 门禁，至少覆盖四变体真实物理批量的 BF16 前向、反向、clip200 和更新，以及 strongest 的完整 validation。门禁通过后必须使用新的正式启动命令；不得把 Job 385011 或当前 ignored `tmp/matcher_gpu_preflight.*` 当成已执行证据。

## Required Next Actions

1. 等待用户提供本轮代码修改意见，并等待独立任务 `codex://threads/01a099cd-293c-7200-9df3-31ebde48b468` 通知第 6 片推理完成；不要接管或干扰该任务。
2. 只在第 6 片已完成并经只读验收后，把允许的完整训练 PDB 集合冻结为新的 train/validation manifest；不要把未完成分片写入入口。
3. 阶段 E 启动前重新执行 GPU 门禁。通过后从同一冻结 manifest 启动四项正式变体，不把不同 GPU 类型伪装成严格结构对照。
4. 四项自然早停后，使用各自 BEST 保存的阈值执行冻结 test_0；测试条件不得重新拟合阈值。随后生成 Stage1+Matcher 指标、test_1 补充结果、第一张七栏图和两张最终 top-k 图。

## Files To Reopen

- `talk_new/反馈/re-9-16.md`
- `文档/exec_plan/Matcher_单一SMILES主线重构执行记录.md`
- `talk/refactor/Matcher_单一SMILES主线逐文件清单.md`
- `文档/mapping/计划执行映射.md`
- `matcher/data/README.md`
- `matcher/datasets/README.md`
- `matcher/model/README.md`
- `matcher/training/README.md`
- `matcher/evaluation/README.md`
- `configs/experiments/final_smiles/README.md`
