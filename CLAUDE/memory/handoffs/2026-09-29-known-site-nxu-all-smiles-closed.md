# Handoff：nxu 固定小分子查询点的全部 SMILES 输入比较收口

日期：2026-09-29

## 当前状态

Matcher 已知位点身份测试的本轮范围只含冻结的 nxu 84 PDB／357 个小分子 occurrence 查询点。真实受体和 CryoAtom2 两条件的原小分子输入 Matcher／EMERALD-ID 结果只读复用；新增两个 Matcher 模型的全部类别身份输入，并从同一组完整 logits 派生不掩码与小分子掩码答案。三次新增正式前向、两份汇总 JSON、八张图和结果文档均已完成。全配体 NxU 不启动、不恢复，也没有正式全配体准确率。

## 已完成

- 原清单仍在服务器 `/storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest/nxu/`；新派生身份轴只新增 `manifest_all_smiles/nxu/pdbs.jsonl`。84 个 PDB 的原小分子身份槽共 114 个，全部类别 164 个；同一 PDB 内无精确 SMILES 跨类别反例。输入文件快照在 `provenance/all_smiles_input/input_files.jsonl`。清单和快照 SHA-256、Job、release、launch 见[独立任务日志](../../../test_真实场景(Structure)/执行记录/nxu全部SMILES输入.md)。
- 正式代码位于 `test_真实场景(Structure)/known_site/`：`build_all_smiles_manifest.py` 只建立完整 PDB 身份轴；`predict_matcher.py` 仍读取原 357 点清单，在完整身份轴上一次前向，保存逐槽 `scored_smiles`、`identity_logits`、不掩码和掩码答案；`summarize_all_smiles.py` 从冻结逐点文件计算固定、自身完整、七模式共同完整三个分母并绘图。正式短脚本位于 `训练与运行/sh/known_site/`，门禁和来源快照只在 `test_真实场景(Structure)/smoke/`。
- 三个正式前向均通过实际 A100 GPU 门禁：density_only-0／真实受体，以及 density_only-106／真实受体、CryoAtom2。各 357 点均有 `scored` JSON；身份轴、logit 长度和两种读出逐点复算均无不一致。density_only-0 从代码和输入源确认不读取受体文件，故 CryoAtom2 复用同一正式完整身份预测，不运行第四次前向。两个模型都从自己的训练 release／BEST 恢复模型定义；没有训练或读取 calibration。
- [完整结果页](../../../测试结果与追溯/测试结果总览/nxu全部SMILES输入比较.md)给出两条件的七模式表、固定 357 点主指标、各方法完整可用子集、真实受体 190 点与 CryoAtom2 222 点共同可用配对，以及八张图。服务器原件在 `all_smiles_comparison/{real,CryoAtom2}/`；本地 `测试结果与追溯/测试结果总览/nxu全部SMILES比较图片/` 的两份 JSON 和八张 PNG 与服务器逐文件 SHA-256 一致。
- 唯一新增 A100 allocation Job `390871` 已在绘图成功、精确 `try_lock` 恢复和无残留模型/汇总进程后，仅删除自身精确 `after_lock`；Slurm Job 与 batch 均 `COMPLETED 0:0`。旧真实受体、CryoAtom2 和被暂停的 NxU 产物都没有覆盖。

## 科学决定与解释边界

- “小分子输入”“全部输入不掩码”“全部输入小分子掩码”是 Matcher 的三种模式；EMERALD-ID 始终只接收原小分子身份集合。配对表统一的是同一批查询点，不是候选身份范围相同。
- 掩码只读同一次全部身份 logits，不重新编码候选。即使掩码后的总正确数与原小分子输入恰好相同，也不能认定逐点预测相同，因为额外身份可影响模型的候选编码和库存交互。
- Matcher 自身完整要求 PDB 候选全集有图与语言向量；EMERALD-ID 自身完整要求该点全部小分子候选有有效官方评分。失败点始终保留在 357 点固定分母。横轴只按冻结 PDB 的小分子精确 SMILES 去重数，单身份组的部分 100% 是闭集定义性质。
- NxU 旧真实受体数组先前安全中断，完整结果 317/995、未完成 678 点；这些旧文件保留，但本轮既不重新启动也不把未完成点当评分失败。用户已明确本轮小分子结果收口，NxU 等另行通知。

## 后续行动

若未来重新讨论 NxU，必须先取得用户新指示，并只读核查旧数组、精确锁、317 个已完成结果及未完成点；不能直接从本轮 nxu 清单推导全配体分母。若只查看本轮结果，先读结果页，再按独立任务日志中的正式 GPU 命令、门禁命令和绘图命令追溯各 release／launch；不要从原 Job 的任务名推断每次动态执行的内容。

## 需要重开的文件

- `测试结果与追溯/测试结果总览/nxu全部SMILES输入比较.md`：七模式表、八图和科学解释。
- `test_真实场景(Structure)/执行记录/nxu全部SMILES输入.md`：清单、输入快照、逐次 Job、release、launch、正式命令和资源退出。
- `test_真实场景(Structure)/执行记录/总日志.md` 与三个方法各自日志：共享状态及按实验归属的结果。
- `test_真实场景(Structure)/known_site/README.md`：当前清单、逐点字段、分母与图的可冷读契约。
- `test_真实场景(Structure)/参考材料 & 想法/Matcher_EMERALD-ID 比较计划书.md`：当前批准的 nxu 科学边界，NxU 只作为暂停范围。
