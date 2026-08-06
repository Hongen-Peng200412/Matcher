# Matcher

本仓库独立承载 AdaLigand 大项目的 Stage2 Matcher。AdaLigand 负责生产 A–G 与 Stage1 推理产物；Matcher 读取这些正式输入，训练候选—occurrence 匹配模型，并把 checkpoint、指标和推理结果写入独立结果目录。

## 当前实现

正式 Python 包只有 `matcher/`，同时支持两个明确模式：

- `ground_truth_context`：不读取 Stage1 推理产物，每个真实 occurrence 产生一个 GT 候选；当前正式训练、推理和评估使用此模式。
- `stage1_context`：读取 Stage1 F1-centered 候选、V、P、A 与完整概率图，PP 在内存中按概率 top-k 产生；当前已有 Dataset、共同模型前向和真实数据 smoke。

两种模式共享同一模型。Stage1 专属条件不存在时，对应参数分支产生严格零增量；模式二可以从模式一 checkpoint 初始化后继续训练基础参数和新增参数。详细科学契约见 `文档/规划文档/Matcher_双模式ABO实施计划.md` 与 `文档/规划文档/Matcher_双模式数据契约.md`。

## 目录

- `matcher/`：数据契约、两个 Dataset、模型、损失、训练和模式一推理。
- `configs/mode1/`：模式一 Phase1、Phase2 正式配置。
- `训练与运行/sh/mode1/`：正式清单生成和端到端训练入口。
- `与服务器交互/`：只面向 `/home/penghongen/My_Project/Matcher` 的同步入口。
- `文档/`：科学计划、数据契约、执行记录与计划—实现映射。
- `licenses/`：PocketXMol 派生图网络与细配对算子的许可证和来源说明。
- `grill_with_memory/07-28-17-50.md`：本轮 Matcher 决策账本的历史来源。

## 本地验证

```powershell
D:\Anaconda\envs\Pocket_Plus_windows\python.exe -m pytest matcher\tests -q
```

## 服务器边界

- 代码：`/home/penghongen/My_Project/Matcher`
- release、launch 与锁证据：`/home/penghongen/Feedback/Matcher`
- 训练和推理结果：`/storage/penghongen/Matcher/Results`
- A–G 与 Stage1 输入：继续读取 `/storage/penghongen/AdaLigand/...`

安全同步不会删除远端 Matcher 目录；删除式同步只允许人类手动运行。正式 GPU 任务必须在用户审阅最终 YAML、shell 与提交命令后再提交。
