# Matcher

本仓库独立承载 AdaLigand 的单候选闭集配体身份分类阶段。AdaLigand 生产 A–G 解析产物、密度图和 Stage1 Find_0 候选；Matcher 对每个候选执行六分类，并在真实类别为 `small_molecule` 且当前语言表示可用时，从同一 PDB 的去重配体身份清单中选择身份。

## 当前科学入口

- `candidate_pretrain`：对全部 Find_0 候选预测 `background/ion/nucleotide_like/peptide_like/small_molecule/sugar`，同时监督候选配体区域和受体结合区域；不读取配体语言表示或配体图。
- `full_matcher`：保留相同候选编码器，再读取当前 PDB 的配体语言表示、未池化配体原子图和排除自身的身份清单，输出闭集身份分数。
- 候选与真实 occurrence 的全局组件覆盖率和 occurrence 覆盖率都至少为 `0.30` 才命中。多个命中使用两侧覆盖率几何平均的最大值，并采用稳定并列规则。
- RAUNet 输入候选中心 `48³` 的 105 个通道。A/P token 保留 Stage1 `80³` 上下文且不受 RAUNet 调制；V/PP token 截断到中心 `48³`，只有 V/PP 读取 RAUNet 特征增量。
- 完整 Matcher 支持 `oracle/detached/joint` 三种身份门控损失；三者都以同一 global batch 中真实且身份目标有效的小分子候选数为身份损失分母。

旧的双模式 Matcher 实现已从活动 Python 入口移除，但仍保留在 Git 历史中，不能把旧规划或旧实验结果解释为当前实现。

## 目录

- `matcher/`：冻结清单、Dataset、批处理、105 通道体素缓存、RAUNet、四模态候选聚合、闭集身份模型、损失、指标和正式训练入口。
- `matcher/tests/`：数据、空模态、模型与监督、global-batch 等价、指标、配置、checkpoint 和产物边界回归测试。
- `configs/experiments/initial/`：七个首轮实验的自包含 YAML；一项 A100 顺序运行两个候选六分类阶段，另外六项比较两种语言表示和三种身份损失。
- `训练与运行/`：通用 Slurm、release、launch、四锁控制，以及 Matcher 的预计算和训练 shell 入口。
- `与服务器交互/`：只面向 `/home/penghongen/My_Project/Matcher` 的安全同步入口。
- `文档/exec_plan/`：逐项命令、Job ID、审查发现、临时决定和结果证据。
- `文档/mapping/`：规划、活动实现和历史实现之间的映射。

## 本地验证

```powershell
python -m compileall -q matcher
python -m mypy matcher
python -m pytest matcher\tests -q
git diff --check
```

本地测试不代替服务器真实数据、CUDA 显存和吞吐验收。

## 服务器边界

- 正式源码：`/home/penghongen/My_Project/Matcher`
- release、launch、Slurm 日志和锁：`/home/penghongen/Feedback/Matcher`
- Matcher 清单、缓存、checkpoint、指标与评估结果：`/storage/penghongen/tmp/Matcher/`
- 只读 A–G 与 Stage1 输入：`/storage/penghongen/AdaLigand/...` 和 `/storage/penghongen/AdaLigand_stage1_inference/...`

安全同步不会删除远端 Matcher 目录；删除式同步只允许人类手动运行。正式任务通过 `训练与运行/submit_task.sh` 申请资源，并由 `pre_lock/try_lock/kill_lock/after_lock` 控制开始、复用、终止当前命令和释放 allocation。
