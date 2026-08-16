# 第 5 批 SMI-TED V 质心 mean/max oracle 学习率 2e-4

- 科学任务：五类非背景完整 Matcher；SMI-TED Light 289M；V 质心 48³；mean/max；oracle。
- 配置：`configs/experiments/batch5/batch5_smi_ted_v_centroid_lr_2e4_oracle_a100.yaml`。
- 唯一科学变量：`learning_rate=2.0e-4`；第 2 批直接基线为 `1.0e-4`。
- 直接对照：第 2 批 `batch2_smi_ted_v_centroid_oracle_a100`，BEST selection score `0.6981048482`。
- 固定条件：`closed_set_v3`、双门 `0.0/0.0`、`gradient_clip_norm=1.0`、seed `3407`、从零初始化、A100 物理批量 8、梯度累积 4、global batch 32，其余模型、数据、损失和选模字段与直接基线逐项相同。
- 动机：第 2 批 oracle 的早期 `2,684` 次更新审计中，裁剪前梯度范数有 `99.93%` 超过阈值 `1.0`；因此本任务回答“强裁剪条件下把 AdamW 初始步长翻倍”是否改善 validation，而不是声明全局最优学习率。第 5 批原六项已全部提交，双门 `0.02` 又自然早停，Job `340899` 成为唯一受管空闲 A100；用户同时要求尽量保持六张卡都有具名任务。
- 判断方式：继续用六分类 macro-F1、精确子类条件身份 Top-1 和宽松五类端到端 F1 的等权平均 selection score 选 BEST，并报告五类固有身份、small-molecule 兼容端到端、两个区域 Dice、非有限值、裁剪前梯度范数和裁剪触发率。单随机种子的约 `1%–2%` 改善只作为后续多随机种子验证线索。
- 预定资源：受管 A100 Job `340899`；16 CPU；物理批量 8、梯度累积 4、global batch 32；不设置训练或 Slurm 时间上限。
- W&B：online project `AdaLigand-Matcher-Batch5`；group `matcher-batch5-learning-rate`；run ID 在受保护触发后回填。
- 状态：该 v3 候选最终没有触发、没有 launch 或 W&B run。独立科学复审发现候选工作树继承联合身份 v4 运行时后，用户决定不建立专用 v3 分支，把正式问题改成 v4 内的 `1e-4/2e-4` 三对复现；本文件只保留未执行方案及其阻断历史。
- 本地开发位置：分支 `codex/matcher-batch5-lr-2e4`；隔离工作树 `C:/Users/15919/Desktop/Matcher_batch5_lr_2e4`。该工作树只保存配置候选，不能直接创建训练 release。后续需要选择“从已验收的 v3 第 5 批端点建立独立历史契约运行分支”或“在当前 v4 主线实现显式 v3/v4 选择”；两者均不得把 v4 数据契约悄悄混入本任务。

## 提交前门禁

- `matcher/tests/test_metrics_and_config.py -k batch5`：`7 passed`；该回归把七份第 5 批 YAML 分别与第 2 批 oracle 做结构化比较，并只允许任务声明的门初值、裁剪阈值、A800 资源适配或学习率差异。
- mypy：改动测试文件零问题；`git diff --check` 通过。
- 严格技术表达复审为 PASS；科学正确性复审因运行时会切到 v4 判为 FAIL。只有 v3 运行时来源、双线 Git 处置、内容寻址 release 和受保护激活全部明确后才可触发 GPU。
- calibration 不进入配置、训练、选择或验收；全部比较只使用 validation。

## 提交与结果

- 结果：未执行，因此没有实现 release、动态命令、Slurm launch、W&B run、首步或 validation 结果。正式 v4 学习率任务见 `../联合身份v4/学习率复现/README.md`。
- 锁边界：只允许在全部共同门禁通过后删除共享 allocation 根中的 `try_lock_340899`；`after_lock_340899` 必须保留，`pre_lock_340899/kill_lock_340899` 不得创建、删除或触碰。
