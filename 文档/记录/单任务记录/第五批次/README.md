# 第 5 批单任务记录

本目录保存第 5 批每个独立任务从决定、提交到 validation 收口的完整证据。任务不得重复；每份记录必须包含科学变量、直接对照、配置路径、实现提交、release、完整提交命令、Slurm Job、launch、W&B project/group/run、锁操作、首步验收、validation 结果和最终判断。

本批正式执行的六项均已自然早停并回填 BEST/last 与 selection score。原第七项 v3 学习率 `2e-4` 候选没有触发；用户随后把正式学习率问题改为联合身份 v4 内的三个随机种子配对，其运行记录位于 [联合身份 v4 学习率复现](../联合身份v4/学习率复现/README.md)。六个 Slurm allocation 与全部 `after_lock` 均继续保留，随后已经安全接续六项 v4 学习率任务：

- [SMI-TED V 质心 mean/max oracle，双门初值 0.02](SMI-TED_V质心_mean_max_oracle_双门0.02.md)
- [SMI-TED V 质心 mean/max oracle，双门初值 0.1](SMI-TED_V质心_mean_max_oracle_双门0.1.md)
- [SMI-TED V 质心 mean/max oracle，仅图门初值 0.02](SMI-TED_V质心_mean_max_oracle_仅图门0.02.md)
- [SMI-TED V 质心 mean/max oracle，仅清单门初值 0.02](SMI-TED_V质心_mean_max_oracle_仅清单门0.02.md)
- [SMI-TED V 质心 mean/max oracle，裁剪阈值 5](SMI-TED_V质心_mean_max_oracle_裁剪5.md)
- [SMI-TED V 质心 mean/max oracle，裁剪阈值 200](SMI-TED_V质心_mean_max_oracle_裁剪200_近似不裁剪.md)
- [未执行的 v3 学习率 2e-4 历史候选](SMI-TED_V质心_mean_max_oracle_学习率2e-4.md)

本目录只记录 validation 驱动的开发任务。正式 calibration 推理和指标必须等待用户在全部迭代结束后的单独确认。
