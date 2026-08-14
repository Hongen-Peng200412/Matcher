# 第 5 批单任务记录

本目录保存第 5 批每个独立任务从决定、提交到 validation 收口的完整证据。任务不得重复；每份记录必须包含科学变量、直接对照、配置路径、实现提交、release、完整提交命令、Slurm Job、launch、W&B project/group/run、锁操作、首步验收、validation 结果和最终判断。

当前已决定 6 项、已提交 1 项；双门 `0.02` 任务已通过多个完整 validation 并继续训练。其余五项的配置与本地机械门禁已经完成，正在等待双线 Git、科学正确性与严格技术表达两项独立复审、release 和受保护触发：

- [SMI-TED V 质心 mean/max oracle，双门初值 0.02](SMI-TED_V质心_mean_max_oracle_双门0.02.md)
- [SMI-TED V 质心 mean/max oracle，双门初值 0.1](SMI-TED_V质心_mean_max_oracle_双门0.1.md)
- [SMI-TED V 质心 mean/max oracle，仅图门初值 0.02](SMI-TED_V质心_mean_max_oracle_仅图门0.02.md)
- [SMI-TED V 质心 mean/max oracle，仅清单门初值 0.02](SMI-TED_V质心_mean_max_oracle_仅清单门0.02.md)
- [SMI-TED V 质心 mean/max oracle，裁剪阈值 5](SMI-TED_V质心_mean_max_oracle_裁剪5.md)
- [SMI-TED V 质心 mean/max oracle，裁剪阈值 200](SMI-TED_V质心_mean_max_oracle_裁剪200_近似不裁剪.md)

本目录只记录 validation 驱动的开发任务。正式 calibration 推理和指标必须等待用户在全部迭代结束后的单独确认。
