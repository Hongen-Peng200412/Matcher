# 第 1 批单实验：SMI-TED 五类 joint

- 预定配置：`configs/experiments/batch1/full_smi_ted_all_foreground_joint_a100.yaml`
- 预定资源：受管 Job `340893` 的 1×A100；保留 `after_lock_340893`。
- 个性化变量：五类非背景身份、12 维固化手动特征、mean/max、joint、物理批量 8 × 累积 4。
- 直接对照：第 0 批没有 SMI-TED joint；只与本批 oracle/detached 比较，不虚构缺失基线。
- 提交命令、release、launch、W&B run、首步和最终结果：等待代码冻结与动态命令放行后填写。
