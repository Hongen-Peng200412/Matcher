# 联合身份 v4 摘要拼接实验

本目录记录 mean/max 候选摘要与候选条件注意力摘要拼接后的完整 Matcher 实验。两种聚合器分别产生 D_m 维表示，拼接后的 2D_m 维表示只进入零输出残差；初始输出逐值等于 mean/max 主路径。实验只使用 train/validation，calibration 仍禁止读取和推理。

- [seed3407、LR 1e-4、A800 零门 clip1](seed3407_lr1e-4_a800_零门clip1.md)：已自然早停；BEST/last=`0/2`，selection=`0.6475402763159072`，相对同硬件 mean/max 基线降低 `6.434098%`。
- [seed3407、LR 1e-4、A800 双门 0.02、clip1](seed3407_lr1e-4_a800_双门0.02_clip1.md)：2026-08-15 22:06 安全触发；launch `train_experiment_job340894_20260815T220601_a14`、W&B run `8e9ufq3b`，首个 global batch 32 已完成。
- [seed3407、LR 1e-4、A800 零门 clip200](seed3407_lr1e-4_a800_零门clip200.md)：已自然早停；BEST/last=`7/9`，selection=`0.6915842525509563`；4470 次更新中裁剪触发 0 次。
- [seed3407、LR 1e-4、A800 双门 0.02、clip200](seed3407_lr1e-4_a800_双门0.02_clip200.md)：已自然早停；BEST/last=`7/9`，selection=`0.6956405636500063`；4470 次更新中裁剪触发 0 次，Job 340895 已安全释放。
- [seed3408、LR 1e-4、A100 零门 clip1](seed3408_lr1e-4_a100_零门clip1.md)：已自然早停；BEST/last=`12/14`，selection=`0.7009100312219069`；Job 340893 已按锁协议安全释放。
- [seed3408、LR 1e-4、A100 双门 0.02、clip1](seed3408_lr1e-4_a100_双门0.02_clip1.md)：已自然早停；BEST/last=`7/9`，selection=`0.6979373129440473`。
- [seed3408、LR 1e-4、A100 零门 clip200](seed3408_lr1e-4_a100_零门clip200.md)：已自然早停；BEST/last=`7/9`，selection=`0.6968598986464777`；4470 次更新中裁剪触发 0 次。
- [seed3408、LR 1e-4、A100 双门 0.02、clip200](seed3408_lr1e-4_a100_双门0.02_clip200.md)：已自然早停；BEST/last=`6/8`，selection=`0.6903778565864686`；4023 次更新中裁剪触发 0 次。

摘要拼接八项覆盖 A800、seed3407 与 A100、seed3408 两套双身份门 {0, 0.02} × 裁剪阈值 {1, 200} 完整 2×2。八项全部固定 LR=1e-4、物理批量 8×累积 4、global batch 32；A100 与 A800 结果只用于检查方向复现，不能冒充跨硬件逐位同轨对照。

用户随后把收尾范围固定为五项，不再采用条件解锁，也不允许五项之外继续扩展：

- [第 1 项：seed3407、A800 摘要拼接双门0.02、clip200](seed3407_lr1e-4_a800_双门0.02_clip200.md)：已自然早停；BEST/last=`7/9`，selection=`0.6956405636500063`，Job 340895 已安全释放。
- [第 2 项：seed3407、A800 mean/max 零门、clip200](seed3407_lr1e-4_a800_meanmax_零门clip200.md)：已自然早停；BEST/last=`3/5`，selection=`0.6889974930159317`；2682 次更新中裁剪触发 0 次，Job 340894 已安全释放。
- [第 3 项：seed3408、A100 mean/max 零门、clip200](seed3408_lr1e-4_a100_meanmax_零门clip200.md)：已自然早停；BEST/last=`7/9`，selection=`0.6944095969666253`；4470 次更新中裁剪触发 0 次，Job 340897 已安全释放。
- [第 4 项：seed3409、A100 摘要拼接零门、clip200](seed3409_lr1e-4_a100_摘要拼接_零门clip200.md)：已自然早停；BEST/last=`11/13`，selection=`0.7057373297694586`；6258 次更新中裁剪触发 0 次，Job 340898 已安全释放。
- [第 5 项：seed3409、A100 mean/max 零门、clip200](seed3409_lr1e-4_a100_meanmax_零门clip200.md)：已自然早停；BEST/last=`7/9`，selection=`0.6925343658028632`；4470 次更新中裁剪触发 0 次，Job 340899 已安全释放。

五项全部自然收口后，训练主线转入结果汇总、最优变体推荐、注释与双线 Git 整理，并按四锁协议安全释放六个受管 allocation。每项仍须保留 release、受保护触发、launch、W&B、首步和最终结果证据；只使用 train/validation，不读取 calibration。

五项现已全部自然收口，六个受管 allocation 也已全部按锁协议安全释放。三组零门 clip200 严格结构配对中，摘要拼接相对 mean/max 的 selection 变化为 `+0.375438%/+0.352861%/+1.906471%`，三组平均 `+0.878257%`；方向一致但幅度存在明显 seed 差异。训练主线不再新增实验，转入完整结果分析、推荐脚本与 Git/工作树整理。

A800、seed3407 摘要拼接 2×2 已完整收口。零门 clip1、双门0.02 clip1、零门 clip200、双门0.02 clip200 的 selection 依次为 `0.6475402763/0.6939342114/0.6915842526/0.6956405637`。门值与 clip200 各自都能修复异常弱的零门 clip1 轨迹，但组合增益强烈不具加和性；组合单元相对同硬件、同 seed 的 mean/max 零门 clip200 只高 `0.964165%`。A100、seed3408 的对应矩阵中门值和 clip200 方向均为负，因此这些超参数效应不能跨硬件、跨 seed 稳定外推。

2026-08-16 的第 2 项触发遭遇 `/home` Lustre OST 写入阈值满。失败尝试没有进入训练首步；空 launch、缓存和旧日志 inode 均按可恢复方式取证或迁移。Job 340894 的活动 launch 树随后安全迁至 `/storage`，六个 Job 的 try_lock 也使用 `/storage` 目标符号链接作为自然结束保险。正式 a241 训练已完成首步；该基础设施处置不改变 release、YAML、模型、数据、随机种子或训练科学字段。
