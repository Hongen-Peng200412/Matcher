# 联合身份 v4 摘要拼接：seed 3408、LR 1e-4、A100 零门 clip1

- 任务：A100、seed 3408 的摘要拼接零门结构锚点；它是 A100 2×2 矩阵的双门 0、clip1 单元。
- 受管 Job：340893，A100，1 GPU，16 CPU。前序 v4 seed3408、LR=1e-4 任务已自然早停；本任务已触发并完成首个 global step。
- 配置：configs/experiments/joint_identity_v4/joint_identity_v4_concat_zero_gate_seed_3408_lr_1e4_oracle_a100.yaml。
- release：/home/penghongen/Feedback/Matcher/releases/Matcher_c225fbd2feee/Matcher。
- 科学字段：联合身份 v4、SMI-TED、oracle、all_non_background、selection_seed=3407、training.seed=3408、LR=1e-4、双身份门 0、clip=1、BF16、物理批量 8、梯度累积 4、global batch 32、从零初始化。
- 摘要结构：mean/max 与候选条件注意力分别产生 D_m 维表示；2D_m 拼接向量经零输出残差后加回 mean/max 主表示。
- 动态命令：/home/penghongen/Feedback/Matcher/allocations/340893/run_cmd_340893.sh，SHA-256 26c5489f8e5edf788749d074bac1fc7c35b3108110aefff730334be69958e961；执行当前 release 的 训练与运行/sh/train_experiment.sh 与上述唯一 YAML。
- launch：/home/penghongen/Feedback/Matcher/launches/340893/train_experiment_job340893_20260815T192321_a6/。
- W&B：entity pencounkdual-111，project AdaLigand-Matcher-JointIdentityV4-SummaryConcat，group joint-identity-v4-summary-concat-a100-seed-3408，run ID 39c2ca9k，[运行链接](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-JointIdentityV4-SummaryConcat/runs/39c2ca9k)。
- 运行根：/storage/penghongen/tmp/Matcher/experiments/joint_identity_v4_concat_zero_gate_seed_3408_lr_1e4_oracle_a100/train_experiment_job340893_20260815T192321_a6/。
- 前序完成证据：train_experiment_job340893_20260815T122610_a5，BEST/last=7/9，global_step=4470，selection=0.6918678973623833，W&B l0m0ao2i。
- 门禁：正式解析器 40 个计划/41 个阶段通过；摘要拼接专项 7 passed；配置科学与技术表达独立终审 PASS；控制器两项独立终审完成后方可删锁。
- 一次性控制器：ignored tmp/joint_identity_v4_audit/activate_v4_concat_matrix_free_jobs.sh，冻结 SHA-256 为 3e2bbccb5f774bfd5d72e2cc0c93232d5e5527eeae542f9738927c4c80ab64d4；运维摘要不进入正式包、YAML 或 resolved.json。
- 首步：global batch 32，identity_eligible_count=10，计算 23.7735677540 s，裁剪前梯度范数 53.4079589844，峰值 CUDA allocated/reserved 为 14,211,529,216/19,719,520,256 bytes。
- 20:34 进度：已完成 epoch 0–1 两次完整 validation，最新 epoch 1 刷新当前 BEST；selection=0.6403636782824584，六分类 macro-F1、固有身份 Top-1、条件身份 Top-1、宽松端到端 F1 为 0.5063940107920865/0.8028649386084584/0.8151487826871056/0.5995482413681833。sample_count/identity_eligible_count 保持 6495/1466，任务继续训练。
- 22:26 进度：已完成 epoch 0–4；epoch 2 把 BEST 刷新到 `0.6747906069882211`，epoch 3 形成一个坏验证轮，epoch 4 又以 selection=`0.6803945989195906` 刷新当前 BEST。任务继续训练，不把该中途 BEST 当作结构终局。
- 最终结果：2026-08-16 03:43:32（Asia/Shanghai）因 patience 自然早停；BEST/last=`12/14`、运行根 global step=`6705`、BEST selection=`0.7009100312219069`。BEST validation 的 `sample_count/identity_eligible_count=6495/1466`；六分类 macro-F1、固有身份 Top-1、条件身份 Top-1、宽松端到端 F1 分别为 `0.5245595969237634/0.8717598908594816/0.9000861326442722/0.6780843640976848`，small-molecule 三项为 `0.763302752293578/0.8297872340425532/0.6842105263157895`，ligand/binding micro Dice 为 `0.30782459442935933/0.08424026613112988`。
- 锁与释放：runner 通过 `/storage` try_lock 替身正常恢复锁；核对运行根 summary、阶段 summary、BEST/LAST、W&B、Job 资源与四锁后，只删除精确 `after_lock_340893`。Job 随后从 Slurm 队列正常退出，逻辑 try_lock 符号链接由 runner 清理，已失效的 `/storage` 目标文件也在确认 Job 退出后精确删除。没有使用 scancel，没有触碰 pre/kill，没有读取 calibration。
