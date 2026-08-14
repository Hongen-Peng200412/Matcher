# 第 2 批 V 质心候选条件注意力六分类

- 科学任务：候选六分类预训练；未增强 candidate-conditioned cross-attention；不读取语言、配体图或身份监督。
- 配置：`configs/experiments/batch2/candidate_v_centroid_cross_attention_a800.yaml`。
- 直接对照：第 0 批未增强候选条件交叉注意力；改变局部视野并加入 12 项模态统计和 4 项采样统计。
- 资源：受管 Job `340895` 的一张 A800；物理批量 16、梯度累积 2、global batch 32；`after_lock_340895` 保留。
- W&B：project `AdaLigand-Matcher-Batch2`，group `matcher-batch2-candidate-v-centroid-sampling`。
- 代码与 release：远端发布源逐文件核对为第 2 批冻结端点，实际复用 release `Matcher_3cc75993d69e`。
- launch：`train_experiment_job340895_20260814T092853_a4`。
- W&B run：[`12oeeshn`](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-Batch2/runs/12oeeshn)。
- 状态：训练中；W&B online 与首个 global batch 32 更新均已通过，最终 validation 待自然早停。

控制端使用下列完整命令执行受保护接续；脚本 SHA-256 为 `092a61bdd10e301bd9f23c05cbe269340cdddfeb7358156d37692e23d00093c4`：

```powershell
& "$env:USERPROFILE\.codex\tools\Invoke-ProjectSsh.ps1" -Command 'bash -s' -InputFile 'C:\Users\15919\Desktop\Matcher\tmp\batch2_candidate_handoff_20260814\activate_a800_candidates.sh'
```

脚本在删除任何锁前核对两项 A800 allocation、旧动态命令、9 个第 2 批源码/配置哈希、目标目录、正式配置语义以及 voxel `14,312/6,495`、PP `480/181` 缓存计数。为本任务安装的动态命令为：

```bash
exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/train_experiment.sh" \
    --plan configs/experiments/batch2/candidate_v_centroid_cross_attention_a800.yaml
```

2026-08-14 09:31:09（Asia/Shanghai）在 mean/max 动态命令先安装后，只删除共享 allocation 根下的 `try_lock_340895`；动态命令 SHA-256 为 `295046288c5f67fa12aa289073bf8b1fcde0bde76c4ff7c98e93098d76db706d`。首步事件发生在有限性检查、参数更新与 CUDA 同步之后：计算 `61.9103197092 s`，身份监督数 `0`，裁剪前梯度范数 `81.2901382446`，峰值已分配/保留显存 `28,107,427,328 / 39,634,075,648 bytes`。候选预训练身份监督数为零是配置契约，不是缺失数据。

2026-08-14 09:56:15 完成 epoch 0 validation：macro-F1 与 selection score 均为 `0.4935270492`。按 `background/ion/nucleotide_like/peptide_like/small_molecule/sugar` 顺序的 F1 为 `0.8972378029/0.6613691932/0/0/0.7750385208/0.6275167785`；混淆矩阵为 `[[4239,518,0,0,200,66],[29,541,0,0,6,0],[0,0,0,0,3,0],[0,0,0,0,6,2],[40,1,0,0,503,7],[118,0,0,0,29,187]]`。训练数据等待 `10.3 s`、计算 `1198.9 s`，validation `390.7 s`；该轮是首轮改善，任务继续运行。
