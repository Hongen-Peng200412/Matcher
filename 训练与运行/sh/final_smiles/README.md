# final_smiles 正式训练与冻结评估脚本

第二轮活动训练脚本是 `train_strongest.sh` 与 `train_blob_only.sh`。其余训练和评估脚本保留历史冻结实验的可恢复入口。每个脚本自包含，文件头给出提交器命令；release、launch、SHA-256 和一次性服务器取证只由任务记录保存，不写入正式脚本。

| 脚本 | 正式作用 |
| --- | --- |
| `train_strongest.sh` | 在 H100 上从头训练 strongest-1 |
| `train_blob_only.sh` | 在 A800 上从头训练 blob_only-0 |
| `train_density_blob_only.sh` | 从头训练 density_blob_only |
| `train_pocket_only.sh` | 从头训练 pocket_only |
| `train_trunk_only.sh` | 从头训练 trunk_only |
| `evaluate_strongest.sh` | 用 strongest 冻结 BEST 生成正式 test_0、test_1 与三张图 |
| `evaluate_density_blob_only.sh` | 用 density_blob_only 冻结 BEST 生成正式 test_0、test_1 与三张图 |

训练入口读取同一份冻结 train/validation manifest。第二轮冻结测试脚本要在训练自然停止后绑定各自真实 BEST，再读取同两份 test_0 manifest 和同一份 Stage1-only 预览元数据；测试不读取 calibration，也不重新拟合前景阈值。

