# final_smiles 正式训练脚本

四个训练脚本分别启动四个且仅四个 Matcher 变体。两个冻结测试脚本分别使用 strongest 与 density_blob_only 的正式 BEST；每个脚本自包含，文件头给出提交器命令。release、launch、SHA-256 和一次性服务器取证由任务记录保存，不写入正式脚本。

| 脚本 | 正式作用 |
| --- | --- |
| `train_strongest.sh` | 从头训练 strongest |
| `train_density_blob_only.sh` | 从头训练 density_blob_only |
| `train_pocket_only.sh` | 从头训练 pocket_only |
| `train_trunk_only.sh` | 从头训练 trunk_only |
| `evaluate_strongest.sh` | 用 strongest 冻结 BEST 生成正式 test_0、test_1 与三张图 |
| `evaluate_density_blob_only.sh` | 用 density_blob_only 冻结 BEST 生成正式 test_0、test_1 与三张图 |

当前脚本是阶段 E 的正式入口。训练入口读取同一份冻结 train/validation manifest；冻结测试入口读取各自训练 run 的 BEST、同两份 test_0 manifest 和同一份 Stage1-only 预览元数据，不在测试集重新拟合阈值。

