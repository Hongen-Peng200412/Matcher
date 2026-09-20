# density_only-0 实验日志

本记录保存 `density_only-0` 的单实验追溯。候选侧只读取 candidate-centered 单通道实验密度；身份侧使用 SMI-TED、SMILES 图和同 PDB 身份清单。

## 当前状态

- 阶段：本地实现与机械门禁进行中，尚未触发服务器 GPU 门禁或正式训练。
- 正式资源：计划顺序复用 A800 数组任务 `379403_1`；锁与反馈目录使用基础编号 `379403`，活动阶段使用 15 个非持久 DataLoader worker。
- 稳定实验标识：`density_only-0`；代码模型变体：`density_only`。
- 正式 YAML：`configs/experiments/final_smiles/density_only.yaml`。
- 冻结 train/validation manifest：`/storage/penghongen/Matcher/manifests/final_train_validation_shards01_06_v1/manifest.json`；SHA-256 `3551e4a7b111c9d51a718fe11586769f31a7a50f66b8d25839bb9742418ad247`。
- 候选输入：`experimental_density_48` 经 0.1%/99.9% 分位截断与全网格 z-score 后形成 `(1,48,48,48)`；不读取或构造 V、A、P、PP、模拟密度、受体栅格和候选人工统计。
- 模型边界：使用无末端多尺度卷积的 RAUNet；最高分辨率特征的空间 mean/max 经投影直接形成候选表示；不建立 token 投影器、体素调制器、候选模态聚合器或区域头。
- 损失权重：前景 `0.30`、ligand-area `0.00`、binding-area `0.00`、SMILES 身份 `1.00`。
- 选模公式：`(0.3 * foreground_PRAUC + supported_foreground_SMILES_accuracy) / 1.3`。

## 正式命令

`bash 训练与运行/sh/final_smiles/train_density_only.sh`

全测试集与 Stage3 小分子正式命令将在 BEST 冻结后绑定实际 checkpoint，并与测试和门禁命令分开记录。

## 测试、门禁和只读核查

- `D:\Anaconda\python.exe -m pytest -q`：`42 passed in 32.88s`。
- `D:\Anaconda\python.exe -m compileall -q matcher ops`：通过。
- 两个活动训练脚本经 Git Bash `bash -n`：通过。
- `density_only.yaml` 与 `pocket_only.yaml` 经 `load_smiles_experiment()` 严格解析：实验标识、模型变体、15 个 worker、损失权重和 patience 均符合第二轮协议。
- 面向本轮拥有文件的 `git diff --check`：通过；用户已有的三个未提交 Python 文件不属于本轮改动或提交范围。

实际 A800 GPU 门禁尚未触发；本节命令不得当作正式训练命令。

## 失败或中断尝试

当前没有失败或中断尝试。后续若出现重试，继续追加在本节，不新建替代日志。
