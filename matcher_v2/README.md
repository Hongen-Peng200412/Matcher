# Matcher v2 代码边界

`matcher_v2` 实现两个显式数据模式和一套共享网络。`ground_truth_context` 不读取
Stage1 推理产物；`stage1_context` 读取 Find 的 F1-centered 产物。两种模式只在
`train.py` 的 `build_dataset` 处二选一，不使用动态注册表或通用多模态框架。

## 两个数据入口

- `data_ground_truth.py`：每个真实 occurrence 产生一个候选，读取真实 ligand-area、
  10 Å A 和中心 48³ 实验密度。正式训练、推理和评估使用此入口。
- `data_stage1.py`：按正式 JSON 指针读取预测候选、V、P、A 和完整图概率；PP 只在
  内存中从当前 48³ 内的概率 top-k 产生。该入口与模式一的清单和后处理互不依赖。

模式二可以从模式一 checkpoint 开始继续训练基础参数和新增参数。V、P、PP、A
条件不存在时，新增候选增量和 A 节点增量不经过带偏置网络，而是直接保持逐元素零。
这保证网络执行模式一基础路径；模式二训练后不要求基础参数仍等于旧 checkpoint。

## 模型与当前范围

`model.py` 在已经通过旧 Matcher smoke 的图、粗分支、FinePair 和 FiLM_plus 算子上，
增加六个 A/B/O、A′/B′/O′ 头以及 Stage1 条件增量。复用主干避免在仓库内复制一份
一千余行的同构实现；`matcher_v2` 的公开输入仍只有 `contracts.py` 中的中性类型。

当前正式模式一只让 `MapBackbone` 和 `MapSummaryHead` 进入网络，不启用 U-Net 自行
预测或选择采样点。此前讨论已明确暂缓 `MapPointHead → MapPointBuilder`；在没有敲定
点的选择监督前，不把它偷偷加入本次 checkpoint。模式二 PP 已经使用确定的 Stage1
完整图概率作为点身份，因此可以安全地从同一次 U-Net 的四尺度特征取得初始表示。

## Stage3 可直接复用的叶子

Stage3 应直接导入以下叶子，而不应导入 Dataset、Matcher、Hungarian、六个预测头或
训练器：

- `data_common.centered_crop`：固定中心、越界零填充的密度裁剪；
- `context.MapBackbone`、`context.MapFeatures`、`context.MapSummaryHead`：四尺度密度特征
  与候选摘要；
- `context.topk_probability_points`：把完整图概率限制到当前 BOX 后选择 top-k 坐标；
- `context.DensityPointBuilder`：在给定局部坐标处采样四尺度特征；
- `contracts.PocketInput`：密度、A 图和候选几何的模型无关内存边界；
- `contracts.Stage1Context`：模式二 V、P、PP、A 的模型无关内存边界；
- `matcher.graph` 中的同构建图叶子，以及 `matcher.model` 中已验证的 GatedGCN、
  FinePair 和 FiLM_plus 科学算子。

当前训练增强只旋转密度张量。同一 PDB 的全部候选使用同一个 90° 立方体旋转。
当前配体图和 A 图只以距离等旋转不变量参与网络，因此世界坐标无需同步改变；以后
若模式一启用空间采样点，点坐标必须与密度执行同一旋转后才能采样 U-Net 特征。
当前没有把这项尚未启用的坐标旋转预先包装成公共函数；Stage3 若启用相同增强，必须
先明确定义局部坐标变换并增加对应测试。

## 运行与后处理

模式一正式入口位于 `训练与运行/sh/matcher_v2/mode1/`。训练支持断点续训、在线
W&B、BF16、每 epoch 五次验证、`events.jsonl` 和原子 checkpoint。模式一后处理只在
`decode_ground_truth.py` 与 `infer_ground_truth.py` 中实现；未来模式二使用独立推理
入口，不在这些文件里加入数据来源条件分支。
