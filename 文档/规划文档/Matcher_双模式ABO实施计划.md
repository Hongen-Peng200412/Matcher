# Matcher 双模式 A/B/O 实施计划

本文规定新版 Stage2 Matcher 的当前目标行为。旧 `matcher/` 及其 Anchor O/O′ 训练只作为已经完成的工程 smoke，不是本文要求兼容的生产接口。新版代码位于 `matcher_v2/`，首先完整训练、推理和评估不依赖 Stage1 产物的 `ground_truth_context`，同时实现能够读取 Stage1 产物的 `stage1_context`。

## 目标与范围

新版 Matcher 接收一个 PDB 的配体身份及 occurrence 数量、候选受体区域和实验密度，在 occurrence 级预测候选与配体 slot 的匹配。模型必须满足：

- `ground_truth_context` 输入为配体、GT ligand-area 的 10 Å 受体包络 A，以及以配体几何中心为中心的 48³ 实验密度 BOX；不读取 Stage1 推理产物。
- `stage1_context` 输入为配体、预测 blob 的 10 Å 受体包络 A、48³ 实验密度 BOX、blob 体素及 `voxel_final` 特征 V、Stage1-Find 已落盘伪原子 P，以及由完整图概率 top-k 产生但不单独落盘的 PP。
- 模式二从模式一 checkpoint 初始化后继续训练基础参数与新增参数。缺少 Stage1 专属 V、P 和概率派生 PP 时，新增条件分支显式返回零增量，网络执行模式一的基础路径；不要求模式二训练后的基础参数与原模式一 checkpoint 数值相同。
- 模式一保留由密度 U-Net 生成同生态位采样点的能力边界，但当前正式版本不启用该路径。此前已经决定暂缓 `MapPointHead → MapPointBuilder`；在点身份及其监督尚未敲定前，本次模式一 checkpoint 只使用 `MapSummaryHead`。模式二 PP 的点身份由 Stage1 完整图概率确定，因此本轮可以安全启用多尺度特征采样。
- 本轮把 `ground_truth_context` 做到正式训练、推理与评估；`stage1_context` 做到正式 Dataset、模型前向和真实产物 smoke，不在本轮等待全量 Stage1 产物后再启动训练。

## 监督对象

设候选 ligand-area mask 为 $b_i$，真实 occurrence 的 ligand-area mask 为 $g_o$，候选中心与 occurrence 配体原子几何中心的距离为 $d_{i,o}$，单位 Å：

$$
A_{i,o}=\frac{|b_i\cap g_o|}{|g_o|},\qquad
B_{i,o}=\frac{|b_i\cap g_o|}{|b_i|},\qquad
O_{i,o}=\frac{1}{1+d_{i,o}/(1\ \mathrm{Å})}.
$$

连续标签均位于 $[0,1]$。与之并行监督的硬标签为：

$$
A'_{i,o}=\mathbb 1[A_{i,o}\ge 0.10],\qquad
B'_{i,o}=\mathbb 1[B_{i,o}\ge 0.10],\qquad
O'_{i,o}=\mathbb 1[d_{i,o}<10\ \mathrm{Å}].
$$

代码变量使用 `A_target`、`B_target`、`O_target` 与 `A_prime_target`、`B_prime_target`、`O_prime_target`。连续三项使用独立预测头和回归损失；硬标签使用独立 logit 和 gamma=2 的 focal loss。第一版损失权重只是可配置起点，不视为科学最优值。

`ground_truth_context` 的候选 mask 直接使用该候选 occurrence 的 GT ligand-area mask，不伪造标签。候选与自身 occurrence 的 A/B 为 1；不同 occurrence 的 mask 若真实重叠，则对应 A/B 保留实际的中间值。一个 PDB 的候选数等于其 occurrence 数，不再模拟 miss、split、bias 或 context 假阳候选。

## 模型边界

模型保持旧讨论中的粗分支、细分支、slot 和身份内匈牙利思想，但不要求逐类复制旧 Anchor 顶层代码。

1. 配体图与候选 A 图采用同构建图规则、不同权重的 GatedGCN，并允许更新边表示。
2. `MapBackbone` 只输出 48³ 实验密度的四尺度特征；`MapSummaryHead` 产生 `Map_repr`。当前模式一基础路径不选择采样点，模式二 PP 在既定坐标处从四尺度特征采样。
3. 每个候选的 `BOX_repr` 由当前可用的候选级表示拼接后经 MLP 得到，不把不同表示展开成统一 token 序列。
4. 粗分支保持配体 slot、`BOX_repr`、`A_repr`、`Map_repr` 等多类表示并存。每层执行配体—候选侧交互及同类集合交互；候选内部的 typed attention 不放在每层循环中。
5. 未塌缩实体只在存在对应实体时回吸。A 与 occurrence 可以回吸；无未塌缩实体的 `BOX_repr`、`Map_repr` 不回吸。
6. `stage1_context` 通过少量显式模块把 V、P、概率派生 PP 形成 `BOX_repr` 候选级增量，把 A 的概率与 L1–L3 特征形成逐原子实体增量。当前不为 V/P/PP 更新未塌缩实体。缺少条件时直接返回形状正确的零张量或跳过增量模块，不依赖零输入穿过带 bias 的 MLP 来近似归零。
7. 粗细两阶段仍允许使用 4+4 Block；最终层数、通道和分块由真实显存画像调整，配置不得硬编码在模型正文。

## Stage3 复用边界

`matcher_v2/` 必须让 Stage3 Builder 无需读取 Matcher Dataset、训练器、匈牙利匹配或 A/B/O 预测头即可复用以下能力：

- 48³ 密度归一化与旋转后的坐标约定；
- `MapBackbone` 的四尺度特征和 `MapSummaryHead`；
- 把完整图概率限制到当前 BOX 后选择 top-k 坐标，并在给定坐标处从 U-Net 四尺度特征组装初始点特征；
- `PocketInput` 与 `Stage1Context` 中 V、P、PP、A 的模型无关内存契约。Stage3 怎样把这些实体注入 PocketXMol 由 Stage3 自己决定，不复用 Matcher 专属 `BOX_repr` 增量。

本轮不提前实现完整 Stage3，也不规定 Stage3 必须怎样把密度信息注入 PocketXMol。未来 Stage3 的固定基线仍需尽可能忠实复现 PocketXMol；任何密度增强在关闭时必须严格退化为该基线。

## 工程入口与产物

- Python 包：`matcher_v2/`。
- 当前契约：`matcher_v2/README.md` 与 `文档/规划文档/Matcher_双模式数据契约.md`。
- 正式配置：`configs/matcher_v2/`。
- 模式一正式人类入口：`训练与运行/sh/matcher_v2/mode1/`。
- 可复用 Stage1 产物清单生成工具：`ops/matcher_v2/`；正式 JSON 不写入临时目录，生成过程的大型中间状态和 smoke 产物写入任务专属临时目录。
- 执行记录：`文档/exec_plan/Matcher_双模式ABO端到端实施.md`。

正式训练使用 AdamW、BF16 autocast、在线 W&B，并在一个 epoch 内支持五次均匀间隔验证。批次继续按 occurrence 数装箱；预算、Map 分块、细配对分块和 U-Net 通道均由 YAML 显式配置。

## 验收

- 两个 Dataset 的数据入口、清单和后处理彼此独立；新增模式不通过大量条件分支污染另一模式。
- 同一个模型 checkpoint 可以运行两种输入；Stage1 专属输入缺失时，所有相应增量逐元素严格为零。
- 合成测试覆盖 A/B/O 与 A′/B′/O′、GT mask 真重叠、同身份多 occurrence、旋转、无 Stage1 条件和带 Stage1 条件。
- 真实 `ground_truth_context` PDB 完成 Dataset、前向、反向、checkpoint、推理和评估 smoke。
- 真实 Stage1 F1-centered 产物完成 `stage1_context` Dataset 与模型前向 smoke。
- 服务器画像记录平均 batch/PDB/occurrence 时间、数据等待、优化步、峰值显存与 OOM 状态。
- 正式模式一训练启动后由当前对话的 heartbeat 监控；旧 Job 335493 按用户授权通过锁协议终止其旧训练并复用 A800 allocation。
