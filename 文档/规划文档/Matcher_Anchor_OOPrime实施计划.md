# Matcher Anchor O/O′ 实施计划

## 文档职责

本文规定“不依赖 Stage1 推理产物”的 Matcher 首个正式版本如何实现、验证和运行。科学与工程决定的最高优先级来源是 `grill_with_memory/07-28-17-50.md` 的 Question 1–105；本文把这些决定整理成执行顺序，不重新解释或改写它们。历史规划与最新决定冲突时，以该决定账本为准。

本轮只实现 `AnchorPocketDataset`、受体原子模态 A、O/O′ 两阶段 Matcher、当前 O/O′ 解码与评估。未来消费 Stage1 推理产物的数据入口和 A/B 解码评估必须作为新的明确实现加入，不能通过当前数据入口中的模式分支、注册器或预留空字段接入。

## 交付边界

本轮完成以下正式能力：

1. 从冻结的 Stage1 新版 train/validation 清单生成 max-slots 为 100 的 Matcher 实验清单；训练候选逐 epoch 采样，验证候选冻结落盘。
2. 实现 `AnchorPocketDataset`、中心 48³ Map、18 Å A、配体与 A 的同构图契约、以 occurrence 数为预算的整 PDB 装箱。
3. 实现完整小型 3D U-Net、Map 摘要、两套独立图编码参数、8 个 MatcherBlock、粗分支和 O/O′ 深监督。
4. 实现 Phase1 与 Phase2 的严格 checkpoint 接力、Phase2 独立 stem、可选零初始化 FiLM_plus、最终细分支及两类辅助监督。
5. 实现 Hungarian 目标、训练损失、独立一致性诊断、O/O′ 解码、阈值扫描、checkpoint 选择、恢复和单卡训练循环。
6. 完成本地单元测试、真实数据 smoke、A100/A800 隔离 GPU smoke 和完整 Phase2 显存画像。
7. 形成正式 YAML 与薄 `.sh`。正式服务器训练提交前停下，请用户审阅并明确授权。

不在本轮实现的内容包括 Stage1 推理产物数据入口、P/V/PP 模态、Map 点生成、A/B 输出与解码、Stage3 位置更新、PairFormer 和通用多模态插件框架。

## 当前冻结契约摘要

### 数据与装箱

- occurrence 的一次互斥采样概率为 miss/split/hit = 0.30/0.20/0.50，对应 0/2/1 个 bias 候选；center 不使用。
- context 数从 `0..floor(2.0 * N_occurrence)` 离散均匀采样。18 Å 球内无 A 的 context 候选有 80% 概率跳过，并继续补抽至目标数或候选池耗尽。
- 全部真实 occurrence slot 始终输入模型。零候选 PDB 只在样本收集边界跳过，后续代码假定候选数大于零。
- O 的正标签为候选中心到 occurrence 质心距离小于 12 Å；O′ 在 24 Å 内为 `1 / (1 + d / 6 Å)`，24 Å 外只约束预测值不高于 0.2。
- 训练 Map 在同一 PDB 内共享一次 90° 立方体旋转；验证和推理不旋转。
- `occurrence_budget_per_batch=64`，按完整 PDB 贪心装箱；超过 64 的 PDB 独占一个 batch。所有正式损失固定除以 64。

### 模型与两阶段训练

- Map 输入是 80³ 候选的中心 48³。MapBackbone 是完整 encoder/decoder，小型初始通道为 `[16,32,48,64]`；正式通道必须由完整 Phase2 训练步显存画像决定。
- 配体与 A 使用同一图字段和 GatedGCN 算法，但参数不共享。节点/边状态维度为 128/64，表示维度为 256；Phase1 和 Phase2 各含 4 个图更新 Block。
- 粗状态为 `CCD_repr/BOX_repr/A_repr/Map_repr`。每个 Block 含两层已冻结微观结构的 coarse layer，完成实体回吸后再由 BOX 读取同候选的 A/Map。
- 每个 Block 有独立 `CoarsePredictionHead`。Phase1 的 Block 1–4 含实体更新，Block 5–8 为 coarse-only；Phase2 冻结 Block 1–4，并为 Block 5–8 新建从原始图出发的独立 stem 和实体路径。
- Phase2 只在 Block 8 运行 ligand↔A 的双向只读细分支。细配对按 2048 对分块并使用 activation checkpoint；零初始化 adapter 把细表示加入 Phase1 已训练的 Block 8 粗配对表示，继续使用同一个 OHead/OPrimeHead。
- `EntityAuxiliaryHead` 在 Phase1 读取 Block 4、在 Phase2 读取 Block 8；`FinePairAuxiliaryHead` 只在 Phase2 Block 8。权重分别为 0.1/0.1，权重 0 是唯一关闭方式。

### 训练、验证和推理

- 每个有效 Block 出口独立执行身份内 Hungarian；同一出口的 O/O′ 共用联合分配。O 使用无 alpha、gamma=2 的 focal 与 Dice；O′ 使用精确值或远距离单边截尾。
- 深监督辅助出口共同分享固定 0.5 权重并按深度线性递增；最终有效出口权重为 1.0。
- 每阶段使用单参数组 AdamW，学习率 `5e-5`、weight decay `0.01`；调度是 0.5% 线性 warmup 后按验证 F1 plateau 降学习率，第三次实际降低后结束，最长 20 epoch。
- 每个 epoch 完整验证 5 次。验证在 0.00–1.00、步长 0.01 的 O 概率阈值上调用同一 `OOPrimeDecoder`，以 occurrence 级最高 F1 选 checkpoint；并列选择更高阈值。
- 当前 decoder 允许 slot 拒绝、候选复用，不在推理时运行 Hungarian。O/O′ 一致性仅作无梯度诊断，不参与训练或 checkpoint 选择。

## 实施顺序与验收

### 阶段一：契约与数据

建立正式实验清单、Anchor 数据入口、图与 Map 张量、装箱和确定性随机协议。使用真实 `10ad` 与 validation 样本核对字段、世界坐标、候选采样和标签。

### 阶段二：主模型

先完成 Map、图层和粗分支，再完成 Phase1 前向、八出口预测和损失。玩具数据测试必须覆盖空 A、同身份 slot 置换、深监督权重和固定 64 分母。

### 阶段三：Phase2 与细监督

完成独立 stem、可选 FiLM_plus、冻结边界、细分块和两个辅助头。严格加载测试只允许已声明的 Phase2 新参数缺失；梯度测试必须覆盖冻结参数无梯度、Map 仍有梯度以及后四 Block 得到梯度。

### 阶段四：训练与运行

完成训练循环、每 epoch 五次验证、checkpoint 阈值元数据、恢复与 Phase1→Phase2 接力。随后运行本地测试、服务器隔离 smoke 和显存画像。正式 `.sh`/YAML 经用户审阅授权后才提交完整训练。

## 文档与历史

- 活动执行证据持续写入 `文档/exec_plan/Matcher_Anchor_OOPrime端到端实施.md`。
- 当前代码接口写入 `matcher/README.md`；输入产物与模块阅读入口分别写入 `talk/Matcher_输入与产物概览.md` 和 `talk/Matcher_模块结构概览.md`。
- 计划、执行日志和实现关系登记在 `文档/mapping/计划执行映射.md`。
- Git 使用实现历史与学习历史双线。实现稳定且端点等价后，才推进 `Learn/CUMULATIVE`。
