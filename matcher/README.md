# Matcher

本目录实现 AdaLigand Stage2 Matcher。首个正式版本只包含不依赖 Stage1 推理产物的 Anchor 路线、受体原子模态 A 和 O/O′ 输出。

## 当前公开边界

- 数据入口：`AnchorPocketDataset`。候选来自已有 bias/context 池，验证候选关系由正式清单冻结。
- 模型入口：`Matcher`。输入是已装配的配体图、候选 A 图、候选中心 48³ Map、PDB/候选/occurrence 索引和可选辅助标签。
- 目标函数：按配体身份执行 Hungarian，对 O 与 O′ 共用分配；辅助任务只在权重大于零时构造。
- 当前解码：`OOPrimeDecoder`。允许 slot 拒绝和候选复用；不执行 A/B merge/split 推理。
- 当前评估：`OOPrimeEvaluation`。完整验证集上扫描全局 O 阈值，以 occurrence 级 F1 选择 checkpoint。

字段级输入、清单、内存张量与 batch 约束见 `文档/规划文档/Matcher_Anchor_OOPrime数据契约.md`。

模型不得读取 `bias/context` 来源标记。真实 occurrence 配体坐标只生成标签，不能进入模型输入。A 允许为空；零候选 PDB 已在样本收集边界剔除。

Phase2 FinePair 始终计算完整的候选框×occurrence 配对网格。`fine_pair_chunk_size` 只限制一次 forward 中联合计算的配对数，不筛选配对。CUDA BF16/FP16 环境安装 `flash-attn` 后，无 attention bias 的单上下文细注意力会自动使用 varlen Flash 内核；其它调用继续使用 PyTorch scaled-dot-product attention，模型定义不随执行内核变化。

未来 Stage1 推理产物使用新的数据与推理入口，不在 Anchor 文件中加入模式分支。未来 A/B 使用独立 decoder/evaluator，不改变当前 O/O′ 的阈值和正确性定义。

## 配置与运行

正式配置位于 `configs/matcher/`，训练入口为 `python -m matcher.train`。本地或服务器 smoke 产物必须写入 `tmp/`；正式运行通过 `训练与运行/sh/matcher/` 的薄脚本调用项目通用任务提交系统。正式服务器训练需要用户在审阅 YAML 与 `.sh` 后单独授权。

## 显存画像

`ops/profile_matcher_memory.py` 先预热一个完整 Phase2 训练步，再测量第二个 forward、loss、backward 与 AdamW step 的峰值显存。画像只生成临时 JSON，不参与正式训练。普通测量命令为：

```bash
python -m ops.profile_matcher_memory \
  --config configs/matcher/anchor_O_O_prime_phase2.yaml \
  --channels 24,48,72,96 \
  --output tmp/matcher_profile/memory_profile.json
```

需要按人工给定的显存上限验收时使用：

```bash
python -m ops.profile_matcher_memory \
  --config configs/matcher/anchor_O_O_prime_phase2.yaml \
  --channels 24,48,72,96 \
  --memory-limit-gib 72 \
  --output tmp/matcher_profile/memory_profile.json
```

命令末尾接受与 `matcher.train` 相同的 OmegaConf 点号覆盖，可直接调整 `occurrence_budget_per_batch`、两个 chunk size、activation checkpoint 和 bottleneck 参数。省略 `--experiment-manifest` 时，脚本准备正式训练清单的 epoch 0，并测量第一个 occurrence 装箱批次。若需指定重负载批次，可传入只用于画像的小清单；该清单应在 `splits.train` 的每个 PDB 中冻结 `candidate_start_zyx`。清单与输出 JSON 都必须位于 `tmp/`，不能成为正式训练输入。

普通 Dataset 的限时吞吐测量使用：

```bash
python -m ops.profile_matcher_throughput \
  --config configs/matcher/anchor_O_O_prime_phase2.yaml \
  --channels 24,48,72,96 \
  --minutes 60 \
  --memory-limit-gib 72 \
  --output tmp/matcher_profile/normal_throughput.json
```

输出同时记录平均 batch、PDB、occurrence 时间、优化步时间、数据等待时间、峰值显存和是否 OOM。该画像执行正式训练的 Dataset、loss、backward、梯度裁剪与 AdamW，但不运行周期性验证或保存 checkpoint，因此不能直接代表包含验证的完整 epoch 用时。
