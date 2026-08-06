# Matcher 模块结构概览

## 冷读顺序

建议按以下顺序阅读首版 Matcher：

1. `matcher/README.md`：当前正式数据与模型接口。
2. `matcher/anchor_data.py`：Anchor 路线的样本生成、Map/A/配体装配。
3. `matcher/batching.py`：以 occurrence 为预算的 PDB 装箱和批量合并。
4. `matcher/graph.py` 与 `matcher/map.py`：两类底层特征提取。
5. `matcher/model.py`：8 个 Block、Phase1/Phase2、粗分支、细分支和预测头。
6. `matcher/objectives.py`：Hungarian、O/O′ 与辅助损失。
7. `matcher/consistency.py` 和 `matcher/oo_prime.py`：不反传诊断、当前路线解码与评估。
8. `matcher/train.py`：单卡训练、验证、checkpoint、恢复和两阶段入口。

## 依赖方向

`anchor_data.py` 和 `batching.py` 不导入模型。`graph.py`、`map.py` 不读取盘上文件。`model.py` 只消费已经装配的张量，不判断数据来自 Anchor 还是未来 Stage1。`objectives.py` 消费模型原始输出与标签。`oo_prime.py` 只负责当前 O/O′ 解码和验证统计，不读取 Dataset 文件。`train.py` 是唯一把这些正式组件连接成训练流程的入口。

未来 Stage1 路线应新增明确命名的数据与推理文件，并直接调用稳定的模型、目标函数或叶子算子；不能让 `anchor_data.py` 增加 `mode` 分支，也不能把当前 decoder 改成通用注册系统。

## 可读性约束

- 每个长期模块只保留真实职责入口；只转发一次或只映射一行字段的函数直接内联。
- 不创建 `utils.py/common.py/base.py`、DataModule、Trainer、Callback、Provider、Registry 或 Factory 层级。
- 训练主调用链直接展示：读一份 YAML → Anchor Dataset → 装箱 → Matcher → loss → O/O′ evaluation → checkpoint。
- 外部文件和配置只在入口集中校验；模型内部信任 Dataset/Collator 已建立的契约。
- 临时分析、smoke 和画像产物进入 `tmp/<任务名>/`，不能混入正式清单、配置或运行目录。
