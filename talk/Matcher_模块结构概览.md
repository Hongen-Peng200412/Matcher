# Matcher 模块结构概览

## 冷读顺序

建议按以下顺序阅读首版 Matcher：

1. `matcher/README.md`：当前正式数据与模型接口。
2. `matcher/contracts.py`：两个模式共同使用的内存样本与批次字段。
3. `matcher/data_ground_truth.py` 与 `matcher/data_stage1.py`：两个显式 Dataset；二者不共享盘上扫描或推理后处理。
4. `matcher/batching.py`：以 occurrence 为预算的 PDB 装箱和批量合并。
5. `matcher/graph.py`、`matcher/map.py` 与 `matcher/context.py`：图网络、密度骨干和 Stage1 条件叶子。
6. `matcher/model_core.py` 与 `matcher/model.py`：共享科学算子、两阶段装配和六个预测头。
7. `matcher/objectives.py`：身份内 Hungarian、六头主损失与辅助损失。
8. `matcher/train.py`：双模式装配、模式一训练与验证、checkpoint 和恢复；`matcher/infer_ground_truth.py` 只负责模式一推理与评估。

## 依赖方向

两个 Dataset 和 `batching.py` 不导入模型。`graph.py`、`map.py`、`context.py` 不扫描盘上训练清单。模型只消费 `contracts.py` 定义的内存对象；`objectives.py` 消费模型原始输出与标签。模式一解码只位于 `decode_ground_truth.py`，模式二未来增加独立推理入口。`train.py` 是唯一把正式 Dataset、模型、目标函数和验证连接成训练流程的入口。

模式二未来新增明确命名的训练验证和推理文件，不能把模式一 decoder 改成通用注册系统。

## 可读性约束

- 每个长期模块只保留真实职责入口；只转发一次或只映射一行字段的函数直接内联。
- 不创建 `utils.py/common.py/base.py`、DataModule、Trainer、Callback、Provider、Registry 或 Factory 层级。
- 训练主调用链直接展示：读一份 YAML → 明确 Dataset → 装箱 → Matcher → loss → 对应模式 evaluation → checkpoint。
- 外部文件和配置只在入口集中校验；模型内部信任 Dataset/Collator 已建立的契约。
- 临时分析、smoke 和画像产物进入 `tmp/<任务名>/`，不能混入正式清单、配置或运行目录。
