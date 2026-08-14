# Handoff: Matcher 第 5 批门控因子矩阵与裁剪对照准备

Date: 2026-08-15

## Current State

活动 Goal 继续覆盖 Matcher 第 1–5 批、validation 头对头、六张受管 GPU、W&B 分批管理和三层记录。当前 Git 分支为 `codex/matcher-batch5-factorial-clip`，基点是 `Learn/CUMULATIVE@f559368a0d058ded70cb7619e1839925dcdd0f60`；新增配置、测试和记录尚未形成实现提交或学习端点。

第 5 批当前为 6 项已决定、1 项已提交。双门 `0.02/0.02` oracle 正在 A100 Job `340899` 运行，已完成 epoch 0–3 四次完整 validation 且连续改善，当前 BEST selection score 为 `0.6773211521`。新增 graph-only、inventory-only、双门 `0.1`、clip5、clip200 五项已建立 YAML、配置回归和逐任务记录，目标 Job 依次为 `340893/340898/340897/340894/340895`；在双线 Git、独立双审、release 和受保护控制器全部通过前不得触发。

第 4 批三项均已自然早停：detached、joint、oracle 分别在 Job `340895/340894/340897` 收口并恢复 `try_lock`。六个 allocation 均必须保留，全部 `after_lock` 不得删除，不得取消、重启或释放。当前没有 heartbeat。

calibration 清单与缓存可作非评估准备，但在全部迭代结束且用户再次明确确认前，不得执行 calibration 模型推理、读取正式指标或用其选择方案。

## Completed

- 2026-08-15 02:29 只读重扫已完成 W&B 历史：第 2 批 oracle run `6soiol5e` 共 6,264 个有限更新，裁剪前范数最大值 `182.8500213623`；阈值 `1.0` 在其中 6,245 次更新生效。
- 双门任务 epoch 3 的 graph/inventory/合计增量相对语言基线中心化 L2 比例为 `0.2174344/0.0457968/0.2068664`；该尺度不作有符号收益解释。
- 独立科学审查建议用零门 `00`、graph-only `10`、inventory-only `01`、双门 `11` 构成完整因子矩阵，并把裁剪对照独立为同一 A800 上的 `5/200`。
- 新增五份 Batch5 YAML；最新快照的六项参数化配置回归 `6 passed`，全套 pytest `131 passed`，mypy 检查 45 个 Python 文件零问题，compileall、pyflakes、28 份正式 YAML 的 29 个阶段解析和 diff-check 均通过。
- 第 4 批 joint 最终 BEST/last epoch `7/9`、selection score `0.6960616522`；相对第 2 批 joint 的主要分类与身份指标总体小幅改善，区域指标一降一升。
- 第 4 批 oracle 最终 BEST/last epoch `7/9`、selection score `0.6976087078`；分类与身份近似持平，ligand/binding Dice 下降 `9.962%/18.706%`。第 4 批不支持把候选预训练初始化作为跨损失默认改进。
- Batch4 joint/oracle 与 Batch5 当前证据已经回填逐任务、批次、总记录、ExecPlan 和 mapping。

## Decisions

- 新增门任务使用 A100 `340893/340898/340897`，保持 `8×4=32`；前两项补齐 `0.02` 的 2×2 四格，Job `340897` 的双门 `0.1` 是尺度探针。新增裁剪任务使用 A800 `340894/340895`，保持 `16×2=32`。
- `clip5` 与 `clip200` 是同硬件主要对照。只有实际 `preclip_norm > 200` 的比例不超过 `0.1%` 时，后者才称近似不裁剪；必须报告总步数、触发数、最大范数和最小缩放系数。
- 当前不提交 `LR=2e-4`；双门任务仍连续改善，调度器已有自动降学习率机制，同时改变学习率会削弱裁剪归因。
- 当前六项继续冻结 `closed_set_v3`。联合身份 `(object_key, PreparedLigand.smiles)` 另升 schema/release，先只用 train/validation 做严格等价审计，再建立新契约零门 oracle 基线。
- 若联合身份没有造成 train/validation 同 PDB 身份拆分，Dataset 张量、固定权重前向，以及评估中的损失、科学指标、样本数和身份有效数除版本元数据外必须精确一致；墙钟耗时与吞吐不作逐值等价要求。若有拆分，未受影响 PDB 仍精确一致，受影响 PDB 逐项报告槽、SMILES、occurrence、target 与监督数变化。
- mean/max 与 candidate-conditioned attention 摘要向量拼接等待其它实验和重构收口；正式批次脚本转正、目录 README、全库注释循环和 `matcher/data` 进一步分层属于后续基础设施，不阻塞当前低编号实验。

## Open Questions

- 双门任务自然早停后，`11` 相对 `00/10/01` 的主效应与交互是否稳定？当前四轮不能替代最终 BEST 比较。
- `clip5/clip200` 的实际触发比例、最小缩放系数、稳定性和完整 validation 是否支持放宽默认裁剪阈值？
- 联合身份 v4 在 train/validation 中实际拆分多少 PDB、身份槽和有效监督候选？用户给出的 17 个全量数据多 SMILES 组不能直接当作 train/validation 结果。
- 第 4 批已收口并显示候选预训练初始化不具备跨三种损失稳定收益；该变量不再叠加进当前第 5 批。

## Next Actions

1. 若独立复审引起代码或配置修改，重跑相称门禁；否则沿用 Completed 中记录的最新冻结证据。
2. 使用现有科学与严格表达代理复核冻结候选；阻断清零后完成实现提交、按人类理解顺序重建学习分支并核验端点等价。
3. 安全同步、创建内容寻址 release，建立只管理 Job `340893/340898/340897/340894/340895` 的 Git 忽略受保护控制器；独立审查后才删除对应 `try_lock`，所有 `after_lock` 保留。
4. 启动后逐项回填 release、控制命令、launch、W&B run、锁边界和首个 global batch；随后只在完整 validation、BEST、自然早停或失败事件回填。
5. 五项稳定运行后，开始联合身份 v4 的 train/validation 清单与等价门禁；不得访问 calibration。

## Files To Reopen

- `文档/记录/目标草案.md`
- `文档/记录/总记录.md`
- `文档/记录/批次记录/第5批次/第5批证据驱动优化.md`
- `文档/记录/单任务记录/第五批次/`
- `文档/exec_plan/Matcher_单候选闭集身份分类重写执行记录.md`
- `文档/mapping/计划执行映射.md`
- `configs/experiments/batch5/`
- `matcher/tests/test_metrics_and_config.py`
- `训练与运行/README.md`
