# 任务脚本目录

本目录保存 Matcher 正式任务的人类入口脚本。每个脚本应自包含地说明所用项目入口、配置、数据位置、结果位置和必要环境变量；不要把某次作业编号或临时目录固化为默认值。

完整使用说明见 [训练与运行说明](../README.md)。

## 精确 SMILES 身份入口

- `build_smiles_identity.sh`：构建全局 SMILES 图包、自同构包、Find_1 validation 标签旁路和单一 SMILES 身份清单。
- `train_smiles_identity.sh`：读取完整 YAML，在 Find_1 validation 上从头训练并以 W&B online 记录同划分监测曲线。

两个文件头分别给出正式提交命令。哈希、短批门禁、`strace` 和 release/launch 取证只进入临时运维脚本与执行记录，不写入正式脚本或 YAML。

## 固定批次训练入口

[批次训练](批次训练/README.md) 保存第 1–5 批共 21 个零参数入口。每个脚本固定对应一份 YAML，并在头部给出 `submit_task.sh` 提交命令；脚本按当前代码和依赖解释历史配置，不承诺复现最初实验环境。

## 最优变体从头训练入口

[最优变体推荐](最优变体推荐/README.md) 保存综合全部 validation 实验后推荐的三个当前入口。每个脚本固定对应一份正式 YAML，并从零开始训练，不包含 Job、release、launch、锁或一次性取证逻辑。
