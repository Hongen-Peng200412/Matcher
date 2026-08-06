# Matcher 与服务器交互

本目录只把当前本机仓库同步到服务器的 `/home/penghongen/My_Project/Matcher`。它不会覆盖或混入 `/home/penghongen/My_Project/AdaLigand`、Pocket_Plus 或其他现有项目。

- `run_sync.bat` 与 `sync_code.ps1` 执行保留远端既有文件的安全同步；AI 可以在核对目标路径后使用。
- `run_syncWithClean.bat` 与 `sync_codeWithClean.ps1` 会先删除整个远端 Matcher 目录，只允许人类手动运行。
- `other/Invoke-PasswordSsh.ps1` 是共享安全连接入口；主机密钥必须已经写入本机 `known_hosts`，脚本不会自动接受变化后的密钥。

正式任务由 `训练与运行/submit_task.sh` 提交，控制证据位于 `/home/penghongen/Feedback/Matcher`，训练和推理结果位于 `/storage/penghongen/Matcher/Results`。输入的 A–G 与 Stage1 产物仍从 AdaLigand 的正式存储目录读取。
