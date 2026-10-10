# Any2Xsampa pre-1.0.1a

[English](https://github.com/Hoshino-Rubi/any2xsampa/blob/pre-1.0.1a/docs/releases/pre-1.0.1a.md)

Windows x64 研究预览的可靠性更新：

- 新增本地诊断 ZIP，Python 启动失败时也能导出错误码、版本、模型预期哈希/大小及资源数值。
- 不包含音频、预测、原始日志、用户名、完整路径或代理凭据，不会自动上传。
- 修复常驻 worker 日志归属，每个任务关联明确的进程会话。
- 新增 Windows 产品回归、新目录回放与 VSQX 读回，以及带 SHA 绑定的构建输入。
- 公开校验清单只列实际附件。

下载 **Any2Xsampa-Setup.exe** 和三个 **Any2Xsampa-Windows-x64-pre-1.0.1a.zip.001／.002／.003**，放在同一目录后运行安装器。请选择新的可写安装目录。Python／.NET 随包提供，模型与 pre-1.0.0 相同；可复用旧版四个离线模型附件。

[完整中文说明](https://github.com/Hoshino-Rubi/any2xsampa/blob/pre-1.0.1a/README_CN.md)

仍为未签名研究预览。本轮没有新增训练，不声称准确率提高。GAME 权重保留 CC BY-NC-SA 4.0 限制。正式音高/起止点联合准确率及另一台电脑的 GUI 启动尚未验收。
