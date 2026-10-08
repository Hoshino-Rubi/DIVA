# Any2Xsampa Windows x64

1. 从 Release 下载 Any2Xsampa-Setup.exe 和三个 Any2Xsampa-Windows-x64-20261008.zip.001／.002／.003 分卷，放在同一个文件夹。双击安装器，选择可写目录；安装器会自动校验、拼接并解压。不要单独打开分卷或只复制桌面 EXE。
   若已获得完整 ZIP，也可直接解压整个目录。支持中文及空格路径。
2. 双击 Any2Xsampa.Desktop.exe。不需要另外安装 Python、.NET、CUDA Toolkit 或启动服务器。
3. 初次打开会自动下载缺失的 PhoneticXeus 和 GAME 固定权重，下载约 2.31 GiB。界面可取消、设置 HTTP(S) 代理、继续下载；连接失败不会自动重试。运行识别时禁止联网。
4. 下载完成后，选择真实模型 guarded-candidate-v2，打开 WAV，点击“开始识别并生成”。默认整首并自动生成 VSQX。
5. 当前音素模型需要 NVIDIA CUDA；GAME 音符识别在 CPU 四线程运行。仍需已安装且兼容 CUDA 12.6 的显卡驱动。Windows 11 22H2 或更新版本、x64。程序解压及模型/下载缓存建议预留 12 GiB 磁盘空间。

## 离线安装

从同一 Release 下载以下四个文件，放到同一个文件夹，文件名保持原样：

- PhoneticXeus-model.safetensors.001
- PhoneticXeus-model.safetensors.002
- PhoneticXeus-model.safetensors.003
- GAME-1.0.3-medium-onnx.zip

取消自动下载后，点击“导入离线模型文件夹”，选择以上文件所在目录。无需手动拼接或解压 GAME ZIP。也支持把原始 model.safetensors 放到该目录。
如已开始网络下载，已验证完成的文件会复用。未完成 .part 保留，点击继续下载可恢复。
程序目录内 models 保存模型；.state 保存缓存、设置、日志和结果。

## CLI（同一本地引擎）

在本目录打开终端：

```bat
any2xsampa.cmd models status
any2xsampa.cmd models install --proxy http://127.0.0.1:18080
any2xsampa.cmd models install --offline-dir "D:\模型"
any2xsampa.cmd models list
any2xsampa.cmd recognize "D:\歌曲.wav" --model guarded-candidate-v2 --whole-file
any2xsampa.cmd score "结果目录\result.json"
```

若服务器不支持已有断点，命令行可显式使用 models install --reset-downloads，重新下载未完成文件。已安装且通过哈希的模型不会重复下载。

## 质量与许可

这是可运行的研究预览版；VSQX 已实现音高、起点、时值写入及自动读回验证，但尚无正式逐音符联合准确率的验收结论。
应用不伪造真值，不要求额外人工审核才能导出。
第三方许可见 THIRD_PARTY_NOTICES.md；GAME 权重是 CC BY-NC-SA 4.0，不能将其整体宣称为 MIT 或用于不符合许可的商业用途。
