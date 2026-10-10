# Any2Xsampa

[English](README.md)

本地音频 → IPA / phones → 音符 → VSQX / X-SAMPA。桌面程序与 CLI 共用本地子进程引擎，正常使用无需启动 HTTP 或 gRPC 服务。

## Windows 研究预览版

从同一个 GitHub Release 下载以下四个文件，保留名称并放在同一个文件夹：

- `Any2Xsampa-Setup.exe`
- `Any2Xsampa-Windows-x64-pre-1.0.1a.zip.001`
- `Any2Xsampa-Windows-x64-pre-1.0.1a.zip.002`
- `Any2Xsampa-Windows-x64-pre-1.0.1a.zip.003`

双击安装器，选择可写目录。安装器会自动校验并解压；不要直接运行分卷或只复制桌面 EXE。无需另装 Python、.NET、CUDA Toolkit，也不用启动服务器。

系统要求：Windows 11 22H2 或更新版本、x64；音素识别需要 NVIDIA GPU 和兼容 CUDA 12.6 的驱动，GAME 音符识别使用 CPU 四线程。建议预留至少 12 GiB。

安装器尚未签名。已完成本机分卷安装、运行时搬迁和保存结果导出验证，尚未在另一台机器上做 GUI 启动验证。

### 自动下载模型

首次启动从固定上游 HTTPS 地址下载 PhoneticXeus 和 GAME 权重，总计约 2.31 GiB。安装前核验大小和 SHA256，损坏文件不会伪装成可用模型。

支持关闭启动时下载、HTTP(S) 代理、取消、断点续传与离线导入。连接失败不自动反复重试；未完成下载保留，可显式继续。识别阶段不联网。

不能联网时另下载三个 `PhoneticXeus-model.safetensors.00x` 分卷和 `GAME-1.0.3-medium-onnx.zip`，放到一个文件夹，在应用中选择“导入离线模型文件夹”。无需手工拼接或解压。

## 桌面操作

1. 完成模型安装，打开 WAV，明确选择 `guarded-candidate-v2`。
2. 选择整首或当前片段，保留“识别后自动生成 VSQX”。
3. 点击“开始识别并生成”，完成后导出 VSQX。

“音符与 VSQX”区域可复用已保存的 GAME 音符，不重复推理。取消仅影响本应用拥有的任务。真实模型不可用会明确失败，不会悄悄改用 fixture。

## CLI

在程序安装目录打开终端：

~~~bat
any2xsampa.cmd models status
any2xsampa.cmd models install --proxy http://127.0.0.1:18080
any2xsampa.cmd models install --offline-dir "D:\模型"
any2xsampa.cmd models list
any2xsampa.cmd recognize input.wav --model guarded-candidate-v2 --device cuda --whole-file --output-dir runs/phones
any2xsampa.cmd score runs/phones/result.json --backend game --output-dir runs/score
~~~

代理需替换为实际地址。源码使用等效的 `python -m any2xsampa.cli`，但不含运行环境或权重，需自行配置。只有存在已核验下载清单的版本才启用自动下载。

### 已保存结果零推理导出

~~~text
python -m any2xsampa.cli export-vsqx prediction.json --phones result.json --output song.vsqx
python -m any2xsampa.cli export-vsqx score.json --output song.vsqx
python -m any2xsampa.cli score result.json --notes prediction.json --output-dir runs/score
~~~

音符、音素证据必须来自同一音频哈希和原唱时间坐标，不猜偏移或升降八度。只提供音符也可导出；缺少可用音素的 note 保留音高与时间，以明确占位和中性 `a` 发音表示，不声称还原歌词。

## 输出保证与边界

- 生成真正的 VSQ4 XML，保留 MIDI 音高、起点、终点、休止和重复音。
- 写出后自动读回核对。120 BPM / 480 PPQ 下端点舍入至多约 0.521 ms，这是编码误差，不是识别准确率。
- 整首使用不超过 20 秒的重叠窗口，仅在两侧证据支持同音高持续音跨缝时拼接，真实同音重复保留。
- 不把 CTC 发射时间当作 note 时值，不用歌词、VSQX 或人工参考作为 GAME 解码条件。
- 未映射音素保留原 IPA 和占位状态，不删除 note 掩盖缺口。
- 不覆盖原始音频、人工工程或已有输出；生成、导出不要求增加人工审核。
- **文件正确写入不等于模型识别准确。** 缺少独立原唱逐音符真值，尚不能声明正式音高/onset/offset 联合准确率。研究预览仍可能含识别错误。

详见 [VSQX 使用说明](docs/vsqx_generation.md)、[分发说明](docs/distribution/README-使用说明.md)。[历史工作流](docs/legacy_product_workflows.md) 只保留研究用途，不是自动生成前置步骤。

## 开发与许可

测试和安装器构建入口见 [英文 README](README.md#development-verification)。`PhoneSegment`、posterior JSON 与显式 fixture/replay 契约保持兼容。X-SAMPA 只是导出层，不是内部识别真值。

PhoneticXeus 为 Apache-2.0；GAME 权重为 **CC BY-NC-SA 4.0**，不是源码 MIT，覆盖材料受非商业与相同方式共享限制。见 [第三方声明](docs/distribution/THIRD_PARTY_NOTICES.md)。应用自己的源码许可证尚未由仓库所有者选择。

## pre-1.0.1a 可靠性更新

“模型环境与已有结果回放”内新增“导出诊断包”，Python 启动失败或模型未就绪时也能使用。仅包含版本、错误码、模型预期哈希/大小、资源数值及任务耗时；不包含音频、预测、原始日志、用户名、完整路径或代理凭据，不会自动上传。大小相符不等于哈希验证通过。

常驻模型进程使用独立会话日志，每个任务保存 worker-session.json 说明所属会话。请安装到新文件夹，旧模型与结果不被覆盖；可复用 pre-1.0.0 离线模型附件。

Windows 回归入口为 python tools/check_windows_release.py --output <新目录>。它使用零模型测试；完整历史研究测试另外需要未公开的本地资料。构建与签名顺序见 [构建说明](docs/release_build.md)。本次更新不声称识别准确率提高。
