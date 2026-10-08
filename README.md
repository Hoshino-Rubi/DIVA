# Any2Xsampa

本地音频 → IPA / phones → 音符 → VSQX / X-SAMPA。桌面程序和 CLI 共用
本地子进程引擎，正常使用不需要启动 HTTP 或 gRPC 服务。

VSQX 生成已接通 **GAME 音高、起点、终点 + 音素识别结果**。可以处理整首音频，
也可以复用已经保存的预测，生成过程不要求听审、标注或填写确认表。

## 使用桌面程序

配置本地音素模型运行包及 GAME 模型目录后：

1. 打开 WAV，选择音素模型和“整首”或“当前片段”。
2. 保持“识别后自动生成 VSQX”勾选，点击“开始识别并生成”。
3. 完成后点击“导出 VSQX（音高/时值保留）”。

“音符与 VSQX”区域支持单独生成，也支持“使用已保存 GAME 音符”，无需重复推理。
音素推理使用所选设备；当前 GAME 适配器明确运行在 CPU 四线程。
取消只终止本应用拥有的任务；失败不会自动重试或改用假结果。

## CLI：已有结果生成 VSQX

在已安装的 Python 环境中运行（建议 Python 3.12 或更新版本）：

~~~text
python -m any2xsampa.cli export-vsqx prediction.json --phones result.json --output song.vsqx
~~~

- prediction.json：已保存的 GAME note_transcription_v1 音符。
- result.json：同一音频、同一时间坐标上的 local_task_result_v1 音素。
- 两份证据的源音频哈希必须一致；不会猜偏移或升降八度。
- 本命令不运行模型。已有 score.json 也可直接导出：

~~~text
python -m any2xsampa.cli export-vsqx score.json --output song.vsqx
~~~

只提供音符 JSON 时也能生成 VSQX；没有音素的 note 会明确标为“待定”，以中性
a 发音占位。它保留音高与时间，不代表已还原歌词或发音。

## CLI：从音频生成

先配置真实音素模型运行包；recognize 不会自动改用 fixture。
使用 ANY2XSAMPA_MODELS 指向与当前源码匹配的注册表；本机桌面发布目录里的
any2xsampa.cmd 已配置对应版本。旧冻结注册表不会被覆盖。

~~~text
python -m any2xsampa.cli models list
python -m any2xsampa.cli recognize input.wav --model MODEL_ID --device cuda --whole-file --output-dir runs/phones
python -m any2xsampa.cli score runs/phones/result.json --backend game --snapshot models/game --output-dir runs/score
~~~

结果包括 runs/score/transcription.vsqx、score.json、来源说明，以及可重放的
GAME 原始张量和分窗记录。更改 BPM 只改变秒数到 tick 的编码网格，不会改变原始
秒数或移调。旧 pYIN 路径保留为显式的 --backend pyin。

已有 GAME 结果可跳过音符推理：

~~~text
python -m any2xsampa.cli score result.json --notes prediction.json --output-dir runs/score
~~~

GAME 使用已有的固定本地快照，不会自动下载。将
[示例配置](examples/local-notes.example.json) 复制为根目录 local-notes.json，
填写现有运行时和快照目录；也可使用 --snapshot 或 ANY2XSAMPA_GAME_SNAPSHOT。
应用源码和模型权重分别管理。仓库不包含 Python 环境、模型、个人音频或研究缓存。
模型和 Python 库需要在运行前安装；本次功能没有新增依赖。
GAME 运行时使用 numpy、soundfile、librosa 和 onnxruntime；
保存的 PhoneticXeus posterior 重放还使用 safetensors。

## 输出保证与边界

- 生成真正的 VSQ4 XML，保留整数 MIDI 音高、起点、终点、休止和重复音。
- 不把 CTC 发射时间当作 note 时值，也不使用歌词、VSQX 或人工答案解码 GAME。
- 每次写出后自动读回，核对 note 数量、音高和时间；120 BPM / 480 PPQ 下，
  起点或终点的 tick 舍入不超过约 0.521 ms。这只是文件编码误差。
- 整首使用最长 20 秒的窗口和上下文重叠。仅在两个窗口都表明同一音高的音符
  跨越拼接点时连接该音符；真实的同音重复边界保留，拼接变更写入来源记录。
- 缺失或不能用于 Miku V4C 的音素会保留原 IPA 并标记占位，不删除音符、伪造歌词。
- **文件正确写入不等于模型识别准确。** 目前没有可支持原唱音高及起止点联合
  准确率声明的独立真值。输出是自动转写，识别错误仍可能存在。
- 不覆盖原始 WAV、人工工程或已存在的输出文件。无需人工确认才能生成或导出。

完整接口及验证说明见 [VSQX 使用说明](docs/vsqx_generation.md)。
旧的参考工程/强制对齐研究流程保留在 [历史工作流](docs/legacy_product_workflows.md)，
它们不是当前自动 VSQX 生成的前置步骤。

## 开发验证

~~~text
python -m pytest tests/test_game_vsqx.py tests/test_local_score.py tests/test_game_notes.py tests/test_local_engine.py tests/test_local_whole_audio.py tests/test_local_desktop_layout.py -q -p no:cacheprovider
dotnet build apps/Any2Xsampa.Desktop/Any2Xsampa.Desktop.csproj --no-restore -c Debug -p:Platform=x64 -p:WindowsPackageType=None
~~~

模块入口和安装后的 any2xsampa 命令等效。VSQX 空白容器与配置文件随 Python 包分发，
不依赖本人的历史工程目录。PhoneSegment、posterior 与旧模板导出契约保持兼容。
