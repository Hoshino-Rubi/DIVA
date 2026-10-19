# DIVA · Windows x64 使用说明

DIVA（Dialect Intonation & Vocal Automation）原名 Any2Xsampa。本说明对应 DIVA v1.0.2，仓库为 https://github.com/Hoshino-Rubi/DIVA。历史版本与文件名保持不变，不要重命名旧分卷、模型或数据目录。内部程序仍名为 Any2Xsampa.Desktop.exe，CLI 仍为 any2xsampa.cmd。

下载 DIVA-Setup-v1.0.2.exe，运行后获取固定版本分卷、校验 SHA256 后安装。离线时，把三个 DIVA-Windows-x64-v1.0.2.zip.001 / .002 / .003 放在安装器旁边。公共自动安装需要已发布的公开 Release。

选择新的可写目录。可选择旧版安装目录，只复制哈希验证通过的模型，旧版和结果不修改。双击 Any2Xsampa.Desktop.exe，无需另装 Python、.NET、CUDA Toolkit 或服务器。

首次启动下载 PhoneticXeus 与 GAME 固定权重，约 2.31 GiB；支持取消、显式续传、代理和离线导入。推理阶段断网。打开 WAV，选择 guarded-candidate-v2，选择整首或片段，点击开始识别并生成，再导出 VSQX。

Windows 11 22H2+ x64；音素识别需要 NVIDIA 驱动兼容 CUDA 12.6，GAME 使用 CPU 四线程。建议初次安装和模型缓存预留 16 GiB。

## 模型、升级和卸载

离线模型使用三个 PhoneticXeus-model.safetensors.00x 与 GAME-1.0.3-medium-onnx.zip，放到同一文件夹后在应用导入。可复用预览版相同哈希的附件，无需手工拼接。

应用 models 保留该版本的模型文件。设置、日志和结果保存于 %LOCALAPPDATA%/Any2Xsampa/state，已验证模型下载缓存位于 %LOCALAPPDATA%/Any2Xsampa/cache/model-downloads。ANY2XSAMPA_DATA_HOME 可指定绝对数据目录；旧预览结果保持原位，可直接回放。

安装器卸载只删除收据中未修改的程序文件，保留模型、共享数据和其他文件。卸载前关闭应用；版本安装失败时旧版不受影响。程序更新仅在点击后检查并打开官方页面，不静默替换。

## CLI

any2xsampa.cmd models status

any2xsampa.cmd models install --proxy http://127.0.0.1:18080

any2xsampa.cmd models install --offline-dir <模型目录>

any2xsampa.cmd recognize <歌曲.wav> --model guarded-candidate-v2 --whole-file

any2xsampa.cmd score <结果目录/result.json> --backend game

## 质量与许可

程序生成 VSQ4 XML 并读回核验音高、起点、时值和数量；这不证明模型识别全部正确。没有独立原唱真值时不宣称正式联合准确率，也不要求用户额外评测才能导出。

GAME 权重使用 CC BY-NC-SA 4.0；PhoneticXeus 使用 Apache-2.0，详见 THIRD_PARTY_NOTICES.md。当前候选未签名。
