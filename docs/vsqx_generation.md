# 自动 VSQX 生成

此入口生成 VSQ4 工程，不依赖原唱真值、人工 VSQX 模板或任何新增听审。
输出的 MIDI、起点和终点来自音符模型；文件编码检查与识别准确率分别记录。

## 共享接口

桌面使用版本化 stdio JSON 协议，CLI 直接调用相同 LocalEngine。无需 HTTP 服务。

| 动作 | 输入 | 行为 |
| --- | --- | --- |
| recognize / recognize-whole | WAV、真实音素模型、设备、范围 | 音素识别，产出 local_task_result_v1 |
| build_score，backend=game | result_path、可选 snapshot、tempo | GAME 音符推理，生成 score.json 与 transcription.vsqx |
| build_score，backend=game，notes_path | 同一音频的音素结果、已有音符 JSON | 零模型生成 VSQX |
| export_vsqx，无 template_path | 已有 score / notes，output_path | 零模型导出 |
| export_vsqx，有 template_path | 旧结果、旧模板 | 兼容原有模板模式，保留模板音高及时间 |
| note_runtime | 可选 snapshot | 检查配置，不加载模型 |

build_score 的旧内部默认仍为 pyin，以保持历史调用兼容；
CLI score 和新桌面明确选择 game。音素默认模型、检查点和历史注册表不变。

## 本地运行配置

根目录 local-notes.json 是机器配置，不纳入 Git。示例：

~~~json
{
  "schema_version": "local_note_runtime_v1",
  "snapshot": "models/game",
  "python": ".venv/Scripts/python.exe"
}
~~~

目录是已经核验的 GAME v1.0.3 medium ONNX 快照，包含 snapshot-lock.json、
config.json、encoder.onnx、segmenter.onnx、bd2dur.onnx、estimator.onnx。
原快照安装工具为 tools/prepare_game_snapshot.py。不会自动下载、换模型或改用 pYIN。

模型目录选择顺序：显式 snapshot、ANY2XSAMPA_GAME_SNAPSHOT、local-notes.json、
models/game。ANY2XSAMPA_NOTES 可以指定另一个配置文件。
相对路径按配置文件所在目录解析。当前 GAME 仅支持显式 CPU；不是 CUDA 失败后的回退。

## 音符与时间

- GAME 的 note_transcription_v1 保留原始 region_index、MIDI、onset、offset 和 duration。
- 已保存的真实预测必须通过原始张量哈希及解码重放；整首预测必须通过所有窗口重放。
  外部手工/测试 JSON 必须明确 mode=fixture，不可冒充真实模型输出。
- 整首窗口最长 20 秒，重叠 2 秒，窗口中点分配时间所有权；重复音与休止保留。
  只连接双方都跨过人工拼接点的同音音符，记录 source_regions。
- 音素仅用于识别后的发音赋值，不进入 GAME。音素和音符文件必须绑定同一源 WAV。
  混音与人声的不同文件不能只凭相似内容混用；不会猜测时间平移。
- 片段导出的时间保持相对所绑定的源 WAV；源 WAV 若本来就是裁剪文件，
  0 秒就是这个裁剪文件的开头。不会自动套用整首歌的未知偏移。
- BPM 为编码网格，480 PPQ；分别舍入 onset 和 offset，再由两者差求 duration。
  无节拍量化、音高移调、同音全局合并或短音阈值删除。
- 不能表示的音符、重叠或非有限数值会报错，不悄悄修成另一个乐谱。

## 音素与发音缺口

IPA 仍是核心表示。Miku V4C 是一个导出目标，并不覆盖所有 IPA。
不支持或缺失的发音不会让音符消失：该 note 使用“待定”及中性 a，
原 IPA、候选映射、失败原因及占位计数全部写进 score.json。
这不表示已识别出 a。无音素输入时所有 note 都会明确标记为此模式。

现在不要求人工确认才能导出。软件不宣称自动转写已经通过原唱联合准确率验收。

## 产物

一次新生成只写新目录：

- score.json：local_note_score_v1，可供程序使用和重导出。
- transcription.vsqx：真实 VSQ4 XML。
- transcription.vsqx.provenance.json：源分数哈希、音符数量、读回结果、舍入误差、
  发音占位数量和质量状态。
- note-prediction.json / windows/：新推理时保存音符、原始张量、输入绑定和逐窗调用记录。
- REQUEST.json / NOTE_PLAN.json / EXIT.json / DELIVERY.json：请求、上限、失败或完成状态。

已有 JSON 直接导出时，按输出文件旁边的 .score.json 和 .provenance.json 记录。
已存在的输出和来源记录均不覆盖。生成并不覆盖人工 VSQX 或原始音频。

## 验证范围

自动测试覆盖完整调用链（fixture worker）、独立 XML 读回、MIDI与时间、重复音、70 ms
短音、休止、半 tick、整首拼接、不同源哈希、张量错配、取消、失败不重试及拒绝参考解码条件。
另外用十份已有真实 mix/vocal GAME 预测和相应已有 phonetic posterior 做零模型集成。

这些检查验证序列化及执行流程。没有新增模型推理来验证新的整首路径，也没有运行
VOCALOID 的实际导入/试听，不能把它们描述成已完成。模型预测准确率不由这些测试推导。

## GitHub 发布

源码和 wheel 不携带模型权重、Python 环境、个人音频、人工工程、local-notes.json、
本机 local-models*.json 或 .state 研究缓存。空白 VSQ4 容器和 IPA 导出规则随包分发。
桌面开发机发布目录中的运行时配置指向本地安装，不能称为已验证的跨机器安装包。

模型权重仍应从各自的合法来源安装。当前改动不会自动创建远端仓库、上传模型或推送代码。
