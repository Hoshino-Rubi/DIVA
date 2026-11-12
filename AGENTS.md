# AGENTS.md

## Project Ground Rules

- The core intermediate layer of this project is phones / IPA / articulatory features, not recognized text.
- X-SAMPA is an export layer only. It must not be treated as the recognizer truth layer or the primary internal representation.
- Prefer small, verifiable iterations. Do not perform unrelated refactors while implementing a feature.
- After every code change, run at least one minimal verification command. After feature completion, run the relevant tests.
- Before spending training/experiment budget, review the executed code for data isolation, loss/initialization, sampling/RNG, update counts, evaluation gates, resource limits, and failure accounting. Save a hash-bound review and passing verification evidence before launch.
- A code change invalidates the corresponding launch review. Count failed launches and retries; never silently restart an entire completed/partially completed experiment group.
- Before adding a new external dependency, explain why it is needed and keep the integration behind a replaceable adapter.
- The recognizer layer must support both real model mode and fixture/mock fallback mode.

## Accuracy Work Red Lines / 准确度工作红线

- 用户要求持续推进时，一轮实验、测试、工具准备或报告完成，不是整个任务的停止条件。尚无可信净收益、且没有真正需要用户操作或新增授权的阻塞时，必须在现有授权和资源预算内继续下一项有证据支持的工作；不得把阶段完成当作目标完成。
- 不得把局部增加一个匹配、占位减少、两路一致、成功导出、测试通过或程序更稳定包装成准确率提升。必须同时报告固定分母、净 TP/FP/FN、逐歌退化、失败/缺失及适用范围；不完整或未达预设门的结果不能推广。
- 路线失败后要明确拒绝该版本，转向具体缺失信号或可检验的新机制；不得反复重述指标、编写交接或重复索要已有授权来代替实施。不得为满足收益要求而猜测、改口径、删失败、反复查看同一验证集调参或无依据穷举。
- 新数据处理或实验前冻结输入、开发/验证划分、参数、指标、资源和失败规则，并保存绑定最终代码及通过验证的哈希审阅。代码变化使对应旧审阅失效；保留全部尝试，不自动重跑已完成或部分完成组。
- 不要求用户听审、标注、填表、检查 GUI 或承担其他人工评测。确需人工窗口时，指出具体不可替代的操作、证据与授权缺口；同范围既有授权持续有效，不重复索权。
- 持续推进不等于无界消耗：遵守用户最新资源上限和禁止事项，主动切换已被证伪的路线，不偷偷扩大范围。预算确实耗尽或缺少不可替代信号时如实记录阻塞，不能伪称已修复。
- 本红线同时适用于工作区及实际发布源仓库；任何交接须保留尚未达成的准确度目标、负结果、禁止重跑的组、有效授权和下一项可执行工作。

## Benefit-first Workflow / 收益优先流程

- 禁止与当前改动、数据隔离、结果正确性或资源安全没有明确关系的哈希扫描、冒烟、重复全回归及防御性脚手架；不得用这些活动代替准确度工作。
- 哈希绑定用于证明实际运行的版本，不是层层审批。最终代码、配置、精确输入及定向通过测试只冻结一次；已固定且未变的模型/依赖复用已有身份，不反复扫描 Python、.venv、全部仓库或历史 run。只有受改动影响的绑定失效，不扩大成全环境重新审计。
- 每项验证必须能说明它防止的具体错误；优先运行覆盖本次机制的定向测试。没有相关变更不得反复跑已通过的集成/合成音/完整回归。
- 并行任务分离文件所有权；最终测试前停止源码编辑。禁止因格式、警告文案或报告修改让无关自然实验重新启动。
- 对照结果必须先全部冻结再读取参考；保留失败、固定分母、逐歌退化及既有资源上限。这些正确性约束不能以提速为由取消。

## Recognizer Rules

- `stub` mode is allowed only as a fallback or debugging path. It must not remain the default main path.
- `fixture` mode should accept external posterior / logits / phones JSON input and convert it into the stable `PhoneSegment` contract.
- `real` mode must be encapsulated in its own adapter. Do not scatter external command or model-specific logic across the pipeline.
- Pipeline and CLI changes must preserve the dataclass / JSON output contract unless an explicit migration is documented.

## Engineering Style

- Keep functions short and typed.
- Put dialect rules, phonesets, and mappings in config files instead of one large Python file.
- Prefer adapter interfaces over direct coupling to third-party tools.
- If a real recognizer cannot run in the current environment, keep the interface complete and fail clearly, while preserving fixture mode as the primary runnable path.
