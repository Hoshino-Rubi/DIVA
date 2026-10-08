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
