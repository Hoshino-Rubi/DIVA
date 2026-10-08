# PhoneticXeus Isolated Runtime

This optional experiment is a replaceable acoustic-phone adapter, not a new
default recognizer. It must never import model code into the main application
process or modify its existing Python/Torch installation.

## Direct Dependencies

- Python 3.12: use the existing interpreter to create a run-local environment.
- Torch 2.8.0+cu126 and torchaudio 2.8.0+cu126: a matched published pair. The
  vendored model imports torchaudio. Its forced-alignment helper is not used.
- Transformers 4.57.6 / huggingface-hub 0.36.2: local model construction only.
- safetensors 0.8.0: verified non-pickle weights and derived tensor artifacts.
- typeguard 4.4.4: decorators imported by the vendored encoder/CTC modules.
- NumPy 2.2.6, SciPy 1.15.3, soundfile 0.13.1: bounded waveform reads and
  polyphase resampling, with no ffmpeg or new separation model.
- PyYAML 6.0.3 and packaging 25.0: pinned model configuration dependencies.

The direct non-Torch requirements are in
tools/requirements/phoneticxeus-py312.txt. Installation must capture pip reports
(including distribution hashes), freeze all transitive versions, and run pip
check. Official matched-wheel reference:
https://pytorch.org/get-started/previous-versions/

All downloads must use http://127.0.0.1:10808; clear NO_PROXY and do not retry
directly. Acquire only changelinglab/PhoneticXeus revision
8d83dee94817a07dc150f87d08f7e0ee01bdb66d. Verify the pinned source tree and the
single model.safetensors file before loading anything.

Runtime must be offline, batch-one for feature extraction, four CPU threads,
explicit CUDA, and bounded to 7 GiB process-reserved / 14 GiB total GPU memory.
Missing weights, architecture mismatches, CUDA failures, and invalid output
must fail clearly. No random initialized replacement, silent CPU fallback, or
synthetic result is permitted.

Launch the worker with -I -B -X utf8. Isolated Python ignores PYTHONUTF8 from the
environment; the upstream builder opens its IPA JSON without an encoding
argument, so a cp950 Windows default fails. Explicit UTF-8 preserves upstream
source bytes and avoids rewriting the pinned model.

## Optional read-locked audio cache

`tools/run_phoneticxeus_worker.py --audio-cache` enables a Windows-only 8 MiB
LRU of mono/resampled audio and up to eight source handles. It is not the
default and never caches neural features or logits. Sources are SHA-256 checked
while read-sharing handles exclude writes/deletion; their identity is checked
on every access, and handles are released after the worker batch.

The matched original-window validation reproduced all 18 non-test baseline
feature/logit artifacts byte-for-byte, and the fixed 8s/20s three-cold/five-warm
benchmark reproduced all 16 corresponding artifacts and real/fixture contracts.
Observed warm median time fell 16.9%; cold startup fell only 0.5%, and GPU reserved
was unchanged. These are scoped measurements, not a general hardware guarantee.
See the quality-first run's performance and cache-validation reports.
