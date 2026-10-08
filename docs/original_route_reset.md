# Original Route Reset

Audit snapshot: 2026-08-19. This document records the recovery route for the
core audio-to-phones system. It is a route and evidence ledger, not a product
acceptance record. Downstream Miku, MFA, Qwen, pitch, lyric, forced-alignment,
and VSQX artifacts must not be promoted to recognizer truth by passing their
own validators.

## Non-negotiable boundary

The internal truth layer is time-bounded phone evidence: phones, IPA,
articulatory features, candidates, confidence/posterior or logit traces, and
finite time ranges. X-SAMPA and engine phonemes are derived export formats.
Recognized text, lyrics, Jyutping, score notes, pitch, MFA boundaries, and
human acceptance can constrain, evaluate, or export phone evidence only after
the producer/recognizer has emitted it. They must never be used to conceal a
real-recognizer failure or to feed acoustic truth backwards into the route.

## Original technical route

```text
user/external audio (read-only)
  -> external producer or local model runtime
  -> versioned posterior/logit JSON
  -> replaceable recognizer adapter
  -> PhoneSegment / phones / IPA / articulatory evidence
  -> aligner v2 (timing and ownership metadata; no phone rewrite)
  -> X-SAMPA or engine exporter
  -> optional downstream product/evaluation artifacts
```

### Stage responsibilities

| Stage | Contract and evidence | Boundary rule |
| --- | --- | --- |
| Producer | Emits `external_posterior_json_v1` (or a documented replay schema) with candidate phones, scores/traces, finite windows, backend identity, and runtime provenance. | It may emit model evidence, never text as the truth representation. |
| Recognizer adapter | Validates fixture, external JSON, command output, or a real backend and constructs the stable `PhoneSegment` dataclass. | Real failures are explicit. Stub output is only an explicit debug fallback. |
| Phone layer | Carries `TimeRange`, `PhoneCandidate`, IPA/feature evidence, and metadata through the pipeline. | This is the only recognizer truth layer. |
| Aligner v2 | Adds monotonic timing, note ownership, and alignment diagnostics around existing segments. | It must not replace candidate phones with lyric or score symbols. |
| Exporter | Derives X-SAMPA or engine aliases from normalized IPA and the configured inventory. | Export strings are terminal output, never recognizer input. |

The current stage implementations are in `src/any2xsampa/core/models.py`,
`src/any2xsampa/adapters/recognizer*.py`,
`src/any2xsampa/pipeline/runner.py`,
`src/any2xsampa/adapters/aligner.py`, and
`src/any2xsampa/exporters/xsampa.py`.

## Recovery provenance boundary

The producer and runtime helpers in this tree are **contract-compatible
reconstructions** of the original route. They preserve the typed JSON,
diagnostics, adapter, and command-line contracts needed by the current
pipeline; they are not byte-level recovery of the original implementation.
The recovery evidence manifest marks the historical producer/runtime helper
objects (including the former recognizer helper and runtime probe) as
`unrecoverable_object`. Their missing bytes and exact implementation lineage
must not be inferred from the current behavior or from passing tests.

When a projection payload contains a `lineage_kind` field (for example
`serialized_runtime_lineage_kind` or
`serialized_mapped_frame_lineage_kind`), it is an evidence classification of
the available runtime or payload provenance. Values such as
`repo_local_current_runtime_lineage` and
`repo_external_sibling_runtime_lineage` describe the current source/runtime
relationship; values containing `historical_exact_*_unrecoverable` explicitly
record that exact historical evidence is unavailable. None of these values
establish byte identity with the lost helper.

## Evidence grades

Grades are property-specific. A higher grade for runtime availability does not
prove language/domain quality, phone accuracy, licensing, or human approval.

| Grade | Meaning | What it does not prove |
| --- | --- | --- |
| **G0 - rule** | `AGENTS.md` and `REDLINES.md` define the representation and failure rules. | That any implementation or model currently satisfies them. |
| **G1 - static contract** | Source/config inspection establishes a typed boundary, schema, or hash-checking code path. | That the path executes or that its acoustic output is correct. |
| **G2 - fixture replay** | A fixture or external JSON artifact round-trips into `PhoneSegment` and the pipeline/export contract. | Real model inference, Cantonese-singing suitability, or human-reviewed boundaries. |
| **G3 - real runtime** | A real adapter creates a model/runtime session, verifies resources, and records active providers and traces. | Cantonese-singing training evidence, acceptable phone accuracy, or product acceptance. |
| **G4 - producer lineage** | A producer-generated JSON artifact, immutable runtime/bundle metadata, and an independent replay test bind the output to its producer and input. | Human approval or a qualified forced-phone model. |
| **G5 - human reference triplet** | The user supplies or identifies the original-song WAV, a human-authored VSQX for the same song/section, and the WAV exported from that exact VSQX; pairing identity and hashes are recorded. | That the recognizer generalizes, or that its audio-only output is correct before evaluation. |

Do not label an artifact with a grade that it has not earned. In particular,
an X-SAMPA string, a `trusted` field, strict-product parser pass, model
confidence, structural validation, or downstream acceptance record is not G3,
G4, or G5 recognizer evidence by itself.

## Recovery matrix (audit snapshot)

| Surface | Evidence and current status |
| --- | --- |
| Stable phone contract | **Recovered (G1/G2).** `PhoneSegment`, `PhoneCandidate`, and `TimeRange` are typed and serialize through `PhoneSegment.to_dict()` (`src/any2xsampa/core/models.py:232-257`); `PipelineResult` serializes the trace and segments (`src/any2xsampa/core/models.py:270-289`). `tests/test_recognizer.py` and `tests/test_pipeline.py` exercise fixture-to-phone and pipeline JSON/X-SAMPA output. |
| Fixture and external posterior replay | **Recovered (G2).** Fixture shorthand and posterior/logit trace fields are normalized before segment construction (`src/any2xsampa/adapters/recognizer.py:299-340`), then fixture and external JSON readers build `PhoneSegment` records (`src/any2xsampa/adapters/recognizer.py:382-452`). Fixture mode is the default pipeline configuration (`src/any2xsampa/pipeline/runner.py:30-40`); it does not require a model download. |
| Explicit recognizer dispatch and failure behavior | **Recovered (G1/G2).** Dispatch keeps fixture and real modes explicit and rejects production `stub` (`src/any2xsampa/adapters/recognizer_dispatch.py:42-81`). Synthetic stub output is reachable only through the explicit debug fallback flag (`src/any2xsampa/adapters/recognizer.py:532-606`). The isolated command/backend adapters remain behind `recognizer_real.py` (`src/any2xsampa/adapters/recognizer_real.py:687-749`). |
| Local ONNX/eSpeak adapter | **Recovered as runtime plumbing (G3), not domain qualification.** The adapter resolves the local model/vocabulary, checks manifest hashes, and loads the model boundary (`src/any2xsampa/adapters/onnx_espeak.py:32-40,63-150,153-220`). The bundled model is a general eSpeak/Wav2Vec2 CTC resource; its presence does not establish Cantonese-singing training. |
| Runtime/bundle diagnostics | **Contract-compatible reconstructed runtime plumbing (G3); external bundle readiness remains conditional.** The historical runtime helper bytes are marked `unrecoverable_object`; current diagnostics report provider/dependency state without claiming byte identity. Diagnostics report provider/dependency state without running inference (`src/any2xsampa/adapters/onnx_espeak_runtime.py:3-7`), expose an archivable schema (`src/any2xsampa/adapters/onnx_espeak_runtime.py:426-435`), and persist bundle manifest/state through `tools/onnx_espeak_bundle_packager.py`. The latest direct probe reported `runtime_status=in_process_ready`, `ready=True`, `backend_kind=onnx-espeak`; that establishes local in-process readiness only. A prepared external bundle is classified as `bundle_missing_runtime` when only its runtime Python is absent, or `bundle_incomplete` when other required components are missing (`src/any2xsampa/adapters/onnx_espeak_runtime.py:610-679,682-685`). The route-surface test checks manifest/state schema, not worker deployment (`tests/test_original_route_surfaces.py:127-181`). |
| Posterior JSON producer | **Contract-compatible reconstructed producer lineage plumbing (G4), not byte-level recovery or model/domain qualification.** The historical producer/runtime helper objects are recorded as `unrecoverable_object`; the current import surface is a compatibility-only re-export (`src/any2xsampa/adapters/producer_compat.py:3-29`). Producer ownership remains in `src/any2xsampa/adapters/posterior_json_producer.py:1-6,261-342`, while the CLI imports that boundary (`tools/posterior_json_producer.py:7-13,75-82`). Producer payloads carry backend identity, real/diagnostic flags, and model metadata; independent replay is covered by `tests/test_producer_lineage_contract.py:9-40` and `tests/test_original_route_surfaces.py:50-125`. The combined producer/runtime and surface gate returned `12 passed` (`test_producer_runtime_reset.py` alone: `9 passed`); diagnostic backends are explicitly marked approximation-only and non-real-model output (`src/any2xsampa/adapters/posterior_json_producer.py:76-86`). Passing this gate proves current contract compatibility only; it cannot restore the lost helper bytes. |
| Aligner and exporter | **Recovered (G1/G2).** `SegmentScorePhoneAligner` attaches score-span evidence without rewriting phones (`src/any2xsampa/adapters/aligner.py:1-5,17-59`); the pipeline runs normalizer, dialect, aligner, and X-SAMPA export in order (`src/any2xsampa/pipeline/runner.py:68-105`). X-SAMPA is filled from normalized IPA (`src/any2xsampa/exporters/xsampa.py:10-19`), proving stage wiring rather than acoustic quality. |
| Cantonese-singing CTC adapter | **Contract present; product model not established (G1).** `recognizer_cantonese_singing_ctc.py` validates an explicit bundle/checkpoint, vocabulary, hashes, and runtime options. No accepted Cantonese-singing checkpoint/training/license record is part of the current model manifest, so the adapter is not a qualified model route. |
| Human-reviewed phone evidence | **Not started (below G5).** Machine MFA/CTC/Qwen output and mechanically rebalanced drafts remain review suggestions. No generated artifact can claim human approval. |
| Miku/pitch/lyrics/forced-alignment/VSQX work | **Downstream only.** Keep these experiments and strict acceptance validators available for later product work, but they neither repair the producer route nor redefine recognizer truth. See `docs/forced_phone_aligner_evidence_audit.md` and `REDLINES.md`. |

The model bytes and revisions currently recorded for the ONNX/eSpeak plumbing
are in `models/MANIFEST.json`. They establish reproducibility of those bytes,
not a Cantonese-singing model claim.

## Verification commands

Run from the repository root. Generated logs, profiles, and JSON belong under
`.state/`; source audio and external delivery files remain read-only.

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider `
  --basetemp .state\temp\route-docs `
  tests\test_recognizer.py tests\test_pipeline.py `
  tests\test_onnx_espeak.py tests\test_onnx_cuda_resources.py `
  tests\test_recognizer_real_command_security.py
```

At this audit snapshot the focused contract/runtime command returned
`29 passed` (using the repository-local `.state` basetemp). This is G2/G3
plumbing evidence only; it does not validate producer lineage or phone quality.

The producer gate below records the current green lineage plumbing required for
G4. It remains separate from model/domain qualification and human approval:

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider `
  --basetemp .state\temp\route-docs-producer `
  tests\test_producer_runtime_reset.py tests\test_original_route_surfaces.py
.\.venv\Scripts\python.exe -B tools\posterior_json_producer.py --diagnose-runtime
```

The current producer/runtime and route-surface run returned `12 passed`,
including `9 passed` in `tests/test_producer_runtime_reset.py`; the focused
contract/runtime command above returned `29 passed`, and the full suite returned
`474 passed, 3 skipped`. The diagnostics command emitted an archivable snapshot
with `schema_version=onnx_espeak_runtime_diagnostics_v1`,
`runtime_status=in_process_ready`, `ready=True`, and
`backend_kind=onnx-espeak`; its external bundle status remains
`bundle_incomplete` until the missing worker/runtime components are supplied.
The checked diagnostic and pipeline artifacts are
`.state/runs/original-route-reset-v1/producer-diagnostic.json` and
`.state/runs/original-route-reset-v1/pipeline.json`; they contain finite
segments, candidate symbols, backend identity, and a replay through
`PhoneSegment`/aligner v2/X-SAMPA. A direct `--backend auto` smoke on
`samples/demo.wav` emitted `external_posterior_json_v1` with
`backend_kind=onnx-espeak`, `real_backend=true`, `real_model_output=true`, 34
segments, and model/vocabulary hash plus provider metadata
(`.state/route-docs-auto.posterior.json`). These are runtime/lineage checks;
they do not establish Cantonese-singing quality, licensing, or human approval.

## First genuine human window

The reset remains autonomous through source recovery, adapter implementation,
fixture replay, producer/runtime diagnostics, and focused tests. The first
genuine human window opens when the next dataset/evaluation unit requires the
user to provide or identify one complete reference triplet:

1. the original-song WAV, which is the target input for the real acoustic
   recognizer;
2. the user's human-authored VSQX for the same song or section, which carries
   the authoritative human note/phone annotation; and
3. the WAV exported or rendered from that exact VSQX, which supplies the
   paired acoustic realization of the human annotation.

The project records all three paths, revisions, pairing identity, and SHA-256
values before use. The VSQX and exported WAV are supervised reference or
calibration evidence. The first baseline on the original-song WAV remains
audio-only: it receives no lyrics, VSQX phones, note context, or rendered-WAV
phones during acoustic decoding. Only after recognition may the reference pair
be used for grouped training, timing alignment, comparison, evaluation, or
downstream export.

The current workspace does not contain a verified complete triplet. It has
three VSQX templates and an original `從前.wav`, but no exported WAV proven to
come from any of those exact VSQX files; the old `Desktop/MIDI` and
`Queens Rd E` directories are absent. Therefore the first human window is to
provide the three paths (or restore the missing directories) and identify the
chosen revision. It is not a phone-by-phone review task.

The following are **not** human windows: a passing pytest run, CLI help,
ONNX Runtime provider output, a generated `.state` artifact, the old `8766`
phone reviewer, strict-product parser success, a `trusted` manifest, model
confidence, or a downstream VSQX smoke/import result. Those are autonomous
diagnostics or downstream gates.

## Downstream strict acceptance remains separate

The qualified forced-phone model route requires immutable weights/resources,
explicit non-NC rights, a complete phone inventory, and primary evidence of
Cantonese-singing training. The human-reviewed route requires G5 evidence as
defined above. Until one route is independently complete, MFA/Qwen/CTC,
lyrics/G2P, pitch fusion, Miku V4C, and VSQX outputs remain product or
evaluation artifacts. They may consume phone evidence after recognition; they
must not become the producer, recognizer, or internal truth layer.

When the producer gate and the core contract are green, update this document's
recovery matrix with the command output and artifact hashes, then stop only at
the first genuine human window described above.

## 2026-08-23 Reset Pass Addendum

The original route was reactivated after the downstream Miku/MFA route was
paused. The producer/runtime surface is now a contract-compatible reconstruction
behind `src/any2xsampa/adapters/recognizer_producer_runtime.py`; the two producer
CLIs import that facade. The prepared external worker is a source-resolving
wrapper rather than a copied CLI, so it can locate the repository through
`ANY2XSAMPA_REPO_ROOT` and fail clearly when the source tree is unavailable.

Verification from this pass:

- `32 passed` for producer recovery/reset, lineage, original-route surfaces,
  recognizer, pipeline, and X-SAMPA focused tests.
- `tools/posterior_json_producer.py --help` and
  `tools/onnx_espeak_bundle_packager.py --help` both succeed.
- `tools/posterior_json_producer.py --audio samples/demo.wav --backend
  onnx-espeak` generated `.state/runs/original-route-reset-20260823/demo.posterior.json`
  with `external_posterior_json_v1` and 34 segments.
- The posterior JSON was replayed through the real `posterior-json` adapter into
  `.state/runs/original-route-reset-20260823/demo.segments.json`: 34
  `PhoneSegment` records, `source_mode=real`, `backend_kind=posterior-json`,
  normalized IPA and X-SAMPA fields present.
- Runtime diagnostics report `runtime_status=in_process_ready` and
  `ready=true`; the external bundle remains `bundle_incomplete` because its
  portable Python and worker deployment are not present.

The CLI Luna audit layer was launched through `codex exec --ignore-rules` after
the local Codex rules file was found syntactically broken. Eight independent
read-only Luna scopes were requested; seven returned useful reports and the
large historical-lineage scan was stopped after it became an unbounded scan of
the 299 MiB recovery archive. No Terra or Sol fallback was activated because
the Luna layer produced results. The reports identified follow-up risks but no
new blocker to reaching the first human window.

The first genuine human window is now open. The required user-owned material is
one revision-matched triplet: (1) original-song WAV for audio-only recognition,
(2) the human-authored VSQX for the same song/section, and (3) the WAV rendered
from that exact VSQX. Record all three paths, revision/pairing identity, and
SHA-256 hashes. Do not provide lyrics, VSQX phones, note context, or rendered
phones to the first recognizer baseline. The old `8766` phone reviewer and
Miku/MFA artifacts are downstream and do not satisfy this window.

The user supplied two final-delivery triplets under
`C:\Users\Example User\Downloads\Miku_V4C_两首歌_最终交付` and stated that both
phone annotations are correct, with rhythm/pitch accuracy at least 90% for
`自由花` and at least 80% for `皇后大道東`. These are recorded as
`user_asserted_*` reference claims, not automatic model scores. Their hashes,
WAV metadata, VSQX note counts, and read-only normalized VSQX copies are in
`.state/runs/original-route-reset-20260823/reference/reference_triplet_manifest.json`.

The original `皇后大道東` VSQX remains untouched. Its XML declaration uses
`encoding='UTF8'`, which Python's strict XML parser rejects at the first CJK
comment character; replacing only the declaration with `UTF-8` parses 636 notes.
VOCALOID4FE opening the original file is compatible with this parser quirk and
does not imply that the source file is unusable.

The producer CLI now accepts explicit `--device auto|cuda|cpu` and `--model`.
It also accepts `--cpu-threads N`, which sets the ONNX Runtime intra-op CPU
thread limit without changing the provider contract; long CUDA jobs should use
one process at a time with a modest limit such as `--cpu-threads 4`.
Short CUDA smokes on the bundled demo and a 10-second `皇后大道東` clip reported
`cuda_active=true`, `provider_fallback_reason=null`, and
`[CUDAExecutionProvider, CPUExecutionProvider]`. An ORT profile for the latter
contained 1,447 CUDA nodes (183.492 ms) and 428 CPU nodes (1,201.823 ms). The
CPU load is caused by CPU graph partitions/memcpy plus NumPy/Python
post-processing and JSON serialization, not by a CPU-only Torch installation.
Full-song inference is paused until chunking and post-processing are optimized;
two long songs must not be run concurrently.

The first complete audio-only baselines were then run serially with
`--device cuda --cpu-threads 4 --chunk-seconds 30 --chunk-overlap-seconds 0.5`:

- Final hash-bound rerun: `自由花` 547 segments in 70.226 seconds;
  `皇后大道東` 301 segments in 68.768 seconds.

Both final outputs report `real_model_output=true`, `cuda_active=true`, verified
model/vocabulary/audio hashes, monotonic finite time ranges, and no lyrics/note
context or rendered-reference phones supplied. The output and summary hashes are stored
in the per-song subdirectories below
`.state/runs/original-route-reset-20260823/`; the triplet manifest records their
post-hoc relationship to the user's VSQX and rendered WAV.

The imported human reference corpus contains 1,100 locked notes (464 for
`自由花`, 636 for `皇后大道東`). A deliberately non-authoritative inventory
comparison found only 30.8% and 25.6% set overlap between baseline top symbols
and human reference tokens respectively. This is recorded in
`.state/runs/original-route-reset-20260823/reference/audio_only_reference_comparison.json`
as diagnostic divergence only: it is not a time-aligned accuracy score and does
not override the user's human phone truth.

The post-hoc timing reports are
`.state/runs/original-route-reset-20260823/reference/freedom_flower_timing_report.json`
and
`.state/runs/original-route-reset-20260823/reference/queen_road_e_timing_report.json`.
They confirm that both baselines were audio-only, but also show that this
general ONNX/eSpeak model supplies only about 5.75% of the human note-active
duration for `自由花` and 2.90% for `皇后大道東`. This is a model/domain
limitation, not a reason to overwrite the user's phone truth or stretch sparse
segments into continuous phone occupancy.

Producer `--note-context` is now explicit about its boundary: it is recorded as
`metadata_only_not_acoustic_input` and is consumed later by the pipeline's
`NoteContextApplier`/score-hint layers. It is never passed into the acoustic
ONNX decoder, so score context cannot silently become recognizer truth.

Direct external candidate lists are normalized by descending confidence before
they become `PhoneSegment.candidate_phones`; an input list's ordering can no
longer silently choose the top phone.

The producer/replay boundary now also binds the input WAV with SHA-256 and WAV
metadata, rejects mismatched audio, removes stale command output before running
a producer, and writes new posterior JSON through a temporary file replacement.

After provenance, candidate-order, and chunking changes, the full product suite
returned `488 passed, 2 skipped` with an isolated basetemp. The skips are the
two grpc bridge guards because grpcio is installed; no test failed.

## 2026-08-24 Continuous Cross-Song Evaluation

The active plan in `REDLINES.md` was executed through its next genuine human
window. The two user triplets were converted into a hash-bound evidence surface
without feeding lyrics or VSQX phones into original-song acoustic decoding:

- 1,100 human-locked VSQX notes were preserved as raw V4C reference tokens.
- A versioned, explicitly lossy V4C-to-recognizer-contract mapping projected the
  reference into the current 15-phone acoustic inventory while retaining IPA,
  articulatory features, and per-token loss notes.
- Phrase construction produced 29 `自由花` and 40 `皇后大道東` examples.
  Two folds hold out one complete song at a time; rendered WAV and normalized
  VSQX hashes do not overlap across folds.
- Dataset hash: `49132c5eaf6bcf180ecc0f7c5618efdf4e8655a0d553fb638e5d835c8d005f45`.

Two automatic feature routes used the same splits:

1. An 80-bin, 16 kHz log-mel linear CTC baseline reached micro similarity
   `0.167418`.
2. Full raw CTC logits from the verified ONNX/eSpeak model
   `74174710e34035bbb7f611601d016c32fc575de7a6f53b1078107dc10a84e7ae`
   were cached as a 392-dimensional pretrained acoustic surface. Bounded
   learning-rate/batch sweeps and additional seeds produced best held-out
   similarities `0.443315` and `0.476471`; best combined micro similarity was
   `0.456868`, and exact sequence ratio remained zero in both directions.

The machine-readable summary and per-phone/confusion reports are under
`.state/runs/continuous-eval-20260824/`, led by
`continuous-evaluation-summary.json`. Checkpoints, feature caches, mapping,
dataset, split identities, reports, and source files carry SHA-256 evidence.
No prediction changed or rejected a human annotation.

No additional eligible local backbone remains in the current route: Qwen3-ASR
is a text ASR asset without an existing phone-feature contract; Qwen3
ForcedAligner is downstream timing evidence; the local HubertFA asset lacks
Cantonese scope; and the SOFA Cantonese download is incomplete with model-weight
rights and singing-training domain unresolved.

The 0.80 evidence gate did not pass. This is human-window condition B. The next
smallest user action is to choose one of:

1. provide at least one additional complete original-WAV + human-VSQX +
   rendered-WAV triplet; or
2. authorize/select a materially different, immutable, hash/license-bound
   pretrained acoustic backbone route.

This window is a data/model-route decision, not permission to overwrite human
phones and not a request for downstream MFA/Miku phone-by-phone review.

The final current-tree verification for this phase returned `494 passed,
3 skipped` with no failures. The skips are the local Tcl/Tk runtime guard and
two grpc bridge guards.

## 2026-08-24 Pretrained Acoustic Backbone Route

The user selected the materially different pretrained acoustic backbone route.
The route was executed on the unchanged 69-example, two-song cross-song dataset;
the original-song crops were the only acoustic inputs. Lyrics, recognized text,
VSQX phones, notes, rendered-reference labels, and MFA boundaries were not passed
to any backbone. Human V4C phones remained post-extraction CTC targets only.

Three complete, immutable, non-NC snapshots were evaluated offline:

- WavLM Base Plus `4c66d4806a428f2e922ccfa1a962776e232d487b`, weights
  `3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51`;
- WavLM Large `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c`, weights
  `fdee460e529396ddb2f8c8e8ce0ad74cfb747b726bc6f612e666c7c1e1963c9d`;
- XLS-R 300M `1a640f32ac3e39899438a2931f9924c02f080a54`, weights
  `d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0`.

The WavLM model cards bind their official license to Microsoft UniSpeech's CC
BY-SA 3.0 text; that text was pinned at Microsoft commit
`6112826ac13a4327f4c9a7afa2a505e35b763514`. XLS-R declares Apache-2.0 in
the immutable model metadata and model-card front matter. Both licenses are
non-NC, but WavLM's attribution/share-alike obligations remain recorded and
must be preserved. MERT v1 95M was rejected without downloading weights because
revision `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5` declares CC BY-NC 4.0 and
custom model code.

All feature extraction ran serially on the RTX 4090 Laptop GPU with explicit
CUDA, four CPU threads, local-files-only loading, and no CPU fallback. The
frozen adapters cache hidden states in the existing CTC feature-cache contract.
Bounded representative-layer, learning-rate, batch-size, and seed experiments
produced these best cross-song results:

- WavLM Base Plus: micro similarity `0.255185`, minimum fold `0.238943`;
- WavLM Large: micro similarity `0.281335`, minimum fold `0.280147`;
- XLS-R 300M layer 12: fold similarities `0.483986` and `0.556618`, micro
  similarity `0.513676`, exact-sequence ratio zero in both folds.

XLS-R seed robustness runs ranged materially lower than the selection result,
confirming that no stable candidate approached the `0.80` evidence gate. The
machine-readable source of truth is
`.state/runs/pretrained-backbone-20260824/pretrained-backbone-route-summary-v2.json`
(SHA-256 `58ed3c367f97783a060675db50ec46871a8d76b00fb132f0218b0229461b866a`).
Detailed fold reports are under the sibling `reports/` directory. No model
prediction changed a human annotation.

The next genuine human window is condition B. The smallest decision is to pick
one route: another immutable non-NC backbone, additional complete triplets, or
a materially different full-backbone fine-tuning route. Full fine-tuning is not
an automatic extension of the frozen-feature experiment because the current two
song folds show severe train/held-out divergence and need a separately approved
overfitting/evaluation plan.

## 2026-08-25 Third Triplet Extension

The user supplied a complete third triplet for 月半小夜曲 / `half-moon-night-song`:
the original-song WAV, the human-authored `月半小夜曲_Miku_V4C_人工最终版.vsqx`,
and its `月半小夜曲_导出.wav`. The three files are recorded by absolute path and
SHA-256 in `.state/runs/continuous-eval-20260825/reference_triplet_manifest_v2.json`.
The manifest is UTF-8 and machine-verified; console mojibake in some Windows
displays does not alter the bytes. The new VSQX has 388 notes.

The existing v1 V4C mapping lacked `uo` and `ua_n`, so the automatic build
stopped instead of guessing. A versioned v2 mapping was added: `uo` is copied
from the existing reference-evaluation contract (`u,o`), while `ua_n` is an
explicit, lossy `u+a+n` projection. The raw V4C token remains preserved and the
loss is recorded; the old v1 mapping and old two-song artifacts are untouched.

The rebuilt three-song CTC evidence contains 101 phrase windows: 29 freedom-flower,
40 queen-road-east, and 32 half-moon-night-song. The corrected dataset builder
now creates three leave-one-song-out folds with the other two songs in each
training split. Dataset SHA-256 is
`1b2b5d641a3e70f48da73d7a052d11e901df7bb3820792a7c93443d841b9180d`; dataset
hash is `bfed464b200ad74def5f0435a90e2772c3c99e90822bbb0b426f4d731170f262`.

Using the already qualified XLS-R 300M snapshot at immutable revision
`1a640f32ac3e39899438a2931f9924c02f080a54`, hidden layer 12, explicit serial
CUDA, and the same frozen CTC-head route, held-out token similarities are:

- queen-road-east: `0.520081`;
- half-moon-night-song: `0.624334`;
- freedom-flower: `0.587500`.

Weighted micro similarity is `0.567033`, with zero exact-sequence ratio in all
three folds. The new song improves the evidence surface and is the strongest
held-out fold, but it does not reach the `0.80` gate and does not authorize
human-phone changes or product acceptance.

The new original-song audio-only baseline ran through ONNX/eSpeak with explicit
CUDA and no lyric/VSQX input. It produced 533 real segments, was replayed via
the real `posterior-json` adapter, and bound the input audio SHA-256 correctly.
The post-hoc timing report is
`.state/runs/continuous-eval-20260825/reference/half-moon-night-song_timing_bound_report.json`;
its note-duration coverage by sparse phones is `0.055187`, so it remains a
diagnostic baseline rather than a continuous phone-accuracy measure.

The current machine-readable route summary is
`.state/runs/continuous-eval-20260825/three-song-route-summary-v2.json`
(SHA-256 `65ec6982cdd1f4e5d253e8040e7336e76579eb0526d43d1a7e463beb5e382e27`). The
human-window condition remains B: choose another eligible backbone, supply more
complete triplets, or explicitly authorize a separate full-backbone fine-tuning
route.

Final verification after the third-triplet extension returned `499 passed,
3 skipped` in `50.23s`. The skips remain the local Tcl/Tk runtime guard and the
two installed-grpc bridge guards; no test failed.

## 2026-09-04 Additional Triplet Expansion

The user supplied two more complete delivery triplets:

- 约定 / `promise`: original WAV `28d79991d3c89edd40c8aa8641c353018246294d1a7730eaf1e50d8bcbc68d2f`, rendered WAV `f9bb442547500bd233771b9bfa9133141330e94738261ad268a20870e823a073`, VSQX `e173c8327fdd7b14b28879d6f998b5cad016f52ea8e69b59fa4868011af21c80`;
- 喜欢你 / `like-you`: original WAV `e82931e7bd2f765119d2890a73f27b5692aa0c23dce918528ac03a7aef12d76c`, rendered WAV `663263d061966d656f766729d9d6b509c87d6ff553ec697d4e5649076a30a965`, VSQX `274801ec546e29a6c3f4a9706505227ab447c1f5468bb497155ddeac18a6d65d`.

The final five-triplet manifest is
`.state/runs/continuous-eval-20260903/reference_triplet_manifest_final.json`
(SHA-256 `9c564f514363f5ed69332d562e5a27dec0c694c06e5045239bd356734b7da21d`).
The user's fully approved phoneme and `>=0.80` rhythm/pitch statements are
recorded as human claims only. All five original WAVs have a matching rendered
WAV and human VSQX; no source delivery file was copied or modified.

The V4C contract build found seven unresolved projection notes: one `7` and
three `uAN` notes in 约定, plus three `iUN` notes in 喜欢你. The raw human tokens
are retained in the reference manifest, while those notes are marked
`training_ready=false` and excluded from phrase windows. Existing evidence
supports `UN`, `ts\\_h`, `ts`` and `ts`_h`; no authoritative mapping was found for
`7`, `uAN`, or `iUN`, so no guess was made.

The resulting five-song dataset has 170 phrase windows and five leakage-safe
leave-one-song-out folds. Dataset hash is
`d44936dd05a6b02137da1b27e63d84a713e335001e3f14e85e2ec8b51964f3c9`, with file
SHA-256 `8f37db7aff3bed302479f78a418582c3063ce5e663261899e9a2cbdcbde62dc9`.
The existing XLS-R 300M layer-12 cache was rebound rather than recomputed only
after all 170 acoustic identity tuples matched exactly. Its feature tensor hash
remains `d0840c1ce087414e87a4ef98def1f00b527eb0acc04f0a8fe3a835f92f25e41a`,
and the final cache SHA-256 is
`1bd6db9cfaae16cf375260c47a878e785c5e080ae23b139675b89e6fa2ebecfa`. Rebound
CTC checkpoints are explicitly marked weights-unchanged; they are not claimed
as newly trained models.

Using the same XLS-R layer-12 frozen-head route, held-out similarities are:

| held-out song | token similarity |
| --- | ---: |
| 皇后大道東 | 0.281647 |
| 约定 | 0.677555 |
| 喜欢你 | 0.238286 |
| 月半小夜曲 | 0.116341 |
| 自由花 | 0.706618 |

Weighted micro similarity is `0.401203`, exact-sequence ratio is zero in every
fold, and the `0.80` gate remains closed. The new songs improve the provenance
surface but do not authorize automatic acceptance or modification of human
phones.

Audio-only ONNX/eSpeak CUDA baselines are bound for all five triplets. The two
new runs produced 375 and 207 real segments respectively; all five records now
show `cuda_active=true` with `[CUDAExecutionProvider, CPUExecutionProvider]`
and input SHA equal to the original WAV. Their post-hoc timing reports are
diagnostic sparse coverage only.

The canonical machine-readable report is
`.state/runs/continuous-eval-20260903/five-song-route-summary-final-v2.json`
(SHA-256 `924120e1f51135b915687da523177d12a66dc9ef7ac08a15fd5fe452eb95d877`).
It includes the five triplet records, backbone manifest, rebound cache and
checkpoint hashes, fold reports, and timing evidence. The route remains at
human-window condition B: choose another eligible backbone, provide additional
complete triplets, or explicitly authorize a separate full-backbone fine-tuning
plan.

During this continuation, three stale no-window `D:\\Anaconda\\python.exe -`
processes (PIDs 2572, 6412, 39576, all created 2026-08-31) were verified and
terminated only after explicit user authorization. They were not current
`.venv` inference processes and were not used as evidence for GPU behavior.

The subagent fallback chain was also exercised without nesting. Eight standard
Luna CLI workers exhausted relay retries with 502/503 errors, so three Terra
workers were launched on Flex. The Terra snapshot audit returned a useful
license-lineage finding; one other Terra worker exceeded its read-only scope and
was stopped, and one unbounded scan was stopped. The primary agent resumed the
interrupted proxy download and independently reverified every hash and runtime
result. Sol was not started because the Terra layer did not completely fail.
The machine record is
`.state/runs/pretrained-backbone-20260824/agent-orchestration-audit.json`.

Final verification after the adapter, CLI, report, redline, and documentation
changes returned `498 passed, 3 skipped` in `42.79s`. The skips are the local
Tcl/Tk runtime guard and the two installed-grpc bridge guards; no test failed.

## 2026-09-04 Continuation Closeout

The post-interruption continuation completed the final five-triplet evidence
binding and the regression pass. The final manifest is
`.state/runs/continuous-eval-20260903/reference_triplet_manifest_final.json`
(SHA-256 `9c564f514363f5ed69332d562e5a27dec0c694c06e5045239bd356734b7da21d`).
It contains five complete original-WAV / human-VSQX / rendered-WAV triplets;
all five audio-only baselines are marked `cuda_active=true` and bind their input
hash to the corresponding original WAV. The final five-song dataset file has
SHA-256 `8f37db7aff3bed302479f78a418582c3063ce5e663261899e9a2cbdcbde62dc9`,
dataset hash `d44936dd05a6b02137da1b27e63d84a713e335001e3f14e85e2ec8b51964f3c9`,
and 170 phrase windows across five leave-one-song-out folds.

The XLS-R layer-12 feature cache was rebound only after all 170 acoustic
identity tuples matched exactly; its feature tensor hash stayed
`d0840c1ce087414e87a4ef98def1f00b527eb0acc04f0a8fe3a835f92f25e41a`, and the
final cache SHA-256 is
`1bd6db9cfaae16cf375260c47a878e785c5e080ae23b139675b89e6fa2ebecfa`.
The five rebound checkpoint weights are unchanged. Held-out similarities remain
`0.281647` (皇后大道東), `0.677555` (约定), `0.238286` (喜欢你), `0.116341`
(月半小夜曲), and `0.706618` (自由花); weighted micro is `0.401203`, below
the `0.80` gate.

The final route report is the file named above. It includes the five triplets,
five timing reports, backbone/cache/checkpoint hashes, unresolved mapping notes,
and the condition-B human decision surface. The final focused regression pass
returned `35 passed`; the latest full suite returned `505 passed, 2 skipped` in
`49.73s`. The two skips are the installed-grpc bridge guards. No current route
inference or training process remains. The three
stale Anaconda Python processes previously identified by PIDs 2572, 6412, and
39576 were terminated only after explicit user authorization; they were not
current project processes and were not used as GPU evidence.

## 2026-09-06 Eight-Triplet Intake And Limited-Data Research

Local date: Asia/Taipei. The user supplied three further complete triplets
(`shalong`, `ke-yi-ke-zai`, and `king-of-karaoke`) and explicitly approved all
their phonemes, with rhythm/pitch accuracy asserted at >=0.95. These are
human-reference claims, not automatic scores. The eight-song reference has
3,060 notes and 270 legacy rendered-audio phrase windows across eight folds.

The new original WAVs completed serial ONNX/eSpeak CUDA inference, real JSON
replay, and post-hoc timing reports. Segment counts are 399, 514, and 430.
Every baseline binds to its original WAV hash and records CUDA active without
fallback. All 24 external source hashes and the previous three canonical
five-song artifact hashes were reverified unchanged. Exact commands are in
`.state/runs/continuous-eval-20260906/run_intake.py` and per-stage command logs.

This continuation corrects two earlier interpretations without rewriting
historical evidence. The five-song XLS-R feature input was rendered WAV crops,
not the original audio. Also, mapping v3 contains 26 damaged IPA entries,
affecting 2,022 current reference notes. Raw human tokens remain intact.
The legacy 15-label score and CTC event-window overlap are not rich-IPA
original-singing accuracy. No eight-song CTC head was trained around these
problems. The previous validation folds also selected their best epochs;
future final tests need a separate development split.

The user cannot substantially expand complete-triplet collection. The
recommended route is reference/measurement repair followed by a bounded,
phone-supervised multilingual model baseline on existing original recordings,
then small-parameter adaptation only if justified. The first proposed model
is PhoneticXeus; POWSM-CTC is the backup. Pinned metadata and small source
files were collected through the required proxy, but no new weights, runtime
execution, or dependency installation is claimed. Singing quality remains
unproven, and model/runtime authorization is the next human decision.

Canonical machine report:
`.state/runs/continuous-eval-20260906/continuation-report.json`
(SHA-256 `3e76903eaf330f6986ef34521f5d52884c6d963323cccaa088d0767d3e6904d7`).
Research, candidate license/revision details, experiment bounds, and primary
sources are in `docs/limited_data_recognizer_route_2026-09-06.md`.
The updated handoff is `HANDOFF_CURRENT_2026-09-06.md`.

Verification: `4 passed` focused and `508 passed, 3 skipped in 48.23s` full.
Skips are the local Tcl/Tk guard and the two installed-grpc bridge guards.
## Limited-Data Execution Follow-Up (2026-09-06 Asia/Taipei)

The later bounded implementation is documented in
HANDOFF_CURRENT_2026-09-06_EXECUTION.md and
.state/runs/limited-data-20260906T050251/. The final execution report provides
actual timing, source hashes, runtime locks, raw evidence, tests and limitations.

PhoneticXeus now runs as a native offline CUDA producer with stable
PhoneSegment/posterior replay. Existing phones/IPA truth and export boundaries
are preserved. V4 derived reference restoration repairs 23 damaged entries
without guessing unresolved tokens or modifying human VSQX.

All three small adapter seeds substantially reduce aggregate rendered
development PER, but none passes admission: aspiration-group PER increases
from 0.64 to 0.72, exceeding the predeclared regression limit. No checkpoint is
selected, no model becomes the default, and no original-singing accuracy or
final-test result is claimed. The three authorized optimization trials are
complete; further trials or new models require a new bounded decision.

The current human window is review-v2/index.html in that run: twelve focused
questions using existing songs. It is not a request for more complete triplets.
Section correspondence and engine-token semantics must remain distinct from
independently reviewed original phone transcription.

## Original Reference Implementation (2026-09-06 Asia/Taipei)

The user clarified that the approved phonemes were manually calibrated one
by one against the original in vocaloid4fe, then authorized implementation.
This is original-reviewed human reference evidence, not merely synthesizer
settings. The remaining engineering work must not be described as absence
of human review or used to request blanket reannotation.

The implementation and results are documented in
docs/original_reference_evaluation.md and HANDOFF_ORIGINAL_REFERENCE_2026-09-06.md.
Current artifacts are .state/runs/original-reference-eval-20260906T154652/offset-final/.
The route rebuilds/hash-checks sources, binds the user's review, separately
tracks timing proposals, and replays the original logits through the stable
real/fixture contracts. It does not retrain or modify the default backend.

The audio-only fixed-offset diagnostic was fitted outside the original
evaluation clips. Shalong estimates are +6.7802/+6.8731 seconds; the score's
four premeasures at 142 BPM provide a separate 6.7606-second observation.
Neither estimator agreement nor that metadata becomes human clip approval.

An optional, independently versioned supplement restores the documented
ua+n convention using intact IPA and checked source-table evidence. Shalong
coverage becomes 310/320 notes, and its three proposed evaluation sequences
are complete. All three corresponding short pairs remain pending; their
conditional PER values are not official original recognition accuracy.

Closing regression: 629 passed, 2 skipped. Independent artifact verification
replayed 18 original outputs, verified all external hashes, checked the six
sealed outputs by hash only, and validated six short previews totaling 108s.
Historical sources, mappings, dataset reports, and failed adapter results are
unchanged. No new full-song collection or complete phoneme reannotation is
requested; only the exact three generated Shalong correspondences remain.

## Shalong Order-Only Confirmation Correction (2026-09-06 Asia/Taipei)

The user clarified that Shalong's phoneme/note order is confirmed for the full
song and that an opening offset exists. This does not confirm the exact crop
boundaries automatically selected for the three 18-second diagnostics. The
corrected canonical run is `.state/runs/shalong-order-confirmed-20260906T173051/`
and keeps official original windows at zero. The earlier broad interpretation
is retained only as superseded audit evidence. Existing adapters were replayed
conditionally on the order/offset proposal, without retraining or deployment.

## Shalong Clock Rule Implementation

The subsequent authorized continuation versioned the original/rendered/score
clock equations under `.state/runs/shalong-clock-rule-20260906T232630/`.
See `HANDOFF_SHALONG_CLOCK_RULE.md` and `docs/shalong_clock_rule.md`.

Common-support offset fitting and independent rendered note/rest checks
support the distinction between waveform, score-absolute and score-display
origins. They do not yet pass formal baseline admission: the first local
offset estimate exceeds the locked stability tolerance and all three old
crops can cut partial edge notes. Safe outward-only recrops were not found
within the declared budgets. No omitted edge material was used to improve PER.

The remaining review is six cut points, with twelve short audio previews
totaling 30 seconds. Existing source phoneme and full-song order approval
remain accepted. This does not require full reannotation or another song.

Focused tests: 66 passed. Full regression: 668 passed, 2 installed-grpc skips.
Five timing artifacts replay byte-identically. Sources, historical diagnostics
and old adapter weights remain unchanged. There was no new training, model
selection, default replacement or final-test prediction access.

## Quality-first technical delivery (2026-09-07)

See `HANDOFF_QUALITY8H_2026-09-07.md` and the canonical run
`.state/runs/quality-first8h-20260907T010738/`. The user authorized a new fixed
separator/Qwen auxiliary route and three CTC+KL seeds. Those trials are now
consumed; seed43 is frozen from rendered development only, not deployed.
Protected development PER is .293078 versus pretrained .727541, with aspiration
PER .48 versus .64. All saved checkpoints were independently replayed.

The 18-context mix/vocal ASR and raw-timestamp pipeline is implemented and run,
but zero new safe windows satisfy the fixed rules. Timestamp repair is retained
only as diagnostic information. Original comparisons remain conditional, with
all unmatched/unknown windows reported. The user's full-song manual approval
is not revoked. The same six pending cut questions now have separated-vocal
auxiliary previews; no full reannotation or new song collection is requested.

An explicit Windows audio cache reduces the fixed warm benchmark median by
16.9% without changing features/logits or PhoneSegment outputs. Cold startup
and peak GPU reserved do not meet improvement targets. Full regression closes
at 719 passed / 3 existing environment skips. Source hashes, historical results,
reference reconstruction and model assets verify; final-test prediction
contents and default model remain untouched.

The continuous eight-hour promise was NOT fulfilled. A long interval without
work evidence separates the initial work and resumed implementation. It cannot
be counted as work or repaired retroactively by later delivery. The report
records technical delivery separately from this failed commitment.
