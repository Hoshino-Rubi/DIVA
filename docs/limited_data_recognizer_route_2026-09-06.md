# Limited-Data Recognizer Route

Local audit date: 2026-09-06, Asia/Taipei. This is a research recommendation,
not an authorization to download new weights, run remote model code, or start
full-backbone fine-tuning. The current user authorized three additional
triplets and asked for a viable path that does not depend on collecting many
more complete songs.

## Recommendation

Use the eight existing triplets as calibration and evaluation material, not
as the sole source from which to learn a universal phone recognizer. Repair
the reference representation and evaluation boundaries first. Then evaluate
an already phone-supervised multilingual model on original singing audio.
Only after that baseline exists should a small adapter or limited-layer
fine-tune be considered. Do not repeat the random linear-head sweep or
replace it immediately with unconstrained full-backbone training.

The first proposed candidate is `changelinglab/PhoneticXeus`; the backup is
`espnet/powsm_ctc`. Neither has been run here, and neither is established by
the reviewed sources as a qualified Cantonese-singing model. Their published
speech results are not forecasts of this project's singing accuracy.

## Completed Intake

The user explicitly approved all new phonemes and asserted rhythm and pitch
accuracy of at least 95%. These are human-reference claims, not model scores.
The final delivery directory remains read-only.

| Song ID | Reference notes | Legacy phrase windows | Original-audio real segments |
| --- | ---: | ---: | ---: |
| shalong | 320 | 38 | 399 |
| ke-yi-ke-zai | 324 | 33 | 514 |
| king-of-karaoke | 365 | 29 | 430 |
| All eight songs | 3,060 | 270 | Not an accuracy metric |

The delivered Shalong VSQX matches the user's `v1.3final` working file byte
for byte. The delivered Ke-yi-ke-zai VSQX matches the user's `v1.3_final`
working file byte for byte. Do not replace these with the older files still
visible in IDE tabs. Exact render-to-VSQX pairing remains a user delivery
assertion, not something proved by filenames, duration, or hashes alone.

All three new original WAVs completed serial ONNX/eSpeak inference with
explicit CUDA, four CPU threads, 30-second chunks, and 0.5-second overlap.
Every new posterior was replayed through the real `posterior-json` adapter,
bound to its original WAV hash, and compared with the VSQX only afterward.
No lyrics, note context, rendered waveform, or human phones entered decoding.
All 24 original/rendered/VSQX source hashes were rechecked at closeout.

The canonical run is `.state/runs/continuous-eval-20260906/`:

- `reference_triplet_manifest_final.json`: eight triplets and bound baselines.
- `reference-ctc-manifest-legacy-v3.json`: raw human tokens and legacy projection.
- `eight-song-dataset-legacy-v3.json`: 270 rendered-audio examples, eight folds.
- `eight-song-integrity-audit-final.json`: source roles and representation failures.
- `research/candidate-metadata-lock.json`: immutable candidate/source metadata.
- `continuation-report.json`: verification, hashes, limits, and recommendation.

The legacy dataset is retained for reproducibility. It is **not cleared for
new quality claims or another training campaign**. No eight-song CTC head was
trained and no old feature cache was relabeled as an eight-song cache.

## Why More Triplets Alone Are Insufficient

### 1. The reference IPA contains actual recovery damage

The v3 mapping contains 26 entries with literal question marks, replacement
markers, or CJK characters in fields designated as IPA. For example, the
`a` entry contains U+003F, and aspirated entries contain U+5906. This is file
content, not merely a PowerShell display problem. The eight-song audit finds
2,022 reference notes touching these entries.

Raw human V4C tokens remain intact. The human annotations were not rejected
or changed. The problem is the software's derived representation. Do not
reverse the existing lossy ONNX map to reconstruct IPA: that map deliberately
merges distinct phones and therefore has no unique inverse.

The currently unresolved tokens are `7`, `uAN`, `iUN`, and the newly observed
`iE_r`. There are 27 unresolved notes across eight songs, including 20 among
the three new songs. Their exclusion is a software mapping limitation, not
a retraction of the user's phoneme approval.

### 2. The CTC evaluation input is rendered audio

All 270 example `audio_sha256` values match the rendered-WAV role, not the
original-WAV role. `FrozenWavLMFeatureExtractor.extract` and its XLS-R subclass
read `audio_path`; the separate `original_audio_path` field is evidence for
identity/leakage checks, not an automatic source switch.

The previous five-song scores, including micro similarity `0.401203`, must be
described as rendered-audio, reduced-inventory validation results. They are
not measured original-singing recognition accuracy. The historical report
bytes remain unchanged; this document corrects their interpretation.

Simply changing `audio_path` to the original WAV is not a fix. VSQX/rendered
and original recordings do not have a verified shared timebase. In particular,
Ke-yi-ke-zai's full-file durations differ by about 67.07 seconds. This does not
prove incorrect pairing or incorrect user rhythm claims: intros, endings,
arrangement, and export extent may differ. It does mean identical crop
timestamps must not be assumed valid on both sources.

### 3. Fifteen output symbols discard useful distinctions

The legacy projection has only 15 targets. It merges, among other distinctions,
`f` into `s`, `x` into `s`, `y` into `u`, and aspirated stops into unaspirated
stops. More examples cannot recover distinctions absent from the target space.
A richer phone model must retain IPA, aspiration, place, and front rounding;
engine aliases should still be produced only at the export boundary.

### 4. Sparse CTC event windows are not phone-duration truth

The current adapter ends a segment at blank output. New baseline segments
average roughly one 20-ms frame. The post-hoc note-duration overlap ratios
are `0.035711`, `0.047201`, and `0.046150`. These measure that adapter's sparse
emission windows on uncalibrated reference timelines. They do not measure
phone error rate, and do not by themselves prove that the remaining duration
was recognized incorrectly. Stretching those windows would not fix recognition.

### 5. The reported validation songs selected checkpoints

`train_ctc_head` uses its validation loader for early stopping and best-epoch
selection. The report then evaluates that same validation fold. Song-level
isolation prevents direct training-example overlap, but those numbers are
development-validation scores, not untouched final-test scores. Future model,
layer, seed, and epoch selection must use a separate development split.

## Candidate Research

Only primary model cards, author repositories, papers, and immutable metadata
were used. Small metadata/source files were downloaded through the required
`http://127.0.0.1:10808` proxy. No model weights or remote code were executed.
Weight hashes below are upstream LFS declarations, not local weight verification.

### First: PhoneticXeus

- Model: `changelinglab/PhoneticXeus`.
- Revision: `8d83dee94817a07dc150f87d08f7e0ee01bdb66d`.
- Model card declares Apache-2.0 and training on 70+ speech languages.
- Its interface provides mono 16-kHz audio to frame-level CTC logits with 428
  labels. That is a closer match to the project's acoustic-phone contract than
  starting with a generic frozen representation and a new random 15-label head.
- `model.safetensors`: 2,300,089,432 bytes; declared SHA-256
  `ad58bf20a60e9d0380327bd8b2d0e8e90a9b8de2adccbfb479f9b21ea85eda18`.

The model card and paper establish speech-domain relevance, not singing
qualification. The source review also found concrete integration work:

1. The model card uses `trust_remote_code=True`; 40 source/config/doc files
   have been pinned for review, but static reading is not a runtime audit.
2. The wrapper has a `snapshot_download` fallback without an immutable revision
   when given a repository name. A future adapter must load an explicit local
   snapshot, prohibit network at runtime, and never invoke that fallback.
3. The vendored code imports a generic `src` package and mutates `sys.path`.
   Keep it in a separate worker process so it cannot shadow application imports.
4. The present environment lacks `torchaudio` and `typeguard`. Resolve exact,
   Torch-compatible dependencies in an isolated runtime, not by blindly running
   the model card's unpinned install command or replacing the existing Torch stack.
5. A missing optional error calculator becomes a no-op in one training path.
   Use the project's independent evaluator; do not treat that as model metrics.
6. Use safetensors; avoid the alternative pickle checkpoint and unsafe loading
   paths. Preserve raw logits, label IDs, vocabulary hashes, and input identity.

### Backup: POWSM-CTC

- Model: `espnet/powsm_ctc`, with an underscore, not `powsm-ctc`.
- Revision: `db47e3d2b7de83de4b572b5df7ea379e23cf8913`.
- Model card declares CC-BY-4.0. It is a phone-supervised encoder/CTC model.
- The documented runtime is ESPnet with the phone-recognition task selected;
  it expects 16-kHz audio and a fixed 20-second input policy.
- Main checkpoint: 1,702,754,550 bytes; declared SHA-256
  `d9d56a166e5dfe3088ca2080c8c3e2eb8124827166a68cd41b673963f99c9ef8`.

The current environment lacks ESPnet and its model zoo. This is a heavier
integration than the first candidate and needs its own pinned, isolated
runtime. Do not use the text-ASR mode as an acoustic-phone shortcut. Full-song
chunking must not silently truncate the song to 20 seconds.

### ZIPA: distinguish versions and licenses

`anyspeech/zipa-cr-s` at `50af0a503d44b9675d9804f587ca89a44e26a14a`
declares Apache-2.0, but the repository contains many approximately 1.03-GB
checkpoints and almost no usage documentation. A particular checkpoint,
loader, tokenizer, and averaging policy still need to be locked.

The smaller ONNX candidate `anyspeech/zipa-small-crctc-ns-700k` at
`86ce6920bcf23f27f449fd21b77b9886804bd734` has a 70,677,672-byte INT8 model,
but its pinned metadata/card does not declare a weight license. It is not
automatically eligible. A library license, another ZIPA revision's license,
or a downstream conversion's license cannot supply missing upstream rights.
Keep it blocked pending explicit, revision-specific weight-license evidence.

## Bounded Next Experiment

The following is proposed, not executed or automatically approved.

### Phase 0: reliable representation and measurements

- Create a new, evidence-backed mapping version; preserve v3 and raw V4C labels.
  Do not guess unresolved tokens or change test expectations to accept damage.
- Keep engine-intended pronunciation and actual acoustic realization distinct.
  A VOCALOID alias approved for rendering is not automatically an exact IPA
  transcription of the original singer.
- Select three diverse 15-20 second original-audio clips per song: 24 clips,
  approximately 6-8 minutes total. This is a bounded validation/review set from
  existing recordings, not a request for more complete song triplets.
- First verify source/section correspondence after audio-only extraction.
  Existing VSQX phone sequences may supply evaluation references only where
  their correspondence is confirmed. Request focused correction only where
  the original realization or section correspondence is genuinely ambiguous.
- Predeclare a five-song training/calibration, one-song development, two-song
  test split before looking at new candidate scores. Keep rendered versions,
  crops, augmentations, and pseudo-labels of test songs out of training.
- Define fixed Unicode/phone normalization and report PER, sequence similarity,
  PFER/feature errors, per-phone errors, and separate timing diagnostics. Do not
  collapse aspiration or rounding to improve a score. The old 15-symbol score
  is a separate baseline; new metrics require a documented gate definition.

### Phase 1: audio-only phone-supervised baseline

- Authorize only the first candidate's immutable snapshot and adapter work.
  Download one safetensors weight file, not duplicate checkpoints or all models.
- Integrate behind an isolated real adapter with fixture support and the stable
  `PhoneSegment` / posterior JSON contract. No text/VSQX/score decoder input.
- Start with one 8-second original-audio clip, batch size one, explicit CUDA,
  four CPU threads, and offline loading. Record real peak memory and elapsed
  time before advancing to the 24-clip set. No full-song run until this passes.
- Proposed safety budget: model-process reserved GPU memory <=7 GiB and total
  GPU use <=14 GiB on the 16-GiB device. These are abort limits, not predictions
  that the model fits. Do not kill unrelated user processes to meet the budget.
- Use the untouched pretrained phone head first. Preserve all raw phone labels
  and CTC blank evidence, rather than projecting immediately into 15 labels.
- If integration, license, or runtime gates fail, stop that candidate and
  consider the separately approved backup. Do not start a model sweep.

### Phase 2: small adaptation only if the baseline justifies it

- Prefer a bounded adapter/LoRA or limited-upper-layer experiment to full-model
  tuning. LoRA is a general low-rank adaptation method; its benefit for this
  singing task is an experiment, not an established result.
- One proposed comparison is rank 8 versus rank 16, three fixed seeds, at most
  six runs. Freeze the acoustic frontend and most encoder parameters. Keep the
  existing phone-supervised head as initialization instead of a random head.
- Select runs and early stopping on the development song only. Require a
  consistent development improvement without damaging important phone classes
  before opening the final test set once.
- Approved rendered clips can supplement training with bounded time, pitch,
  and channel augmentation. Their derivatives remain in the same song group.
  Synthetic augmentation does not establish real-domain accuracy or create
  additional independent test songs.
- Original-audio pseudo-labels are machine evidence, never human approval.
  Keep their provenance, confidence, and disagreement separate; do not train
  on held-out song audio, even without its human labels. Do not begin iterative
  self-training until a trustworthy original-audio baseline exists.

### Stop and review conditions

Stop on damaged/unresolved reference semantics, absent license evidence,
offline-runtime failure, memory-budget violation, or lack of consistent
development improvement. Do not respond by expanding the dataset indefinitely.
Any candidate meeting a predefined evidence gate still needs human listening
approval; neither 0.80 similarity nor passing tests permits annotation changes.

## Verification and Limits

New audit tests: `4 passed`. Full suite: `508 passed, 3 skipped in 48.23s`.
Skips are one Tcl/Tk environment guard and two installed-grpc bridge guards.
The three previous canonical five-song hashes still match. Original audio,
human VSQX, old caches, checkpoints, and reports were not modified.

The intake/research tasks are complete. New-model runtime execution, repaired
IPA mapping, original-audio reference calibration, and adaptation are not
claimed complete. No new weights or dependencies were installed this turn.

## Primary Sources

- R1: PhoneticXeus model card and pinned source, `https://huggingface.co/changelinglab/PhoneticXeus`.
- R2: An Empirical Recipe for Universal Phone Recognition, `https://arxiv.org/abs/2603.29042`.
- R3: POWSM-CTC model card, `https://huggingface.co/espnet/powsm_ctc`.
- R4: PRiSM: Benchmarking Phone Realization in Speech Models, `https://arxiv.org/abs/2601.14046`.
- R5: ZIPA author repository, `https://github.com/lingjzhu/zipa`.
- R6: Exact older ZIPA model card, `https://huggingface.co/anyspeech/zipa-cr-s`.
- R7: Exact small ONNX ZIPA model card, `https://huggingface.co/anyspeech/zipa-small-crctc-ns-700k`.
- R8: LoRA: Low-Rank Adaptation of Large Language Models, `https://arxiv.org/abs/2106.09685`.

The research lock records revisions, retrieved document hashes, upstream weight
hash declarations, proxy policy, and the distinction between downloaded
metadata/source and weights that were not downloaded.
