# Phase 1 Miku V4C VSQX Pipeline Plan

Date: 2026-06-27

## Product direction

The first product slice is Windows-first:

```text
Cantonese audio on Windows
-> optional vocal cleanup / separation evidence
-> optional lyric evidence
-> Cantonese phones / IPA / articulatory evidence
-> Miku V4C VOCALOID phoneme-token export
-> VOCALOID4 FE importable and playable VSQX
```

Ground rules for this slice:

- The internal truth layer remains phones / IPA / articulatory evidence.
- X-SAMPA and VOCALOID phoneme strings are export layers only.
- Miku V4C is the first supported singer target.
- Users must not be required to know or provide Jyutping.
- Lyrics are optional user input. If the user supplies lyrics, they are helpful evidence; if not, the pipeline may use ASR lyric evidence.
- Qwen3-ASR 1.7B is planned as lyric evidence, not as phone/IPA truth.
- Future audio preprocessing should be adapterized and Windows-friendly. The intended stack is BS-RoFormer for vocal separation, followed by UVR-DeEcho-DeReverb and UVR-DeNoise where needed.

## Phase 1 goal

Phase 1 is not full automatic song transcription. Its goal is a narrow,
reproducible Miku V4C export target:

- build a short Cantonese phrase candidate from the project phone evidence;
- export a VSQX by cloning a known-good VOCALOID4 template;
- import the result in VOCALOID4 FE on Windows;
- assign or resolve the singer as Miku V4C;
- play the phrase audibly without silent notes caused by invalid phoneme tokens.

The current phrase-level smoke target is:

```text
??? = x ei / f u n / n ei
```

## Phase 1 acceptance criteria

Phase 1 is accepted only when the generated candidate is both importable and
playable in the local VOCALOID4 FE + Miku V4C environment.

Required acceptance checks:

1. The VSQX is produced through the template-clone path, not only by a minimal XML scaffold.
2. VOCALOID4 FE opens/imports the file without an `Invalid VSQ4 File` or equivalent parse/import error.
3. The editor shows a vocal track and the expected note count for the short phrase.
4. Tempo, time signature, note positions, note durations, lyrics, and note phoneme fields are visible and plausible.
5. The singer is Miku V4C, or the track can be explicitly switched to Miku V4C without changing the note phoneme contract.
6. Playback is audible for every note in the phrase.
7. The accepted phrase uses the current playable Miku V4C token sequence:
   - `喜` / `hei`: `x ei`
   - `歡` / `fun`: `f u n`
   - `你` / `nei`: `n ei`
8. The accepted candidate does not use the known silent token patterns <code>i\</code>, <code>i&#96;</code>, bare <code>@N</code>, or <code>f UN</code> in the accepted phrase.
9. The manual validation result is recorded as evidence under `tmp_validation/` or another generated-output location, not as source truth.

Known silent token shorthand: i\ / i` / bare @N / f UN.

Non-acceptance signals:

- XML well-formedness alone is not enough.
- VSQX import alone is not enough if playback is silent.
- A token appearing in PHG2 or in the editor is not enough unless it is audible in the Miku V4C VSQX note `p` context.

## Phase 1 non-goals

- No WinUI or packaged desktop client is required for Phase 1.
- No full automatic lyrics transcription quality claim is required.
- No user-facing Jyutping workflow is required.
- No broad Miku V4C phoneme inventory coverage is required beyond the accepted short phrase.
- No new external dependency should be added unless it is behind a replaceable adapter.

## Later phases

### Phase 2: optional evidence automation

- Add the Windows-first audio preprocessing chain behind adapters:
  BS-RoFormer, UVR-DeEcho-DeReverb, and UVR-DeNoise.
- Add Qwen3-ASR 1.7B as optional lyric evidence.
- Keep lyrics optional: user-provided lyrics should override or constrain ASR evidence, but the user should not have to provide lyrics.
- Keep Jyutping optional and internal/debug-only; the product path should accept normal lyrics or no lyrics.
- Preserve the phone / IPA / articulatory truth layer before VOCALOID export.

### Phase 3: Miku V4C phoneme coverage

- Expand phrase probes with short natural phrases, not isolated single characters.
- Record PHG2-derived candidate tokens separately from VSQX note-context playback results.
- Promote only phrase-playable token sequences into default mappings.
- Keep known silent tokens blocked or marked as unsafe until phrase-context evidence changes.

### Phase 4: Windows product workflow

- Wrap the evidence chain into a Windows-first operator flow.
- Make import/playback validation artifacts easy to reproduce.
- Keep external model, audio cleanup, ASR, pitch, and VSQX export components replaceable.

### Phase 5: broader singer/export support

- Add other singers only after the Miku V4C contract is stable.
- Treat each singer as its own token and playback-validation target.
- Do not assume that Miku V4C PHG2 token behavior transfers to another VOCALOID singer.



## Draft product runner

The current first product-facing entrypoint is:

```powershell
$env:PYTHONPATH='src'
$env:PYTHONIOENCODING='utf-8'
python tools/build_miku_v4c_draft_vsqx.py `
  --lyrics "???" `
  --template tmp_validation/vocaloid_probe/phrase_probe_round2/miku_v4c_phrase_probe_round2.vsqx `
  --output-root tmp_validation/miku_v4c_product_draft/current `
  --pitches 64,67,69 `
  --pretty
```

It emits:

- `miku_v4c_draft_phoneme_lattice.json`
- `miku_v4c_draft_draft.json` (`vocal_track_draft_v1`)
- `miku_v4c_draft.vsqx`
- `miku_v4c_draft_export_summary.json`
- `miku_v4c_draft_product_summary.json`

The runner is deliberately labelled as a draft-product lane, not final automatic transcription. It keeps `model_output=false`, `training_data=false`, `user_jyutping_required=false`, and requires manual VOCALOID4FE import/playback review before any phoneme candidate can be promoted.



## Audio-facing draft runner

The next pre-UI entrypoint is:

```powershell
$env:PYTHONPATH='src'
$env:PYTHONIOENCODING='utf-8'
python tools/build_miku_v4c_audio_draft_vsqx.py `
  --audio samples/demo_real.wav `
  --lyrics "???" `
  --template tmp_validation/vocaloid_probe/phrase_probe_round2/miku_v4c_phrase_probe_round2.vsqx `
  --output-root tmp_validation/miku_v4c_product_draft/audio_current `
  --pitches 64,67,69 `
  --pretty
```

This command adds the audio evidence envelope before the draft VSQX runner. If user lyrics are supplied, they override ASR text. If no lyrics are supplied, the same runner can consume `asr_transcript_v1`, a Qwen fixture result, or a real Qwen3-ASR model path. Rough note evidence can come from `pitch_note_analysis_v1` or the optional pitch analyzer. This is still a draft lane: ASR text is lyric evidence, pitch is rough note evidence, and VOCALOID4FE manual playback remains required.



## WinUI-facing gRPC contract skeleton

The first desktop boundary is now represented by:

- `proto/any2xsampa/v1/miku_v4c_product.proto`
- `src/any2xsampa/grpc/product_bridge.py`
- `src/any2xsampa/grpc/server.py`

The `InferDraftVsqx` request carries audio path, VOCALOID4 template path, output root, optional lyrics/Jyutping, optional ASR transcript, optional pitch analysis, and fallback pitch/tempo parameters. The response returns `ok`, error fields, product summary JSON, and generated artifact paths. The Python bridge is tested without requiring `grpcio`; the network server is enabled later via the optional `grpc` extra and generated protobuf stubs. This keeps the WinUI 3 client boundary concrete without blocking the current VSQX product lane on packaging.



## WinUI 3 desktop skeleton

The Windows client skeleton is now in `apps/Any2Xsampa.Desktop/` with solution file `any2xsampa-windows.sln`. The app uses WinUI 3/Windows App SDK, Mica, Fluent card-style layout, `x:Bind`, MVVM Toolkit source generators, and dependency injection. `MainPage.xaml.cs` only initializes the injected ViewModel; it does not call gRPC or perform inference. `GrpcAudioInferenceService` is the isolated service layer that calls `InferDraftVsqx`.

Current verification:

```powershell
dotnet build any2xsampa-windows.sln -c Debug -p:Platform=x64
$env:PYTHONPATH='src'; python -m pytest -q tests/test_winui_desktop_scaffold.py
```

This establishes the desktop shell and service boundary. Backend auto-start, file pickers, installer packaging, and richer result visualization remain later product work.

## Reproducible smoke command

Run the Phase 1 machine smoke lane with one command:

```powershell
$env:PYTHONPATH='src'
$env:PYTHONIOENCODING='utf-8'
.venv\Scripts\python.exe tools\run_miku_v4c_phase1_smoke.py `
  --output-root tmp_validation\vocaloid_probe\phase1_smoke_latest `
  --summary-output tmp_validation\vocaloid_probe\phase1_smoke_latest_summary.json
```

The smoke runner builds the Cantonese draft, exports through the template-clone
import-safe VSQX path, and verifies the XML/phoneme contract. A passing machine
smoke has `machine_ok=true` and `manual_status=pending` until VOCALOID4 FE
playback is manually recorded.

After manual import/playback validation, record the observation with:

```powershell
.venv\Scripts\python.exe tools\record_miku_v4c_playback_validation.py `
  --vsqx tmp_validation\vocaloid_probe\phase1_smoke_latest\miku_v4c_cantonese_smoke.vsqx `
  --output tmp_validation\vocaloid_probe\miku_v4c_cantonese_smoke_manual_playback.json `
  --imported yes `
  --all-notes-audible yes `
  --notes "Imported into VOCALOID4 FE and all smoke notes were audible with Miku V4C"
```

Then require the manual evidence in the smoke runner:

```powershell
.venv\Scripts\python.exe tools\run_miku_v4c_phase1_smoke.py `
  --output-root tmp_validation\vocaloid_probe\phase1_smoke_latest `
  --manual-validation tmp_validation\vocaloid_probe\miku_v4c_cantonese_smoke_manual_playback.json `
  --require-manual
```
