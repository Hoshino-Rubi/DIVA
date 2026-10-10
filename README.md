# Any2Xsampa

[简体中文](README_CN.md)

Local audio → IPA / phones → notes → VSQX / X-SAMPA.
The Windows desktop app and CLI share a local subprocess engine. Normal use does **not** require an HTTP or gRPC server.

## Windows research preview · pre-1.0.1a

Download these four files from the same GitHub Release and keep them in one folder:

- `Any2Xsampa-Setup.exe`
- `Any2Xsampa-Windows-x64-pre-1.0.1a.zip.001`
- `Any2Xsampa-Windows-x64-pre-1.0.1a.zip.002`
- `Any2Xsampa-Windows-x64-pre-1.0.1a.zip.003`

Run the installer and choose a writable directory. It verifies the volumes and extracts the complete application. Do not run a volume directly or copy only the desktop EXE.
Python, .NET and required libraries are bundled; no separate server or CUDA Toolkit installation is needed.

**Requirements:** Windows 11 22H2 or later, x64, an NVIDIA GPU and a driver compatible with CUDA 12.6 for phoneme recognition. GAME note transcription runs on four CPU threads. Allow at least 12 GiB of free disk space for installation, weights and caches.

The release is unsigned. Its self-contained installer, relocated runtimes and saved-result VSQX export have been tested locally; a GUI launch on a separate machine has not been verified.

### Automatic model setup

On first launch, the app downloads missing, revision-pinned PhoneticXeus and GAME weights from their upstream HTTPS sources (about 2.31 GiB total). File sizes and SHA256 hashes are verified before models become available.

The setup panel supports disabling startup downloads, HTTP(S) proxies, cancellation, explicit resumption and offline import.
Failures do not trigger automatic retry loops; partial downloads are retained. Inference stays offline.

For offline setup, download the three `PhoneticXeus-model.safetensors.00x` volumes and `GAME-1.0.3-medium-onnx.zip` from the Release. Put them in one folder, then select **Import offline models**. No manual concatenation or archive extraction is required.

## Desktop workflow

1. Finish model setup, open a WAV file and explicitly select `guarded-candidate-v2`.
2. Choose the whole file or an interval. Keep automatic VSQX generation enabled.
3. Start recognition, then export the generated VSQX.

The note panel can reuse saved GAME predictions without rerunning inference.
Cancellation affects only application-owned jobs. Missing real models fail clearly; fixture data is never silently substituted for recognition.

## Diagnostics and updates

In **Model environment and saved-result replay**, choose **Export diagnostic bundle**. It creates a local ZIP even when Python or model setup failed. It includes version, allowlisted error codes, expected model hashes/file sizes, memory counters and recent task timings. Audio, predictions, raw logs, complete paths, usernames and proxy credentials are excluded; nothing uploads automatically. Matching file sizes are not reported as verified hashes.

Install this version into a new folder. Existing installations, models and results are not overwritten. You can reuse the verified offline model assets from pre-1.0.0.

## CLI

The portable application includes `any2xsampa.cmd`. Run from the installation directory:

~~~bat
any2xsampa.cmd models status
any2xsampa.cmd models install --proxy http://127.0.0.1:18080
any2xsampa.cmd models install --offline-dir "D:\models"
any2xsampa.cmd models list
any2xsampa.cmd recognize input.wav --model guarded-candidate-v2 --device cuda --whole-file --output-dir runs/phones
any2xsampa.cmd score runs/phones/result.json --backend game --output-dir runs/score
~~~

Use your actual proxy address, not the example unless it matches your setup.
The source checkout uses the equivalent `python -m any2xsampa.cli` entry point but contains neither runtimes nor weights. Configure dependencies and model packages separately; automatic downloads require a verified download catalog.

### Export saved results with zero inference

~~~text
python -m any2xsampa.cli export-vsqx prediction.json --phones result.json --output song.vsqx
python -m any2xsampa.cli export-vsqx score.json --output song.vsqx
python -m any2xsampa.cli score result.json --notes prediction.json --output-dir runs/score
~~~

`prediction.json` is a saved GAME `note_transcription_v1` result; `result.json` contains phones in the same original-audio time coordinates. Source audio hashes must match. No offset or octave shift is guessed.

Notes-only export is supported. Notes without usable phones retain their pitch and timing and are explicitly marked as placeholders with neutral `a` pronunciation; they do not claim to recover lyrics.

## Output guarantees and limitations

- Generates actual VSQ4 XML with MIDI pitch, onset, offset, rests and repeated notes.
- Reads exports back to verify note count, pitch and timing. At 120 BPM / 480 PPQ, endpoint tick rounding is at most about 0.521 ms. This is serialization error, **not recognition accuracy**.
- Processes whole songs in overlapping windows of at most 20 seconds. Only continuing same-pitch notes supported on both sides are joined; real repeated notes remain separate.
- CTC emission times are not note durations. Lyrics, VSQX and manual references are not GAME decoding conditions.
- Unmapped phones preserve original IPA and explicit placeholder status. Notes are not deleted to hide mapping gaps.
- Original audio, manual projects and existing outputs are not overwritten. No extra manual review is required.
- **Correct file generation does not establish accurate transcription.** Formal joint pitch/onset/offset accuracy has not been validated against independent original-singing note ground truth. This is a research preview, not an accuracy-certified release.

See [VSQX documentation](docs/vsqx_generation.md), [distribution instructions](docs/distribution/README-使用说明.md) and [legacy reference workflows](docs/legacy_product_workflows.md). Legacy research workflows are not required for automatic generation.

## Development verification

~~~text
python -m pytest tests/test_model_install.py tests/test_game_vsqx.py tests/test_local_score.py tests/test_local_engine.py tests/test_local_whole_audio.py tests/test_local_desktop_layout.py -q -p no:cacheprovider
dotnet build apps/Any2Xsampa.Desktop/Any2Xsampa.Desktop.csproj --no-restore -c Debug -p:Platform=x64 -p:WindowsPackageType=None
dotnet run --project apps/Any2Xsampa.Setup.Tests/Any2Xsampa.Setup.Tests.csproj -- <new-test-output-directory>
~~~

`tools/split_portable_release.py` generates the installer's manifest, embedded using `ReleaseManifestPath`. Runtime packaging uses existing audited installations; release tools do not train or silently download new models.

Run the Windows CI entry with `python tools/check_windows_release.py --output <new-directory>`. It covers installer faults, model installation failures, diagnostic privacy and relocated stdio/VSQX replay without loading weights. Build inputs and signing are documented in [release build instructions](docs/release_build.md).

The full historical research suite needs additional unpublished artifacts. Public product checks do not depend on those artifacts.

Stable `PhoneSegment`, posterior JSON and explicit fixture/replay contracts remain supported. X-SAMPA is an export representation, not the recognizer truth layer.

## Licenses

PhoneticXeus uses Apache-2.0. GAME's pretrained weights use **CC BY-NC-SA 4.0**, not the MIT license of its source code; noncommercial and share-alike restrictions apply to covered model material.
See [third-party notices](docs/distribution/THIRD_PARTY_NOTICES.md) and license texts. The repository owner has not selected a license for the application's own source yet.
