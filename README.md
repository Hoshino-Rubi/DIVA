# DIVA

[简体中文](README_CN.md)

**DIVA (Dialect Intonation & Vocal Automation)**, formerly Any2Xsampa, is a local vocal-production pipeline for preparing singing audio as VOCALOID-editable VSQX projects. Phones and the International Phonetic Alphabet (IPA) form its intermediate layer; X-SAMPA is an export representation. DIVA aims to reduce manual phonetic entry for Chinese dialect singing, including Cantonese and Hokkien.

The current implementation combines PhoneticXeus-based phone recognition, a project-specific guarded candidate and GAME note transcription. Dialect coverage and transcription quality remain subject to the limitations below. It does not promise perfect alignment or provide end-to-end synthesized vocal audio rendering.

Local audio → IPA / phones → notes → VSQX / X-SAMPA.
The Windows desktop app and CLI share a local subprocess engine. Normal use does **not** require an HTTP or gRPC server.

## Naming transition

DIVA is the product name starting with **v1.0.2**, at [Hoshino-Rubi/DIVA](https://github.com/Hoshino-Rubi/DIVA). The old **v1.0.1** binaries still display Any2Xsampa and are unchanged, including `Any2Xsampa-Setup-v1.0.1.exe`. The `any2xsampa` Python package and commands, internal executable names, environment variables and data directories retain their existing names for compatibility. No settings, models or saved results need to be moved. See the [branding migration notes](docs/branding_migration.md).

## Windows x64 · v1.0.2

Download **DIVA-Setup-v1.0.2.exe** from the [v1.0.2 Release](https://github.com/Hoshino-Rubi/DIVA/releases/tag/v1.0.2) and run it. Setup downloads the fixed application volumes, checks their sizes and SHA256, then installs into a new folder. Partial downloads support explicit resumption; failures do not start retry loops. Public automatic installation requires an anonymously accessible, published Release.

For offline installation, keep these beside Setup:

- `DIVA-Windows-x64-v1.0.2.zip.001`
- `DIVA-Windows-x64-v1.0.2.zip.002`
- `DIVA-Windows-x64-v1.0.2.zip.003`

Run the installer and choose a writable directory. It verifies the volumes and extracts the complete application. Do not run a volume directly or copy only the desktop EXE.
Python, .NET and required libraries are bundled; no separate server or CUDA Toolkit installation is needed.

**Requirements:** Windows 11 22H2 or later, x64, an NVIDIA GPU and a driver compatible with CUDA 12.6 for phoneme recognition. GAME note transcription runs on four CPU threads. Allow at least 16 GiB of free disk space for installation, weights and caches.

This build is unsigned. Read the release acceptance report for the actual tested scope. Application stability is separate from recognition-accuracy certification.

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

Install this version into a new folder. Select a previous installation in Setup to copy only models matching the new catalog hashes; old models and results stay untouched. Verified downloaded model blobs are cached across versions.

Settings, logs and results use `%LOCALAPPDATA%\Any2Xsampa\state`; verified model downloads use `%LOCALAPPDATA%\Any2Xsampa\cache\model-downloads`. `ANY2XSAMPA_DATA_HOME` selects an absolute portable data directory. Old preview results remain in their original installation and can be replayed directly.

Setup can uninstall unchanged program files while preserving models, shared data and unlisted or modified files. Close the app first. **Check program updates** is explicit and read-only: it opens the official release page and never silently replaces an application or model.

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
- **Correct file generation does not establish accurate transcription.** Formal joint pitch/onset/offset accuracy has not been validated against independent original-singing note ground truth. The recognizer retains research-quality limitations; a stable application version is not an accuracy certificate.

See [VSQX documentation](docs/vsqx_generation.md), [distribution instructions](docs/distribution/README-使用说明.md) and [legacy reference workflows](docs/legacy_product_workflows.md). Legacy research workflows are not required for automatic generation.

## Development verification

~~~text
python -m pytest tests/test_product_branding.py tests/test_model_install.py tests/test_game_vsqx.py tests/test_local_score.py tests/test_local_engine.py tests/test_local_whole_audio.py tests/test_local_desktop_layout.py -q -p no:cacheprovider
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
