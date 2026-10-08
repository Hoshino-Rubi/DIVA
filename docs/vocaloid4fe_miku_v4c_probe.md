# VOCALOID4 FE + Miku V4C/V4X local probe and VSQX import validation prep

This note records the local, reproducible install probe for VOCALOID4 FE and
Hatsune Miku V4C/V4X evidence.  It is deliberately a *metadata-only* probe:
it is meant to answer "is the import target present, and where can a human
operator validate a VSQX import?" without collecting license or activation
material.

## Scope and safety boundary

Allowed evidence:

- executable/install paths;
- display/version-like product metadata;
- VSQX sample/template candidate paths;
- voicebank-related public file names such as `.ddb`, `.ddi`, and `.vvd`;
- generated probe reports under `tmp_validation/vocaloid_probe/`.

Explicitly not collected:

- license keys, serial numbers, activation state, tokens, passwords, accounts,
  or credentials;
- file contents from voicebanks, VSQX files, registry blobs, or personal files;
- unrelated user directories.

The probe uses an allow-list for registry value names and skips sensitive-looking
registry value names/key paths such as `SerialNumber`, `LicenseKey`,
`ActivationCode`, `License`, `Credential`, and similar variants.

## Reproducible probe command

From the repo root:

```powershell
python tools\probe_vocaloid_install.py --output tmp_validation\vocaloid_probe\vocaloid_install_probe.json
python tools\probe_vocaloid_install.py --format text --output tmp_validation\vocaloid_probe\vocaloid_install_probe.txt
```

Useful constrained variants:

```powershell
# Avoid registry reads and scan only an explicitly authorized directory.
python tools\probe_vocaloid_install.py `
  --no-registry `
  --no-default-roots `
  --extra-root "C:\Program Files (x86)\VOCALOID4FE" `
  --output tmp_validation\vocaloid_probe\vocaloid_install_probe_explicit_root.json

# Human-readable stdout summary.
python tools\probe_vocaloid_install.py --format text
```

## Local evidence from this workstation

Probe run:

```text
python tools\probe_vocaloid_install.py --output tmp_validation\vocaloid_probe\vocaloid_install_probe.json
python tools\probe_vocaloid_install.py --format text --output tmp_validation\vocaloid_probe\vocaloid_install_probe.txt
```

Observed status:

- `vocaloid4_fe_status`: `evidence-found`
- `miku_v4_status`: `evidence-found`
- `miku_v4c_status`: `evidence-found`
- `miku_v4x_status`: `evidence-found`
- `vsqx_samples`: `0` sample/template VSQX candidates found by the metadata
  probe.

Registry display/install metadata found:

- `Miku V4 Chinese Vocaloid4 Library`
  - install root: `C:\Program Files (x86)\VoiceDB\`
- `MikuV4x Vocaloid4 Library`
  - install root: `C:\Program Files (x86)\VoiceDB\`
- `Vocaloid 4.5.2 Free Edition Plus`
  - install root: `C:\Program Files (x86)\Vocaloid4FE\`
- `Vocaloid4 4.2.1 Free Edition`
  - install root: `C:\Program Files (x86)\Vocaloid4FE\`

Executable candidates found:

- `C:\Program Files (x86)\VOCALOID4FE\VOCALOID4.exe`
- `C:\Program Files (x86)\VOCALOID4FE\VoiceLoader.exe`

Voicebank directory evidence:

- `C:\Program Files (x86)\VoiceDB\MikuV4C`
- `C:\Program Files (x86)\VoiceDB\MikuV4x`

Representative public voicebank file-name evidence:

- `C:\Program Files (x86)\VoiceDB\MikuV4C\...\MIKU_V4_Chinese.ddb`
- `C:\Program Files (x86)\VoiceDB\MikuV4C\...\MIKU_V4_Chinese.ddi`
- `C:\Program Files (x86)\VoiceDB\MikuV4C\...\MIKU_V4_Chinese.vvd`
- `C:\Program Files (x86)\VoiceDB\MikuV4x\...\MIKU_V4X_Original_EVEC.ddb`
- `C:\Program Files (x86)\VoiceDB\MikuV4x\...\MIKU_V4X_Original_EVEC.ddi`
- `C:\Program Files (x86)\VoiceDB\MikuV4x\...\MIKU_V4X_Original_EVEC.vvd`

The full metadata-only JSON/text reports are under
`tmp_validation/vocaloid_probe/`.

## VSQX import validation preparation

The import validation should remain a manual/editor-side compatibility check.
It must not be treated as evidence that the ASR recognizer truth layer changed:
the project rule still stands that phones / IPA / articulatory evidence are the
intermediate truth layer, while X-SAMPA/VOCALOID phoneme strings are export
layers.

Recommended validation path:

1. Generate or select a minimal `.vsqx` candidate in
   `tmp_validation/vocaloid_probe/`.
2. Launch VOCALOID4 FE from:
   `C:\Program Files (x86)\VOCALOID4FE\VOCALOID4.exe`.
3. In VOCALOID4 FE, import/open the candidate `.vsqx`.
4. Select the Miku V4C singer/voicebank if the editor does not bind it
   automatically.
5. Verify only these import-surface properties:
   - file opens without parse/import error;
   - track/note count is present;
   - tempo and time signature are plausible;
   - lyrics and exported phoneme fields appear on the notes;
   - singer assignment can be switched to or resolved as Miku V4C;
   - playback/render validation is optional and should be recorded separately.
6. Save any human-observed result as a short note under
   `tmp_validation/vocaloid_probe/`, not as a change to ASR/VSQX exporter code.

Current generated candidate:

- `tmp_validation/vocaloid_probe/miku_v4c_cantonese_smoke_phoneme_lattice.json`
- `tmp_validation/vocaloid_probe/miku_v4c_cantonese_smoke_draft.json`
- `tmp_validation/vocaloid_probe/miku_v4c_cantonese_smoke.vsqx`
- `tmp_validation/vocaloid_probe/miku_v4c_cantonese_smoke_vsq4.vsqx`

The candidate uses the Cantonese/Jyutping override `hei2 fun1 nei5` for
`喜歡你`, maps the chosen candidates into Miku V4C phoneme tokens, and preserves
the phone/IPA evidence in the draft sidecar.  The first XML-well-formed
VSQX-like candidate was rejected by VOCALOID4 FE as `Invalid VSQ4 File`; that
legacy candidate is kept as
`tmp_validation/vocaloid_probe/miku_v4c_cantonese_smoke_invalid_legacy.vsqx` for
comparison only.  The current candidate has been regenerated with the official
VOCALOID4 VSQ4 short-tag structure (`vender/version/mixer/masterTrack/vsTrack/
vsPart/note` and note fields `t/dur/n/v/y/p/nStyle`).  XML parse smoke confirmed
a VSQ4 `vsq4` root, a `vsPart`, no legacy `musicalPart`, and 3 note nodes.
VOCALOID4 FE editor import remains the decisive validation step.

## Test coverage

The probe tests use fake directories and a fake registry abstraction.  They do
not require, assume, or inspect the user's real VOCALOID installation.

```powershell
python -m pytest tests/test_vocaloid_install_probe.py -q
```

Expected result in this run:

```text
4 passed
```
