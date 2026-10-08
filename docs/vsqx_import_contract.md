# VSQX Import Contract for Miku V4C Phase 1

Date: 2026-06-27

## Scope

This contract defines the local VSQX import surface for the first Windows-first
product slice: Miku V4C in VOCALOID4 FE. It is an export/import contract only.
The project truth layer remains phones / IPA / articulatory evidence.

## Export mode hierarchy

1. **Template clone is the accepted path for import validation.**
   Clone a known-good VOCALOID4 VSQX template and replace only the project
   labels, tempo/time signature, part length, and note nodes needed by the
   candidate.
2. **Minimal generated VSQX is a developer smoke path only.**
   A minimal XML scaffold can be useful for unit tests, but XML well-formedness
   does not prove VOCALOID4 FE importability or Miku V4C playback.

## Template-clone requirements

The template-clone exporter should preserve the VOCALOID4 scaffold that the
editor already accepts:

- VSQ4 root namespace and schema metadata;
- `vender` and `version` fields;
- `masterTrack` structure;
- `mixer` and track unit structure;
- `vVoiceTable` structure from the template, with singer identity normalized to Miku V4C;
- `vsTrack` / `vsPart` structure;
- `sPlug`, `singer`, controller lanes, and `plane` nodes.

The exporter may replace:

- project labels/comments;
- tempo and time signature;
- `vsPart` play time;
- note nodes with candidate notes containing `t`, `dur`, `n`, `v`, `y`, `p`, and `nStyle`.
- singer identity fields for this phase:
  - internal `singer_target`: `miku_v4c`;
  - VSQX `vVoiceTable/vVoice/name`: `MIKU_V4_Chinese`;
  - VSQX `vVoiceTable/vVoice/id`: `BNGE7CP7EMTRSNC3`;
  - VSQX `singer/bs` and `singer/pc`: `0` / `0`.

## Draft input expectations

The source draft should remain a `vocal_track_draft_v1`-style payload with:

- `singer_target`: `miku_v4c` for this phase;
- one or more tracks;
- notes with stable start tick, duration, pitch, lyric, and phoneme-token fields;
- provenance showing that VOCALOID phoneme strings are export-layer tokens, not internal truth.

Lyrics are optional product input. A note lyric may come from user-provided
lyrics, ASR lyric evidence, or a generated fallback for validation, but the
phoneme field must still be derived from the phone/IPA evidence and Miku V4C
mapping rules.

## Phoneme-token contract

The VSQX note `p` field contains Miku V4C PHG2-style phoneme tokens separated
by spaces. These tokens are not X-SAMPA and must not be treated as recognizer
truth.

PHG2 token evidence is necessary but not sufficient:

- PHG2 membership is the token-vocabulary evidence for Miku V4C.
- The product acceptance truth is whether the token sequence imports and plays
  audibly in the Miku V4C VSQX note `p` context.
- A token that is present in PHG2 can still be unsafe or silent in a specific
  VSQX note context.

Current known silent or unsafe token patterns: i\ / i` / bare @N / f UN.

- <code>i\</code>
- <code>i&#96;</code>
- bare <code>@N</code>
- <code>f UN</code>

Current accepted playable phrase mapping:

```text
喜歡你 = x ei / f u n / n ei
```

The accepted Phase 1 phrase must use:

- `喜` / `hei`: `x ei`
- `歡` / `fun`: `f u n`
- `你` / `nei`: `n ei`

Do not replace `f u n` with `f UN` for the current Miku V4C default.

## Import/playback acceptance

A candidate is accepted only when all of the following are true:

1. The file is exported by template clone.
2. VOCALOID4 FE opens/imports it without an invalid-file error.
3. The expected notes appear in the editor.
4. Lyrics and note phoneme fields are visible on the notes.
5. The singer resolves to Miku V4C (`MIKU_V4_Chinese`) or can be switched to Miku V4C without changing the phoneme sequence.
6. The phrase plays audibly through all notes.
7. The playback result is recorded separately from XML parse or import success.

Failure classification:

- XML parse success means only that the XML is well formed.
- VOCALOID4 FE import success means only that the editor accepted the file.
- Editor-visible phoneme tokens mean only that the tokens were serialized.
- Phase 1 success requires audible playback for the accepted phrase.

## Probe rules

- Probe short natural phrases, not isolated single characters.
- Keep phrase length small enough to identify which token sequence caused a failure.
- Change one variable at a time: template, singer, note timing, lyric, or phoneme sequence.
- Keep template clone as the default probe vehicle.
- Record import behavior and playback behavior as separate observations.
- Do not promote a token sequence into default mapping from isolated-character evidence alone.
