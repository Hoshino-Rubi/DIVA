# Miku V4C Phoneme Findings

Date: 2026-06-27

## Scope

These findings track the Miku V4C phoneme-token behavior needed for the first
VSQX import/playback target. They are singer-specific and should not be assumed
to apply to other VOCALOID singers.

The project truth layer remains phones / IPA / articulatory evidence. Miku V4C
phoneme strings are export-layer tokens written into VSQX note `p` fields.

## PHG2 token truth

PHG2 is the current token-vocabulary evidence source for Miku V4C phoneme
strings. For this project, PHG2 token truth means:

- use Miku V4C PHG2-style tokens for the VSQX `p` field;
- keep those tokens separate from X-SAMPA and recognizer phones;
- validate token sequences in an actual Miku V4C VSQX note context before
  treating them as playable defaults.

PHG2 membership alone is not enough. A token can be listed or accepted by the
editor and still be silent in playback for a particular note context.

## Current playable phrase

The current Phase 1 playable phrase target is:

```text
喜歡你 = x ei / f u n / n ei
```

| Lyric | Working pronunciation target | Miku V4C note `p` tokens |
| --- | --- | --- |
| `喜` | `hei` | `x ei` |
| `歡` | `fun` | `f u n` |
| `你` | `nei` | `n ei` |

Mapping implication: keep Cantonese `fun` as `f u n` by default. Do not replace
it with `f UN` for Miku V4C Phase 1.

## Known silent or unsafe tokens

Known silent token shorthand: i\ / i` / bare @N / f UN.

| Token pattern | Current observation | Mapping action |
| --- | --- | --- |
| <code>i\</code> | Silent in the observed `*_ibs` note probes. | Do not use as a default until phrase-context playback evidence changes. |
| <code>i&#96;</code> | Silent in the observed `*_ibt` note probes. | Do not use as a default until phrase-context playback evidence changes. |
| bare <code>@N</code> | Silent as a standalone final in the observed probe. | Do not promote from isolated/bare-token evidence. Test only in phrase note context. |
| <code>f UN</code> | Silent in the narrow PHG2 probe. | Keep `fun` as `f u n`. |

Additional context:

- Onset-attached `@N` can behave differently from bare `@N`, so the note `p`
  context matters.
- Some onset+final combinations can sound even when a related bare final did not.
- The safe default should be based on short-phrase playback, not token presence alone.

## Probe methodology

Use short phrase probes, not isolated single-character probes.

Required probe discipline:

1. Use a template-cloned VOCALOID4 VSQX file as the probe vehicle.
2. Keep the singer as Miku V4C.
3. Keep pitch, duration, tempo, and note spacing stable while testing one phoneme-token change.
4. Test token sequences inside note `p` fields with realistic onset/final context.
5. Record import success separately from audible playback success.
6. Promote only audible phrase-context results into default mappings.

Reason: isolated single characters can hide singer-context behavior, note-boundary
behavior, and onset/final interactions. The product target is phrase playback,
so phrase probes are the useful acceptance evidence.

## Product implications

- Users should not have to provide Jyutping.
- Jyutping remains useful as an internal override or debugging handle, but it is
  not a user requirement for the Windows-first product path.
- Lyrics are optional input. User lyrics or Qwen3-ASR 1.7B lyric evidence can
  guide phrase selection, but neither should replace the phones / IPA evidence layer.
- Miku V4C is the first singer target; each future singer needs its own phoneme
  findings and playback validation.

## Open follow-up

- Expand validated short phrases beyond `喜歡你`.
- Keep a clear list of PHG2 tokens that are vocabulary-only versus phrase-playable.
- Add future probe results only when they include both import and playback observations.
