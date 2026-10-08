# Shalong Clock Rule

This independent audit is not a new recognizer or a shortcut around reference
acceptance. Current PER remains conditional until exact crop sequences have
adequate evidence. See `HANDOFF_SHALONG_CLOCK_RULE.md` for results.

## Clock Domains

`reference_clock.py` supplies named exact-rational unit-rate transforms and
explicit floor/ceil sample conversion. Only the verified VSQ4 constant-tempo,
single-signature case is supported; other clocks fail rather than silently
adopting these equations.

- `score_absolute`: tempo-integrated `part_tick + note_tick`.
- `score_display`: absolute time minus premeasure duration.
- `rendered`: the exported WAV sample origin.
- `original`: the recognizer input WAV sample origin.

The waveform origin is not inferred from XML premeasures alone.
`reference_clock_audio.py` separately checks original/rendered descriptor
correlation and rendered/score note-rest activity. Recognizer logits, phones
and error scores never enter fitting. Score activity is only post-hoc timing
evidence, not acoustic decoding context.

## Usage

```powershell
.venv\Scripts\python.exe -B -X utf8 tools/audit_shalong_clock.py `
  --reference .state/runs/limited-data-20260906T050251/reference/reference-manifest.json `
  --human-review .state/runs/reference-review-clarification-20260906T154045/user-reference-evidence.json `
  --order-evidence .state/runs/shalong-order-confirmed-20260906T173051/shalong-order-confirmation.json `
  --mapping-evidence .state/runs/original-reference-eval-20260906T154652/manual-evidence/source.json `
  --descriptor-report .state/runs/limited-data-20260906T050251/correspondence-v2/shalong.json `
  --output-dir .state/runs/NEW_RUN/clock-audit
```

Use a new output directory. Wrap production commands in
`tools/run_limited_data_command.py` for four-thread limits and immutable logs.
No installation, network access or GPU model execution is needed.

`tools/build_clock_boundary_review.py` takes the output `--clock-rule`, the
same `--reference` and `--mapping-evidence`, and a new `--output-dir`. It
creates six pending cut-point items and short derived audio previews only.

## Boundaries

- Order approval does not confirm a numeric offset or machine cuts.
- Lag candidates use common frame support, not differing audio durations.
- Offset envelopes are observed sensitivity, not probabilistic intervals.
- Complete note containment must hold throughout the supported envelope.
- Partial/unknown notes remain in evidence; they cannot be dropped for a
  better score. Note activity does not establish phone durations.
- Rest proposals only expand old crops and do not select by PER.
- Failed gates are reported, not tuned until they pass.
- The official evaluator's explicit clip-binding requirement is unchanged.
  This audit never manufactures a `human_confirmed` item.
