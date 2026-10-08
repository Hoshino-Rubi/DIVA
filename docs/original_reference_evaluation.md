# Original Reference Evaluation

The user's approved VSQX phonemes were manually calibrated one by one while
listening against the original in vocaloid4fe. Treat this as human original
reference evidence. It is not necessary to repeat the complete annotation.

## Separate Evidence Layers

- Human annotation: original-reviewed engine phonemes, with the exact user
  statement and source revision hashes.
- Representation: evidence-backed IPA derived from those engine phonemes.
  This is a declared project reference convention, not a new independent
  acoustic IPA transcription.
- Timing: the specific original crop and its complete reference sequence.
  Machine timing proposals do not automatically become human confirmations.

Sequence PER requires the correct crop sequence, not millisecond boundaries
for every phone. Phone-duration and product-alignment gates are unchanged.

## Implementation

`tools/evaluate_original_reference.py` rebuilds the immutable base reference,
binds review evidence, proposes bounded whole-song CENS/DTW correspondence,
and replays the original PhoneticXeus logits through real and fixture adapters.

`tools/probe_original_reference_offset.py` additionally evaluates an audio-only
constant-offset hypothesis. Fitting excludes all three fixed original clips
per train/development song. Two fixed resolutions are reported; neither is
chosen using recognition scores. The existing VSQX premeasure metadata is
reported afterward and does not fit or validate the audio offset by itself.

Both tools preserve the eight-song split. They process only the six train/dev
groups. Final-test source hashes may be checked, but prediction content is
never parsed. No model is trained, downloaded, or replaced by these tools.

## Mapping Supplement

The opt-in `--mapping-evidence` argument enables exactly one documented
compound restoration: `ua_n = ua + n`, producing `[u, U+0250, n]` under the
existing project convention. The v2 mapping explicitly records this compound;
its broken IPA is not inverted. Intact v4/v1-derived constituents supply IPA.

The installed editor manual, printed pages 168-169, corroborates the symbol
structure. These pages were rendered and inspected because plain PDF text
extraction misaligned table rows. The manual provides pronunciation examples,
not a Cantonese IPA table. Other unknown aliases remain unresolved.

The supplement never overwrites v4, raw VSQX, the old rendered dataset, or test
note projections. Its source files and hashes are part of the new manifest.

## Actual Run

Run root: `.state/runs/original-reference-eval-20260906T154652/`.

- `offset-final/`: current offset diagnostic and supplemented reference.
- `evaluation-v1/`: earlier global-DTW diagnostic; correspondence did not pass.
- `offset-probe-v1/`: retained failed attempt, before out-of-range proposals
  were explicitly excluded instead of aborting the complete job.
- `offset-probe-v2/`: successful offset diagnostic before the mapping supplement.
- `manual-evidence/`: source bindings and two visually inspected table pages.
- `verification-final.json`: independent replay, integrity and media checks.
- `commands/`: immutable commands, source fingerprints, exits and logs.

The current Shalong offset proposals are +6.7802 and +6.8731 seconds. Its four
premeasures at 142 BPM correspond to 6.7606 seconds, a corroborating observation
only. Fit margins are small and some other songs give implausible offsets;
resolution agreement alone is not an acceptance criterion.

Shalong has 310/320 mapped notes after the supplement. All three fixed Shalong
windows now have complete proposed reference sequences, totaling 236 phones.
Their conditional PER values are 0.803279, 0.734940 and 0.706522 for both
resolutions. These are explicitly not official original-accuracy results.

## Review And Replay

`offset-final/FOCUSED_REVIEW.md` links three short original/rendered audio
pairs. Only their specific crop/sequence correspondence is pending, not the
source phoneme approval. Six derived WAVs total 108 seconds; full source WAVs
were not copied. Audio format, hashes and waveform images were checked; the
agent does not claim to have listened or verified browser playback.

`clip-binding-review-template.json` starts with pending responses. Do not
automatically flip them to confirmed. An explicit confirmed record must bind
the source hashes, exact frozen original interval, contiguous note IDs, review
evidence and reviewer statement. The importer rejects partial/unknown targets,
source mismatches, shifted times and final-test items.

After genuine clip confirmation, use `tools/evaluate_original_reference.py`
with `--clip-bindings` and the same `--mapping-evidence`, writing a new output
directory. All inputs are the paths already recorded in `offset-final/inputs.json`.
Do not overwrite the diagnostic reports or retroactively call them official.

### Shalong order-only confirmation

The user subsequently confirmed the full-song Shalong phoneme/note order and
the existence of a leading offset. This does not confirm the exact boundaries
of the three fixed machine-selected crops. The corrected canonical run is
`.state/runs/shalong-order-confirmed-20260906T173051/offset-diagnostic/`;
its official window count is zero. Existing adapter transfer values there are
conditional offset diagnostics only. The earlier run that temporarily marked
the clips as official is retained with a `SUPERSEDED.md` notice.

Closing regression: 629 passed, 2 skipped (the two installed-grpc guards).
The first focused run using the shared OS pytest temp root encountered a
cleanup permission error after assertions passed. Run-local `--basetemp`
avoided that unrelated path; no user temp directory was deleted.
