from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from any2xsampa.cantonese import CantonesePhonemeResolver  # noqa: E402
from any2xsampa.vocaloid.draft import (  # noqa: E402
    DEFAULT_NOTE_DURATION_TICKS,
    DEFAULT_SINGER_TARGET,
    DEFAULT_TICKS_PER_QUARTER,
    VOCAL_TRACK_DRAFT_SCHEMA_VERSION,
)


DEFAULT_SMOKE_PITCHES = [64, 67, 69, 71, 72, 71, 69, 67]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a vocal_track_draft_v1 payload from Cantonese lyrics. "
            "Users are not expected to provide Jyutping; --jyutping-override is a developer/manual correction aid."
        )
    )
    parser.add_argument("lyrics", help="Chinese lyrics text. Whitespace is treated as phrase separation by the resolver.")
    parser.add_argument("--output", type=Path, required=True, help="Output vocal_track_draft_v1 JSON path")
    parser.add_argument("--jyutping-override", help="Optional internal/developer Jyutping override; not required for users")
    parser.add_argument("--tempo", type=float, default=128.0)
    parser.add_argument("--ticks-per-quarter", type=int, default=DEFAULT_TICKS_PER_QUARTER)
    parser.add_argument("--duration-ticks", type=int, default=DEFAULT_NOTE_DURATION_TICKS)
    parser.add_argument("--start-tick", type=int, default=0)
    parser.add_argument(
        "--pitches",
        help="Comma-separated MIDI pitches. If omitted, a simple smoke-test contour is repeated.",
    )
    parser.add_argument("--track-name", default="Miku V4C Cantonese smoke")
    parser.add_argument("--singer-target", default=DEFAULT_SINGER_TARGET)
    parser.add_argument("--pretty", action="store_true")
    return parser


def build_draft_payload(
    lyrics: str,
    *,
    jyutping_override: str | None = None,
    tempo: float = 128.0,
    ticks_per_quarter: int = DEFAULT_TICKS_PER_QUARTER,
    duration_ticks: int = DEFAULT_NOTE_DURATION_TICKS,
    start_tick: int = 0,
    pitches: list[int] | None = None,
    track_name: str = "Miku V4C Cantonese smoke",
    singer_target: str = DEFAULT_SINGER_TARGET,
    resolver: Any | None = None,
) -> dict[str, Any]:
    resolver = resolver or CantonesePhonemeResolver()
    lattice = resolver.resolve_text(lyrics, jyutping_override=jyutping_override)
    syllables = lattice.get("syllables", [])
    if not isinstance(syllables, list) or not syllables:
        raise ValueError("lyrics did not resolve to any Cantonese syllables")
    pitch_values = pitches or DEFAULT_SMOKE_PITCHES
    notes: list[dict[str, Any]] = []
    for index, syllable in enumerate(syllables):
        if not isinstance(syllable, dict):
            raise ValueError(f"syllables[{index}] must be an object")
        candidates = syllable.get("candidates")
        if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
            raise ValueError(f"syllables[{index}] has no usable phoneme candidate")
        candidate = candidates[0]
        notes.append(
            {
                "id": f"note-{index + 1}",
                "start_tick": start_tick + (index * duration_ticks),
                "duration_ticks": duration_ticks,
                "pitch": pitch_values[index % len(pitch_values)],
                "lyric": str(syllable.get("text", "la")),
                "phonemes": list(candidate.get("v4c_phonemes", [])),
                "phoneme_source": "cantonese_phoneme_lattice_v1",
                "provenance": {
                    "jyutping": candidate.get("jyutping"),
                    "internal_phones": candidate.get("internal_phones"),
                    "ipa": candidate.get("ipa"),
                    "source": candidate.get("source"),
                    "notes": candidate.get("notes", []),
                    "user_jyutping_required": False,
                },
            }
        )
    return {
        "schema_version": VOCAL_TRACK_DRAFT_SCHEMA_VERSION,
        "tempo": tempo,
        "time_signature": {"numerator": 4, "denominator": 4},
        "ticks_per_quarter": ticks_per_quarter,
        "singer_target": singer_target,
        "provenance": {
            "created_by": "tools/build_vocal_track_draft.py",
            "source_text": lyrics,
            "jyutping_override": jyutping_override,
            "user_jyutping_required": False,
            "truth_layer": "phones_ipa_articulatory_features",
            "xsampa_role": "export_only_not_internal_truth",
            "import_validation": "prepared_not_claimed",
        },
        "tracks": [
            {
                "id": "track-1",
                "name": track_name,
                "singer_target": singer_target,
                "notes": notes,
            }
        ],
    }


def _parse_pitches(value: str | None) -> list[int] | None:
    if value is None:
        return None
    pitches: list[int] = []
    for part in value.split(","):
        text = part.strip()
        if text:
            pitches.append(int(text))
    if not pitches:
        raise ValueError("--pitches must contain at least one MIDI pitch")
    return pitches


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = build_draft_payload(
            args.lyrics,
            jyutping_override=args.jyutping_override,
            tempo=args.tempo,
            ticks_per_quarter=args.ticks_per_quarter,
            duration_ticks=args.duration_ticks,
            start_tick=args.start_tick,
            pitches=_parse_pitches(args.pitches),
            track_name=args.track_name,
            singer_target=args.singer_target,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema_version": VOCAL_TRACK_DRAFT_SCHEMA_VERSION,
                "output": str(args.output),
                "lyrics": args.lyrics,
                "note_count": len(payload["tracks"][0]["notes"]),
                "user_jyutping_required": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
