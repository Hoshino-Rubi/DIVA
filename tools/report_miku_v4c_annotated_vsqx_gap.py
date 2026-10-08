from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from any2xsampa.cantonese import CantonesePhonemeResolver, load_lexicon_file  # noqa: E402


SCHEMA_VERSION = "miku_v4c_annotated_vsqx_resolver_gap_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare aligned user-authored VSQX phoneme slots with the current Cantonese resolver lattice. "
            "A mismatch means the lattice cannot reproduce the annotation exactly; it is not an automatic truth verdict."
        )
    )
    parser.add_argument("corpus", type=Path, help="miku_v4c_annotated_vsqx_corpus_v1 JSON")
    parser.add_argument("--lexicon", type=Path, help="Optional merged Jyutping lexicon JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    return parser


def run_report(
    *,
    corpus: Path,
    output: Path,
    lexicon: Path | None = None,
    pretty: bool = True,
) -> tuple[int, dict[str, Any]]:
    corpus_payload = _read_json(corpus)
    resolver = CantonesePhonemeResolver(lexicon=load_lexicon_file(lexicon) if lexicon else None)
    observations = _slot_observations(corpus_payload)
    rows = [_compare_slot(row, resolver=resolver) for row in observations]
    resolved = [row for row in rows if row["resolver_candidate_count"] > 0]
    exact = [row for row in rows if row["manual_exact_lattice_match"]]
    top = [row for row in rows if row["manual_top_candidate_match"]]
    unresolved = [row for row in rows if row["resolver_candidate_count"] == 0]
    misses = [row for row in rows if row["resolver_candidate_count"] > 0 and not row["manual_exact_lattice_match"]]
    by_character = _character_summary(rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "errors": [],
        "corpus": str(corpus),
        "lexicon": str(lexicon) if lexicon else None,
        "segment_count": len(corpus_payload.get("segments", [])),
        "slot_count": len(rows),
        "resolved_slot_count": len(resolved),
        "unresolved_slot_count": len(unresolved),
        "manual_exact_lattice_match_count": len(exact),
        "manual_top_candidate_match_count": len(top),
        "manual_sequence_not_in_lattice_count": len(misses),
        "manual_exact_lattice_match_ratio": round(len(exact) / len(rows), 6) if rows else 0.0,
        "manual_top_candidate_match_ratio": round(len(top) / len(rows), 6) if rows else 0.0,
        "unique_character_count": len(by_character),
        "characters": by_character,
        "rows": rows,
        "model_output": False,
        "training_data": False,
        "automatic_comparison_not_ground_truth": True,
        "notes": [
            "Repeated notes mapped to one lyric character are concatenated into one comparison slot.",
            "Token comparison is exact and preserves raw VSQX symbols; no alias normalization is applied.",
            "A lattice miss identifies a mapper/ranker coverage gap, not proof that the user-authored annotation is universally correct.",
        ],
    }
    _write_json(output, payload, pretty=pretty)
    return 0, payload


def _slot_observations(corpus: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    segments = corpus.get("segments")
    if not isinstance(segments, list):
        raise ValueError("corpus must contain segments[]")
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for note in segment.get("notes", []):
            if isinstance(note, Mapping):
                grouped[int(note.get("character_index") or 0)].append(note)
        for character_index in sorted(index for index in grouped if index > 0):
            notes = sorted(grouped[character_index], key=lambda row: int(row.get("note_index") or 0))
            manual_tokens = [
                str(token)
                for note in notes
                for token in note.get("phoneme_tokens", [])
                if str(token)
            ]
            result.append(
                {
                    "segment_id": segment.get("segment_id"),
                    "variant_group": segment.get("variant_group"),
                    "revision_order": segment.get("revision_order"),
                    "character_index": character_index,
                    "character": str(notes[0].get("lyric_character") or ""),
                    "note_indexes": [note.get("note_index") for note in notes],
                    "note_count": len(notes),
                    "pitches": [note.get("pitch") for note in notes],
                    "manual_note_sequences": [note.get("phoneme_text") for note in notes],
                    "manual_phoneme_tokens": manual_tokens,
                    "manual_phoneme_sequence": " ".join(manual_tokens),
                    "source_vsqx": segment.get("source_vsqx"),
                }
            )
    return result


def _compare_slot(row: Mapping[str, Any], *, resolver: CantonesePhonemeResolver) -> dict[str, Any]:
    char = str(row.get("character") or "")
    syllable = resolver.resolve_syllable(char).to_dict()
    raw_candidates = syllable.get("candidates")
    candidates = [candidate for candidate in raw_candidates if isinstance(candidate, Mapping)] if isinstance(raw_candidates, list) else []
    candidate_rows = [
        {
            "jyutping": candidate.get("jyutping"),
            "source": candidate.get("source"),
            "score": candidate.get("score"),
            "v4c_phonemes": [str(token) for token in candidate.get("v4c_phonemes", [])],
            "v4c_phoneme_sequence": " ".join(str(token) for token in candidate.get("v4c_phonemes", [])),
        }
        for candidate in candidates
    ]
    manual_tokens = [str(token) for token in row.get("manual_phoneme_tokens", [])]
    exact_indexes = [
        index
        for index, candidate in enumerate(candidate_rows)
        if candidate["v4c_phonemes"] == manual_tokens
    ]
    return {
        **dict(row),
        "resolver_candidate_count": len(candidate_rows),
        "resolver_candidates": candidate_rows,
        "resolver_top_jyutping": candidate_rows[0]["jyutping"] if candidate_rows else None,
        "resolver_top_phoneme_sequence": candidate_rows[0]["v4c_phoneme_sequence"] if candidate_rows else None,
        "manual_exact_lattice_match": bool(exact_indexes),
        "manual_exact_lattice_match_indexes": exact_indexes,
        "manual_top_candidate_match": bool(exact_indexes and exact_indexes[0] == 0),
        "model_output": False,
        "training_data": False,
    }


def _character_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("character") or "")].append(row)
    result: list[dict[str, Any]] = []
    for char in sorted(grouped):
        observations = grouped[char]
        manual_counts = Counter(str(row.get("manual_phoneme_sequence") or "") for row in observations)
        top_counts = Counter(str(row.get("resolver_top_phoneme_sequence") or "") for row in observations)
        result.append(
            {
                "character": char,
                "observation_count": len(observations),
                "resolved_observation_count": sum(1 for row in observations if row.get("resolver_candidate_count", 0)),
                "manual_exact_lattice_match_count": sum(1 for row in observations if row.get("manual_exact_lattice_match")),
                "manual_top_candidate_match_count": sum(1 for row in observations if row.get("manual_top_candidate_match")),
                "manual_phoneme_sequence_counts": dict(
                    sorted(manual_counts.items(), key=lambda item: (-item[1], item[0]))
                ),
                "resolver_top_phoneme_sequence_counts": dict(
                    sorted(top_counts.items(), key=lambda item: (-item[1], item[0]))
                ),
            }
        )
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any], *, pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rc, payload = run_report(
            corpus=args.corpus,
            lexicon=args.lexicon,
            output=args.output,
            pretty=args.pretty,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
