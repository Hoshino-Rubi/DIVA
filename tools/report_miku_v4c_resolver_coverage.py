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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from any2xsampa.cantonese import CantonesePhonemeResolver, load_lexicon_file  # noqa: E402
from tools import build_miku_v4c_phrase_batch as phrase_batch_tool  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "tmp_validation" / "miku_v4c_goal" / "resolver_coverage" / "resolver_coverage_report.json"
SCHEMA_VERSION = "miku_v4c_resolver_coverage_report_v1"
PATCH_SCHEMA_VERSION = "cantonese_jyutping_lexicon_patch_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report Cantonese resolver coverage for Miku V4C phrase inputs and emit an auditable "
            "Jyutping lexicon patch template. This does not change the default lexicon."
        )
    )
    parser.add_argument("inputs", nargs="*", type=Path, help="Phrase batch JSON, transcript/lyric files, or directories")
    parser.add_argument("--phrase", action="append", help="Add one phrase directly; can be repeated")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--patch-output", type=Path, help="Write the generated empty Jyutping patch template separately")
    parser.add_argument("--lexicon", type=Path, help="Optional merged Jyutping lexicon JSON for coverage what-if runs")
    parser.add_argument("--include-audio-titles", action="store_true", help="Use CJK audio filenames as low-confidence phrase fallbacks")
    parser.add_argument("--min-chars", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pretty", action="store_true")
    return parser


def run_report(
    *,
    inputs: Sequence[Path],
    phrase_args: Sequence[str] | None = None,
    output: Path = DEFAULT_OUTPUT,
    patch_output: Path | None = None,
    lexicon: Path | None = None,
    include_audio_titles: bool = False,
    min_chars: int = 2,
    max_chars: int = 8,
    limit: int | None = None,
    pretty: bool = True,
) -> tuple[int, dict[str, Any]]:
    if min_chars < 1:
        raise ValueError("--min-chars must be >= 1")
    if max_chars < min_chars:
        raise ValueError("--max-chars must be >= --min-chars")

    resolver = CantonesePhonemeResolver(lexicon=load_lexicon_file(lexicon) if lexicon else None)
    rows, load_errors = _load_phrase_rows(
        inputs,
        phrase_args=phrase_args,
        include_audio_titles=include_audio_titles,
        min_chars=min_chars,
        max_chars=max_chars,
        limit=limit,
    )

    phrase_rows: list[dict[str, Any]] = []
    unresolved_counter: Counter[str] = Counter()
    char_phrase_examples: dict[str, list[str]] = defaultdict(list)
    seen_phrases: set[str] = set()
    duplicate_count = 0

    for index, row in enumerate(rows):
        lyrics = str(row.get("lyrics") or row.get("text") or "").strip()
        if not lyrics:
            continue
        if lyrics in seen_phrases:
            duplicate_count += 1
            continue
        seen_phrases.add(lyrics)
        coverage = _phrase_coverage(lyrics, resolver)
        unresolved = list(coverage["resolver_unresolved_chars"])
        for char in unresolved:
            unresolved_counter[char] += 1
            examples = char_phrase_examples[char]
            if lyrics not in examples and len(examples) < 5:
                examples.append(lyrics)
        phrase_rows.append(
            {
                "phrase_id": str(row.get("phrase_id") or f"phrase-{len(phrase_rows) + 1:05d}"),
                "lyrics": lyrics,
                "source": row.get("source") or "direct_phrase",
                "source_path": row.get("source_path"),
                "source_index": row.get("source_index"),
                "chunk_index": row.get("chunk_index"),
                "resolver_syllable_count": coverage["resolver_syllable_count"],
                "resolver_covered_syllable_count": coverage["resolver_covered_syllable_count"],
                "resolver_coverage": coverage["resolver_coverage"],
                "resolver_unresolved_chars": unresolved,
                "fully_covered": coverage["resolver_coverage"] == 1.0,
                "model_output": False,
                "training_data": False,
            }
        )
        if limit is not None and len(phrase_rows) >= limit:
            break

    full_count = sum(1 for row in phrase_rows if row["fully_covered"])
    zero_count = sum(1 for row in phrase_rows if row["resolver_coverage"] == 0.0)
    partial_count = len(phrase_rows) - full_count - zero_count
    unresolved_chars = _unresolved_rows(unresolved_counter, char_phrase_examples)
    patch_template = _patch_template(unresolved_chars)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "ok": not load_errors,
        "errors": load_errors,
        "inputs": [str(path) for path in inputs],
        "include_audio_titles": include_audio_titles,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "phrase_count": len(phrase_rows),
        "duplicate_phrase_count": duplicate_count,
        "full_coverage_phrase_count": full_count,
        "partial_coverage_phrase_count": partial_count,
        "zero_coverage_phrase_count": zero_count,
        "coverage_ratio": round((full_count / len(phrase_rows)) if phrase_rows else 0.0, 6),
        "unresolved_char_count": len(unresolved_chars),
        "unresolved_chars": unresolved_chars,
        "phrases": phrase_rows,
        "lexicon_patch_template": patch_template,
        "patch_output": str(patch_output) if patch_output else None,
        "lexicon": str(lexicon) if lexicon else None,
        "model_output": False,
        "training_data": False,
        "notes": [
            "Coverage is resolver readiness only; it is not phoneme truth and not training data.",
            "Fill lexicon_patch_template.entries with reviewed Jyutping before applying it to the default lexicon.",
            "Low-confidence audio-title fallbacks should be replaced by lyric/Qwen transcript snippets when available.",
        ],
    }
    _write_json(output, payload, pretty=pretty)
    if patch_output is not None:
        _write_json(patch_output, patch_template, pretty=pretty)
    return (0 if not load_errors else 1), payload


def _load_phrase_rows(
    inputs: Sequence[Path],
    *,
    phrase_args: Sequence[str] | None,
    include_audio_titles: bool,
    min_chars: int,
    max_chars: int,
    limit: int | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, phrase in enumerate(phrase_args or []):
        text = str(phrase).strip()
        if text:
            rows.append(
                {
                    "lyrics": text,
                    "source": "direct_phrase",
                    "source_index": index,
                    "chunk_index": 0,
                }
            )
    if inputs:
        files = list(phrase_batch_tool._iter_input_files(inputs, include_audio_titles=include_audio_titles))  # noqa: SLF001
        if not files:
            errors.append("no supported phrase source files found")
        for path in files:
            try:
                rows.extend(
                    phrase_batch_tool._rows_from_file(  # noqa: SLF001
                        path,
                        min_chars=min_chars,
                        max_chars=max_chars,
                        include_audio_titles=include_audio_titles,
                    )
                )
            except Exception as exc:  # keep report useful when one source is malformed
                errors.append(f"{path}: {exc}")
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        errors.append("no phrase rows found")
    return rows, errors


def _phrase_coverage(lyrics: str, resolver: CantonesePhonemeResolver) -> dict[str, Any]:
    lattice = resolver.resolve_text(lyrics)
    syllables = lattice.get("syllables") if isinstance(lattice, Mapping) else []
    if not isinstance(syllables, list):
        syllables = []
    total = len(syllables) or phrase_batch_tool._cjk_len(lyrics)  # noqa: SLF001
    covered = 0
    unresolved: list[str] = []
    for syllable in syllables:
        if not isinstance(syllable, Mapping):
            continue
        text = str(syllable.get("text") or syllable.get("char") or "")
        candidates = syllable.get("candidates")
        if isinstance(candidates, list) and candidates:
            covered += 1
        elif text:
            unresolved.append(text)
    coverage = round((covered / total) if total else 0.0, 6)
    return {
        "resolver_syllable_count": total,
        "resolver_covered_syllable_count": covered,
        "resolver_coverage": coverage,
        "resolver_unresolved_chars": unresolved,
    }


def _unresolved_rows(counter: Counter[str], examples: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    rows = []
    for char, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        rows.append(
            {
                "char": char,
                "unresolved_occurrences": count,
                "example_phrases": list(examples.get(char, [])),
                "suggested_jyutping": [],
            }
        )
    return rows


def _patch_template(unresolved_chars: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": PATCH_SCHEMA_VERSION,
        "entries": {str(row["char"]): [] for row in unresolved_chars},
        "model_output": False,
        "training_data": False,
        "notes": [
            "Empty lists are placeholders. Fill reviewed Jyutping syllables before applying this patch.",
            "Keep multiple readings as a ranked list, most likely first.",
        ],
    }


def _write_json(path: Path, payload: Mapping[str, Any], *, pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rc, payload = run_report(
            inputs=args.inputs,
            phrase_args=args.phrase,
            output=args.output,
            patch_output=args.patch_output,
            lexicon=args.lexicon,
            include_audio_titles=args.include_audio_titles,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            limit=args.limit,
            pretty=args.pretty,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
