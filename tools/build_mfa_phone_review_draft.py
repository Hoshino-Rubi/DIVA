"""Build a full-song, hash-bound MFA draft for explicit human phone review."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from any2xsampa.adapters.forced_aligner import ForcedPhoneAlignerError
from any2xsampa.adapters.mfa_forced_aligner import (
    build_mfa_full_song_review_draft,
    build_mfa_timeline_constrained_review_draft,
)
from any2xsampa.lyrics.timeline import (
    LyricsTimelineError,
    load_lyrics_timeline,
    sha256_file,
)


class ReviewDraftBuildError(ValueError):
    """The full-song MFA artifact cannot become a human-review draft."""


def _read_alignment(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8-sig"))
    except FileNotFoundError as exc:
        raise ReviewDraftBuildError(f"missing MFA alignment: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewDraftBuildError(f"invalid MFA alignment JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ReviewDraftBuildError("MFA alignment JSON must be an object")
    return dict(payload), hashlib.sha256(raw).hexdigest()


def write_review_draft(path: Path | str, payload: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(str(output), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ReviewDraftBuildError(f"refusing to overwrite existing output: {output}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
    timing = payload.get("provenance", {}).get("timing_comparison", {})
    return {
        "path": str(output),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "segment_count": len(payload.get("segments", [])),
        "timing_within_policy": bool(timing.get("within_policy")),
        "review_status": payload.get("review", {}).get("status"),
        "product_accepted": payload.get("provenance", {}).get("product_accepted"),
    }


def build_review_draft(
    *,
    alignment_path: Path | str,
    vocal_stem_path: Path | str,
    timeline_path: Path | str,
    model_path: Path | str,
    model_revision: str,
    output_path: Path | str,
    dictionary_path: Path | str | None = None,
    timeline_constrained: bool = False,
) -> dict[str, Any]:
    alignment = Path(alignment_path).expanduser().resolve()
    vocal_stem = Path(vocal_stem_path).expanduser().resolve()
    timeline_file = Path(timeline_path).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    dictionary = (
        Path(dictionary_path).expanduser().resolve()
        if dictionary_path is not None
        else None
    )
    if not model.is_file():
        raise ReviewDraftBuildError(f"missing MFA model: {model}")
    payload, alignment_hash = _read_alignment(alignment)
    try:
        timeline = load_lyrics_timeline(timeline_file)
        build_draft = (
            build_mfa_timeline_constrained_review_draft
            if timeline_constrained
            else build_mfa_full_song_review_draft
        )
        draft = build_draft(
            payload,
            vocal_stem=vocal_stem,
            timeline=timeline,
            model_hash=sha256_file(model),
            model_revision=model_revision,
            dictionary_path=dictionary,
            alignment_sha256=alignment_hash,
        )
    except (ForcedPhoneAlignerError, LyricsTimelineError, OSError) as exc:
        raise ReviewDraftBuildError(str(exc)) from exc
    return write_review_draft(output_path, draft)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a complete MFA JSON alignment into a machine-only phone draft "
            "that must still receive human short-clip and full-song acceptance."
        )
    )
    parser.add_argument("--alignment", required=True, type=Path)
    parser.add_argument("--vocal-stem", required=True, type=Path)
    parser.add_argument("--lyrics-timeline", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dictionary", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--timeline-constrained",
        action="store_true",
        help=(
            "opt in to a machine-only draft whose MFA phone intervals are "
            "mapped into the supplied lyric-item bounds"
        ),
    )
    args = parser.parse_args(argv)
    try:
        report = build_review_draft(
            alignment_path=args.alignment,
            vocal_stem_path=args.vocal_stem,
            timeline_path=args.lyrics_timeline,
            model_path=args.model,
            model_revision=args.model_revision,
            dictionary_path=args.dictionary,
            output_path=args.output,
            timeline_constrained=args.timeline_constrained,
        )
    except ReviewDraftBuildError as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(report, ensure_ascii=False))
    return 0


__all__ = ["ReviewDraftBuildError", "build_review_draft", "main", "write_review_draft"]


if __name__ == "__main__":
    raise SystemExit(main())
