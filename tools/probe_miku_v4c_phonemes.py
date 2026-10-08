from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from any2xsampa.vocaloid import (  # noqa: E402
    DEFAULT_SINGER_TARGET,
    VOCAL_TRACK_DRAFT_SCHEMA_VERSION,
    VocalTrackDraft,
    collect_miku_v4c_policy_issues,
    export_vsqx_from_template,
    parse_phg2_path,
)


PROBE_DRAFT_SOURCE = "miku_v4c_phg2_probe"
DEFAULT_TOKENS_PER_NOTE = 3
DEFAULT_MAX_NOTES = 48
KNOWN_SILENT_OR_UNSAFE_DEFAULT_TOKENS = {"i\\", "i`", "UN"}
KNOWN_SILENT_OR_UNSAFE_DEFAULT_SEQUENCES = {"bare @N", "f UN"}


def build_probe_draft_payload(
    phg2_payload: Mapping[str, Any],
    *,
    tokens_per_note: int = DEFAULT_TOKENS_PER_NOTE,
    max_notes: int = DEFAULT_MAX_NOTES,
    exclude_tokens: set[str] | None = None,
) -> dict[str, Any]:
    excluded = exclude_tokens or set()
    tokens = [token for token in _string_list(phg2_payload.get("tokens")) if token not in excluded]
    if not tokens:
        raise ValueError("PHG2 payload has no tokens to probe")
    groups = _group_lookup(phg2_payload.get("groups"))
    chunks = _chunk_tokens(tokens, tokens_per_note=max(2, tokens_per_note), max_notes=max(1, max_notes))
    notes: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        label = _short_probe_label(chunk, groups, index=index)
        notes.append(
            {
                "id": f"phg2-probe-{index + 1:02d}",
                "start_tick": index * 480,
                "duration_ticks": 480,
                "pitch": 60 + (index % 5),
                "lyric": label,
                "phonemes": chunk,
                "phoneme_source": PROBE_DRAFT_SOURCE,
                "provenance": {
                    "truth_layer": "miku_v4c_phg2_export_token_probe",
                    "xsampa_role": "not_used",
                },
                "metadata": {
                    "probe_label": label,
                    "probe_kind": "short_phrase_or_label",
                    "token_count": len(chunk),
                    "token_groups": sorted({groups.get(token, "ungrouped") for token in chunk}),
                },
            }
        )

    return {
        "schema_version": VOCAL_TRACK_DRAFT_SCHEMA_VERSION,
        "tempo": 112.0,
        "time_signature": {"numerator": 4, "denominator": 4},
        "ticks_per_quarter": 480,
        "singer_target": DEFAULT_SINGER_TARGET,
        "tracks": [
            {
                "id": "miku-v4c-phg2-probe",
                "name": "Miku V4C PHG2 token probe",
                "singer_target": DEFAULT_SINGER_TARGET,
                "notes": notes,
                "provenance": {
                    "source": PROBE_DRAFT_SOURCE,
                    "note_design": "short multi-token probe labels, not isolated single-character notes",
                },
            }
        ],
        "provenance": {
            "source": PROBE_DRAFT_SOURCE,
            "phg2_schema_version": phg2_payload.get("schema_version"),
            "token_count": len(tokens),
            "excluded_tokens": sorted(excluded),
            "safety": (
                "Derived only from parsed PHG2 token strings; license, serial, "
                "activation, and unrelated DDI metadata are not collected."
            ),
            "xsampa_role": "export_only_not_internal_truth",
        },
        "metadata": {
            "probe_note_count": len(notes),
            "tokens_per_note_target": max(2, tokens_per_note),
        },
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse Miku V4C PHG2 tokens from an authorized .ddi file and build "
            "short multi-token VOCALOID probe drafts."
        )
    )
    parser.add_argument("--ddi", required=True, type=Path, help="Input .ddi path containing a PHG2 section.")
    parser.add_argument("--output-tokens-json", type=Path, help="Write parsed PHG2 token JSON.")
    parser.add_argument("--output-draft", type=Path, help="Write a vocal_track_draft_v1 probe JSON.")
    parser.add_argument(
        "--output-vsqx",
        type=Path,
        help="Write a VSQX probe by cloning --template with export_vsqx_from_template.",
    )
    parser.add_argument("--template", type=Path, help="VOCALOID4 VSQX template used when --output-vsqx is set.")
    parser.add_argument("--tokens-per-note", type=int, default=DEFAULT_TOKENS_PER_NOTE)
    parser.add_argument("--max-notes", type=int, default=DEFAULT_MAX_NOTES)
    parser.add_argument(
        "--allow-known-silent-sequences",
        action="store_true",
        help="Allow generated draft notes that match known-silent Miku V4C note-p sequences.",
    )
    parser.add_argument(
        "--exclude-known-silent",
        action="store_true",
        help="Exclude currently known silent/unsafe default tokens such as i\\, i`, and UN from generated probe drafts.",
    )
    parser.add_argument(
        "--exclude-token",
        action="append",
        default=[],
        help="Token to exclude from generated probe drafts. Can be repeated.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_vsqx and args.template is None:
        parser.error("--template is required when --output-vsqx is provided")

    try:
        phg2_payload = parse_phg2_path(args.ddi)
        excluded_tokens = set(args.exclude_token or [])
        if args.exclude_known_silent:
            excluded_tokens.update(KNOWN_SILENT_OR_UNSAFE_DEFAULT_TOKENS)
        draft_payload = build_probe_draft_payload(
            phg2_payload,
            tokens_per_note=args.tokens_per_note,
            max_notes=args.max_notes,
            exclude_tokens=excluded_tokens,
        )
        policy_issues = collect_miku_v4c_policy_issues(
            (
                str(note.get("id") or f"note-{index + 1}"),
                [str(token) for token in note.get("phonemes", [])],
            )
            for index, note in enumerate(draft_payload["tracks"][0]["notes"])
            if isinstance(note, Mapping)
        )
        if policy_issues and not args.allow_known_silent_sequences:
            raise ValueError(
                "known Miku V4C silent/unsafe phoneme sequence present; "
                + json.dumps(policy_issues, ensure_ascii=False, sort_keys=True)
            )
        if args.output_tokens_json:
            write_json(args.output_tokens_json, phg2_payload)
        if args.output_draft:
            write_json(args.output_draft, draft_payload)
        if args.output_vsqx:
            draft = VocalTrackDraft.from_mapping(draft_payload)
            export_vsqx_from_template(draft, args.output_vsqx, template_path=args.template)
    except Exception as exc:  # pragma: no cover - subprocess tests cover CLI failure shape.
        print(str(exc), file=sys.stderr)
        return 1

    summary = {
        "schema_version": "miku_v4c_phg2_probe_summary_v1",
        "ddi": str(args.ddi),
        "tokens": len(phg2_payload["tokens"]),
        "groups": len(phg2_payload["groups"]),
        "draft_notes": sum(len(track["notes"]) for track in draft_payload["tracks"]),
        "excluded_tokens": sorted(excluded_tokens),
        "phoneme_policy_issues": policy_issues,
        "outputs": {
            "tokens_json": str(args.output_tokens_json) if args.output_tokens_json else None,
            "draft": str(args.output_draft) if args.output_draft else None,
            "vsqx": str(args.output_vsqx) if args.output_vsqx else None,
        },
        "vsqx_exporter": "export_vsqx_from_template" if args.output_vsqx else None,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def _chunk_tokens(tokens: Sequence[str], *, tokens_per_note: int, max_notes: int) -> list[list[str]]:
    chunks = [list(tokens[index : index + tokens_per_note]) for index in range(0, len(tokens), tokens_per_note)]
    if len(chunks) > 1 and len(chunks[-1]) == 1:
        chunks[-2].extend(chunks.pop())
    chunks = chunks[:max_notes]
    return [chunk if len(chunk) > 1 else [chunk[0], chunk[0]] for chunk in chunks if chunk]


def _group_lookup(groups: object) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if not isinstance(groups, list):
        return lookup
    for group in groups:
        if not isinstance(group, Mapping):
            continue
        name = str(group.get("name") or "ungrouped")
        for token in _string_list(group.get("tokens")):
            lookup.setdefault(token, name)
    return lookup


def _short_probe_label(chunk: Sequence[str], groups: Mapping[str, str], *, index: int) -> str:
    group_names = [groups.get(token, "tok") for token in chunk]
    group_name = _safe_label(group_names[0] if group_names else "tok")
    token_hint = "-".join(_safe_label(token) for token in chunk[:2])
    label = f"{group_name}-{index + 1:02d}-{token_hint}".strip("-")
    if len(label) < 2:
        return f"probe-{index + 1:02d}"
    return label[:32]


def _safe_label(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "-", str(value).strip()).strip("-").lower()
    return text or "tok"


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [part for part in value.split() if part]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


if __name__ == "__main__":
    raise SystemExit(main())
