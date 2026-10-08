"""Audit audio roles and damaged IPA without rewriting historical evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ipa_corruption_markers(symbol: str) -> list[str]:
    """Detect known recovery damage, not validate the entire IPA alphabet."""
    reasons: list[str] = []
    if "?" in symbol or "\ufffd" in symbol:
        reasons.append("placeholder_or_replacement_character")
    if any("\u3400" <= char <= "\u9fff" for char in symbol):
        reasons.append("cjk_character_in_ipa_field")
    return reasons


def build_audit(
    triplets: Mapping[str, Any],
    mapping: Mapping[str, Any],
    reference: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> dict[str, Any]:
    items = triplets.get("items", [])
    by_song = {str(item["song_id"]): item for item in items}
    if len(by_song) != len(items):
        raise ValueError("triplet song IDs must be unique")
    damaged: list[dict[str, Any]] = []
    projected: dict[tuple[str, ...], list[str]] = {}
    for token, entry in mapping.get("tokens", {}).items():
        ipa = [str(symbol) for symbol in entry.get("ipa", [])]
        reasons = sorted({reason for symbol in ipa for reason in ipa_corruption_markers(symbol)})
        if reasons:
            damaged.append({"token": token, "ipa": ipa, "reasons": reasons})
        projected.setdefault(tuple(entry.get("phones", [])), []).append(token)
    damaged_tokens = {entry["token"] for entry in damaged}
    roles: Counter[str] = Counter()
    per_song_roles: dict[str, Counter[str]] = {}
    unknown_examples: list[str] = []
    for row in dataset.get("examples", []):
        song_id = str(row.get("song_id") or row.get("group_id") or "")
        item = by_song.get(song_id, {})
        digest = str(row.get("audio_sha256") or "")
        original = str(item.get("original_wav", {}).get("sha256") or "")
        rendered = str(item.get("rendered_wav", {}).get("sha256") or "")
        matches = [role for role, expected in (("original", original), ("rendered", rendered)) if digest and digest == expected]
        role = matches[0] if len(matches) == 1 else "ambiguous" if matches else "unknown"
        roles[role] += 1
        per_song_roles.setdefault(song_id, Counter())[role] += 1
        if role in {"unknown", "ambiguous"}:
            unknown_examples.append(str(row.get("example_id") or ""))
    rows = reference.get("rows", [])
    affected = [row for row in rows if damaged_tokens.intersection(row.get("v4c_tokens", []))]
    songs: list[dict[str, Any]] = []
    for song_id, item in by_song.items():
        original_duration = float(item["original_wav"].get("audio", {}).get("duration_sec", 0))
        rendered_duration = float(item["rendered_wav"].get("audio", {}).get("duration_sec", 0))
        song_rows = [row for row in rows if row.get("song_id") == song_id]
        songs.append({
            "song_id": song_id,
            "human_claims": item.get("human_claims", {}),
            "note_count": len(song_rows),
            "unresolved_note_count": sum(row.get("mapping_status") == "unresolved" for row in song_rows),
            "unresolved_tokens": sorted({token for row in song_rows for token in row.get("unmapped_v4c_tokens", [])}),
            "ipa_corruption_affected_notes": sum(bool(damaged_tokens.intersection(row.get("v4c_tokens", []))) for row in song_rows),
            "feature_audio_roles": dict(per_song_roles.get(song_id, {})),
            "original_duration_sec": original_duration,
            "rendered_duration_sec": rendered_duration,
            "duration_difference_sec": round(original_duration - rendered_duration, 6),
            "timelines_interchangeable": False,
        })
    return {
        "schema_version": "reference_dataset_integrity_audit_v1",
        "triplet_count": len(items),
        "reference_note_count": len(rows),
        "example_count": len(dataset.get("examples", [])),
        "fold_count": len(dataset.get("folds", [])),
        "feature_audio_roles": dict(roles),
        "unknown_or_ambiguous_examples": unknown_examples,
        "mapping_schema_version": mapping.get("schema_version"),
        "target_inventory_size": len(mapping.get("target_inventory", [])),
        "damaged_ipa_entries": damaged,
        "ipa_corruption_affected_notes": len(affected),
        "lossy_token_count": sum(bool(entry.get("loss")) for entry in mapping.get("tokens", {}).values()),
        "projection_collisions": [{"phones": list(phones), "raw_tokens": sorted(tokens)} for phones, tokens in sorted(projected.items()) if len(tokens) > 1],
        "songs": songs,
        "gates": {
            "audio_roles_resolved": not unknown_examples,
            "ipa_integrity_passed": not damaged,
            "original_song_evaluation": bool(roles) and set(roles) == {"original"},
            "qualified_recognizer": False,
        },
        "scope": "Legacy projection diagnostics only; no annotation or artifact is rewritten.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("triplets", "mapping", "reference", "dataset", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; use a new immutable audit path")
    names = ("triplets", "mapping", "reference", "dataset")
    inputs = {name: json.loads(getattr(args, name).read_text(encoding="utf-8-sig")) for name in names}
    report = build_audit(**inputs)
    report["sources"] = {name: {"path": str(getattr(args, name).resolve()), "sha256": _sha256(getattr(args, name))} for name in names}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("triplet_count", "reference_note_count", "example_count", "feature_audio_roles", "ipa_corruption_affected_notes", "gates")}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
