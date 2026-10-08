"""Build a conservative, hash-bound reference repair proposal (v3).

This tool only projects already existing reference notes into the accepted
alignment-audit windows.  It never edits the reference manifest, VSQX, or
mapping files and never starts a recognizer or training job.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import (  # noqa: E402
    read_json,
    sha256_file,
    verify_file,
    write_json_new,
)
from any2xsampa.training.original_reference_mapping import apply_original_mapping_overlay  # noqa: E402


DEFAULT_ACCEPTED = ROOT / ".state/runs/alignment-audit-intake-20260915T132143Z/accepted"
DEFAULT_REFERENCE = ROOT / ".state/runs/limited-data-20260906T050251/reference/reference-manifest.json"
DEFAULT_OVERLAY = ROOT / "src/any2xsampa/config/v4c_original_reference_overlay_v1.json"

MISALIGNED_IDS = {f"C{i:02d}" for i in (1, 2, 4, 6, 7, 8, 9, 12, 14, 15, 17, 18, 19)}
CORRESPONDENCE_IDS = {"C03", "C11", "C13", "C20"}
BOUNDARY_IDS = {"C05", "C10", "C16"}
DEV_IDS = {f"B{i:02d}" for i in range(1, 7)}
KNOWN_REPAIR_IDS = MISALIGNED_IDS | CORRESPONDENCE_IDS | BOUNDARY_IDS | DEV_IDS


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    return write_json_new(path, value)


def _row_map(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        review_id = row.get("review_id")
        if not isinstance(review_id, str) or not review_id:
            raise ValueError("audit rows require review_id")
        if review_id in result:
            raise ValueError(f"duplicate audit review_id: {review_id}")
        result[review_id] = row
    return result


def _notes_by_id(reference: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    notes: dict[str, Mapping[str, Any]] = {}
    songs: dict[str, Mapping[str, Any]] = {}
    for song in reference.get("songs", []):
        song_id = song.get("song_id")
        if not isinstance(song_id, str) or song_id in songs:
            raise ValueError("reference manifest has invalid or duplicate song_id")
        songs[song_id] = song
        for note in song.get("notes", []):
            note_id = note.get("note_id")
            if not isinstance(note_id, str) or note_id in notes:
                raise ValueError(f"reference manifest has invalid or duplicate note_id: {note_id}")
            notes[note_id] = note
    if not songs or not notes:
        raise ValueError("reference manifest contains no notes")
    return notes, songs


def _validate_overlay(reference: Mapping[str, Any], overlay_path: Path) -> dict[str, Any]:
    overlay = read_json(overlay_path)
    if overlay.get("schema_version") != "v4c_original_reference_overlay_v1":
        raise ValueError("unsupported mapping overlay schema")
    binding = reference.get("bindings", {}).get("mapping", {})
    mapping_path = Path(binding.get("path", ""))
    expected_mapping_hash = binding.get("sha256")
    if not mapping_path.is_file() or not isinstance(expected_mapping_hash, str):
        raise ValueError("reference manifest mapping binding is missing")
    actual_mapping_hash = sha256_file(mapping_path)
    if actual_mapping_hash != expected_mapping_hash:
        raise ValueError("reference mapping source hash mismatch")
    if overlay.get("base_mapping_sha256") != expected_mapping_hash:
        raise ValueError("mapping overlay base hash does not match reference mapping")
    # These flags are safety claims, not instructions to reinterpret a label.
    if overlay.get("historical_mapping_or_annotations_modified") not in (False, None):
        raise ValueError("mapping overlay claims historical annotations were modified")
    if overlay.get("general_string_splitting_allowed") not in (False, None):
        raise ValueError("mapping overlay enables unauthorised string splitting")
    return overlay


def _verify_source_bindings(audit_rows: Iterable[Mapping[str, Any]], songs: Mapping[str, Mapping[str, Any]]) -> None:
    for row in audit_rows:
        song_id = row.get("song_id")
        if song_id not in songs:
            raise ValueError(f"audit row references unknown song: {song_id}")
        song = songs[song_id]
        expected_audio = song.get("original_audio", {}).get("sha256")
        expected_vsqx = song.get("human_vsqx", {}).get("sha256")
        if row.get("source_audio_sha256") != expected_audio:
            raise ValueError(f"source audio hash mismatch for {row.get('review_id')}")
        if row.get("source_vsqx_sha256") != expected_vsqx:
            raise ValueError(f"source VSQX hash mismatch for {row.get('review_id')}")


def _crop(row: Mapping[str, Any]) -> dict[str, Any]:
    binding = row.get("rendered_preview_binding") or {}
    source = binding.get("source_crop") or {}
    return {
        "effective_start_sec": source.get("effective_start_sec"),
        "effective_end_sec": source.get("effective_end_sec"),
        "requested_start_sec": source.get("requested_start_sec"),
        "requested_end_sec": source.get("requested_end_sec"),
        "model_sample_rate": source.get("model_sample_rate"),
        "model_samples": source.get("model_samples"),
        "audio_sha256": binding.get("audio_sha256"),
    }


def _explicit_note_ids(item: Mapping[str, Any]) -> set[str]:
    return {
        str(ref["note_id"])
        for ref in item.get("bound_existing_notes", [])
        if isinstance(ref, Mapping) and isinstance(ref.get("note_id"), str)
    }


def _overlapping_notes(song: Mapping[str, Any], start: Any, end: Any) -> list[Mapping[str, Any]]:
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return []
    if end <= start:
        return []
    result = []
    for note in song.get("notes", []):
        nstart, nend = note.get("start_sec"), note.get("end_sec")
        if isinstance(nstart, (int, float)) and isinstance(nend, (int, float)) and nstart < end and nend > start:
            result.append(note)
    return result


def _boundary_state(row: Mapping[str, Any]) -> str | None:
    final = row.get("final_audit_response") or {}
    previous = row.get("previous_human_response") or {}
    return final.get("boundary_state") or previous.get("boundary_state")


def _candidate_kind(review_id: str, row: Mapping[str, Any]) -> str:
    if review_id in MISALIGNED_IDS or row.get("decision") == "misaligned":
        return "historical_misalignment_excluded"
    if review_id in CORRESPONDENCE_IDS:
        return "correspondence_only"
    if review_id in BOUNDARY_IDS:
        return "boundary_repair"
    if review_id in DEV_IDS:
        return "dev_boundary_evidence"
    return "unclassified_excluded"


def _phone_rows(note: Mapping[str, Any], *, edge_unresolved: set[int], excluded: bool, reason: str | None) -> list[dict[str, Any]]:
    ipa = note.get("model_target_ipa")
    if not isinstance(ipa, list):
        ipa = []
    mapping_status = note.get("mapping_status", "unresolved")
    result = []
    for index, symbol in enumerate(ipa):
        status = "resolved_existing_mapping" if mapping_status == "resolved_proxy" else "unresolved_mapping"
        phone_reason = None
        if index in edge_unresolved:
            status = "unresolved_boundary"
            phone_reason = "boundary_phoneme_not_safely_decidable"
        if excluded:
            status = "excluded"
            phone_reason = reason
        result.append({"phone_index": index, "ipa": symbol, "status": status, "reason": phone_reason})
    return result


def _note_projection(note: Mapping[str, Any], *, in_window: bool, edge_unresolved: set[int], excluded_reason: str | None) -> dict[str, Any]:
    mapping_status = note.get("mapping_status", "unresolved")
    base_reason = excluded_reason
    if not in_window:
        base_reason = "outside_proposed_full_window"
    elif mapping_status != "resolved_proxy":
        base_reason = "existing_mapping_unresolved"
    retained = in_window and mapping_status == "resolved_proxy" and not edge_unresolved and not excluded_reason
    return {
        "note_id": note.get("note_id"),
        "lyric": note.get("raw_lyric"),
        "start_sec": note.get("start_sec"),
        "end_sec": note.get("end_sec"),
        "ipa": list(note.get("model_target_ipa") or []),
        "legacy_ipa": list(note.get("legacy_intended_ipa") or []),
        "mapping_status": mapping_status,
        "in_proposed_full_window": in_window,
        "retained": retained,
        "excluded_reason": base_reason,
        "phones": _phone_rows(note, edge_unresolved=edge_unresolved, excluded=bool(base_reason), reason=base_reason),
    }


def _make_candidate(row: Mapping[str, Any], item: Mapping[str, Any] | None, notes_by_id: Mapping[str, Mapping[str, Any]], songs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rid = str(row["review_id"])
    kind = _candidate_kind(rid, row)
    song = songs[row["song_id"]]
    crop = _crop(row)
    start, end = crop.get("effective_start_sec"), crop.get("effective_end_sec")
    overlap = _overlapping_notes(song, start, end)
    by_id = {str(n.get("note_id")): n for n in overlap}
    explicit = _explicit_note_ids(item or {})
    # Existing worklist IDs are anchors only; the full set is derived from the
    # already parsed reference note timing.  This avoids asking for nxxx input.
    for note_id in explicit:
        if note_id in notes_by_id:
            by_id.setdefault(note_id, notes_by_id[note_id])
    ordered = sorted(by_id.values(), key=lambda n: (n.get("start_sec", 0), n.get("note_id", "")))
    edge_state = _boundary_state(row)
    edge_notes: dict[str, set[int]] = {}
    if kind == "boundary_repair" and ordered:
        if edge_state in {"truncated_start", "truncated_both", "unknown"}:
            first = ordered[0]
            first_ipa = first.get("model_target_ipa") or []
            edge_notes[str(first["note_id"])] = set(range(len(first_ipa)))
        if edge_state in {"truncated_end", "truncated_both", "unknown"}:
            last = ordered[-1]
            last_ipa = last.get("model_target_ipa") or []
            if last_ipa:
                edge_notes.setdefault(str(last["note_id"]), set()).update(range(len(last_ipa)))
        # A word-level remark cannot identify which phone survived a cut.
        # Keep explicitly named edge notes unresolved rather than guessing.
        for note_id in explicit & set(by_id):
            ipa = by_id[note_id].get("model_target_ipa") or []
            edge_notes[note_id] = set(range(len(ipa)))
    c20_risk = False
    if rid == "C20":
        c20_risk = any(str(n.get("raw_lyric", "")).strip() in {"說", "说"} for n in ordered)
        if not c20_risk:
            # Preserve the human answer's warning even if a reduced fixture
            # omitted the lyric row; this is still not an inferred mapping.
            c20_risk = any(str(ref.get("source_lyric", "")).strip() in {"說", "说"} for ref in (item or {}).get("bound_existing_notes", []))
    notes = []
    for note in ordered:
        note_id = str(note["note_id"])
        reason = None
        if kind == "historical_misalignment_excluded":
            reason = "historical_audit_misaligned_never_revived"
        elif c20_risk and note.get("raw_lyric") in {"\u8aaa", "\u8bf4"}:
            reason = "C20_missing_or_severely_truncated_lyric_risk"
        elif kind == "correspondence_only":
            reason = "correspondence_candidate_not_formal_phone_reference"
        elif kind == "dev_boundary_evidence":
            reason = "dev_boundary_evidence_not_a_full_dev_window"
        notes.append(_note_projection(note, in_window=True, edge_unresolved=edge_notes.get(note_id, set()), excluded_reason=reason))
    # Explicit notes outside the crop remain visible and excluded (C16 is the
    # intended example), so the result never silently drops an audit binding.
    for note_id in sorted(explicit - set(by_id)):
        note = notes_by_id[note_id]
        notes.append(_note_projection(note, in_window=False, edge_unresolved=set(), excluded_reason="outside_proposed_full_window"))
    notes.sort(key=lambda n: (n.get("start_sec", 0), n.get("note_id", "")))
    excluded_reasons: list[str] = []
    if kind == "historical_misalignment_excluded":
        excluded_reasons.append("historical_audit_misaligned_never_revived")
    if kind == "correspondence_only":
        excluded_reasons.append("correspondence_only_formal_phone_reference_created_false")
    if kind == "dev_boundary_evidence":
        excluded_reasons.append("B01-B06_are_boundary_evidence_not_complete_dev_windows")
    if kind == "boundary_repair" and edge_notes:
        excluded_reasons.append("unresolved_first_or_last_phoneme")
    if c20_risk:
        excluded_reasons.append("C20_missing_or_severely_truncated_lyric_risk_not_counted")
    # A proposal is deliberately never promoted to a formal phone reference in
    # this phase, including corresponding/boundary rows.
    retained = kind in {"correspondence_only", "boundary_repair"} and bool(notes) and not (kind == "correspondence_only" and rid == "C20" and c20_risk)
    return {
        "review_id": rid,
        "candidate_kind": kind,
        "song_id": row.get("song_id"),
        "split": row.get("split"),
        "decision": row.get("decision"),
        "source_example_id": row.get("source_example_id"),
        "boundary_state": edge_state,
        "original_audio_window": {
            "start_sec": row.get("start_sec"),
            "end_sec": row.get("end_sec"),
            "original_cut_sec": row.get("original_cut_sec"),
            "cut_in_preview_sec": row.get("cut_in_preview_sec"),
            "side": row.get("side"),
        },
        "rendered_audio_window": crop,
        "source_audio_sha256": row.get("source_audio_sha256"),
        "source_vsqx_sha256": row.get("source_vsqx_sha256"),
        "notes": notes,
        "retained_as_candidate": retained,
        "formal_phone_reference_created": False,
        "training_eligible": False,
        "dev_full_window": False if kind == "dev_boundary_evidence" else None,
        "excluded_reasons": excluded_reasons,
        "audit_answer_source": (item or {}).get("human_answer_verbatim") or row.get("final_audit_response") or row.get("previous_human_response"),
    }


def build_report(effective: Mapping[str, Any], worklist: Mapping[str, Any], reference: Mapping[str, Any], overlay: Mapping[str, Any], *, input_bindings: Mapping[str, Any], candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    training = [c for c in candidates if c["split"] == "train" and c["candidate_kind"] in {"correspondence_only", "boundary_repair"} and c["retained_as_candidate"] and c["formal_phone_reference_created"]]
    dev_full = [c for c in candidates if c["split"] == "dev" and c["dev_full_window"] is True]
    train_songs = {c["song_id"] for c in training}
    train_phones = sum(sum(1 for n in c["notes"] for p in n["phones"] if p["status"] == "resolved_existing_mapping") for c in training)
    dev_phones = sum(sum(1 for n in c["notes"] for p in n["phones"] if p["status"] == "resolved_existing_mapping") for c in dev_full)
    boundary_evidence = [c for c in candidates if c["candidate_kind"] == "dev_boundary_evidence"]
    counts = {
        "audit_rows": len(effective.get("rows", [])),
        "training_rows": sum(1 for r in effective.get("rows", []) if r.get("split") == "train"),
        "misaligned_training_excluded": sum(1 for c in candidates if c["candidate_kind"] == "historical_misalignment_excluded"),
        "correspondence_candidates": sum(1 for c in candidates if c["candidate_kind"] == "correspondence_only"),
        "boundary_repair_candidates": sum(1 for c in candidates if c["candidate_kind"] == "boundary_repair"),
        "dev_boundary_evidence_candidates": len(boundary_evidence),
        "formal_phone_reference_created": 0,
    }
    gates = {
        "training": {"required_windows": 20, "required_songs": 3, "actual_windows": len(training), "actual_songs": len(train_songs), "actual_reference_phones": train_phones, "passed": False},
        "shalong_dev": {"required_windows": 3, "required_reference_phones": 100, "actual_windows": len(dev_full), "actual_reference_phones": dev_phones, "boundary_evidence_windows_not_counted": len(boundary_evidence), "passed": False},
        "training_gate_met": False,
        "dev_gate_met": False,
        "all_gates_met": False,
    }
    coverage = {
        "training_windows": len(training),
        "training_songs": len(train_songs),
        "training_reference_phones": train_phones,
        "shalong_full_dev_windows": len(dev_full),
        "shalong_full_dev_reference_phones": dev_phones,
        "shalong_boundary_evidence_windows": len(boundary_evidence),
        "formal_phone_reference_created": False,
    }
    report = {
        "schema_version": "reference_repair_v3_coverage_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "conservative_reference_repair_candidates_only",
        "formal_phone_reference_created": False,
        "starts_training": False,
        "inputs": dict(input_bindings),
        "reference_manifest_hash": input_bindings["reference_manifest"]["sha256"],
        "mapping_overlay": {"sha256": input_bindings["mapping_overlay"]["sha256"], "base_mapping_sha256": overlay.get("base_mapping_sha256")},
        "counts": counts,
        "coverage": coverage,
        "gates": gates,
        "candidates": [{"review_id": c["review_id"], "candidate_kind": c["candidate_kind"], "retained_as_candidate": c["retained_as_candidate"], "formal_phone_reference_created": False, "excluded_reasons": c["excluded_reasons"]} for c in candidates],
        "policy": {"historical_misalignment_never_revived": True, "c20_lyric_risk_not_counted": True, "dev_boundary_evidence_not_full_dev": True, "new_mapping_meanings_guessed": False, "original_files_modified": False},
    }
    return report, {"training": training, "dev_full": dev_full}


def build(args: argparse.Namespace) -> Path:
    accepted = args.accepted_dir.resolve()
    effective_path = (args.effective_audit or accepted / "effective-audit.json").resolve()
    worklist_path = (args.worklist or accepted / "reference-repair-worklist.json").resolve()
    reference_path = (args.reference or DEFAULT_REFERENCE).resolve()
    overlay_path = (args.mapping_overlay or DEFAULT_OVERLAY).resolve()
    for path in (effective_path, worklist_path, reference_path, overlay_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    effective = read_json(effective_path)
    worklist = read_json(worklist_path)
    reference = read_json(reference_path)
    overlay = _validate_overlay(reference, overlay_path)
    # Apply only the already pinned evidence-backed projection (ua+n). This
    # produces a derived view for proposals; the source manifest remains intact.
    bundle_path = accepted.parent / "raw/bundle.json"
    bundle = read_json(bundle_path)
    expected_reference = bundle["sources"]["reference"]
    if str(reference_path.resolve()) != str(Path(expected_reference["path"]).resolve()):
        raise ValueError("reference path does not match the hash-bound audit source")
    if sha256_file(reference_path) != expected_reference["sha256"]:
        raise ValueError("reference manifest source hash mismatch")
    expected_overlay = bundle["mapping_overlay"]
    if str(overlay_path.resolve()) != str(Path(expected_overlay["path"]).resolve()):
        raise ValueError("mapping overlay path does not match the hash-bound audit source")
    if sha256_file(overlay_path) != expected_overlay["sha256"]:
        raise ValueError("mapping overlay source hash mismatch")
    manual_evidence_path = Path(bundle["mapping_overlay"]["manual_evidence_path"])
    verify_file(manual_evidence_path, bundle["mapping_overlay"]["manual_evidence_sha256"])
    reference, _overlay_binding = apply_original_mapping_overlay(reference, manual_evidence_path)
    notes_by_id, songs = _notes_by_id(reference)
    rows = list(effective.get("rows", []))
    row_by_id = _row_map(rows)
    items = _row_map(worklist.get("items", []))
    _verify_source_bindings(rows, songs)
    if effective.get("input_sha256") != worklist.get("input_sha256") or effective.get("bundle_id") != worklist.get("bundle_id"):
        raise ValueError("effective audit and repair worklist source bindings differ")
    expected_ids = {f"C{i:02d}" for i in range(1, 21)} | DEV_IDS
    if set(row_by_id) != expected_ids:
        raise ValueError("effective audit IDs must be exactly C01-C20 plus B01-B06")
    answered_ids = {rid for rid, row in row_by_id.items() if row.get("decision_source") == "final_audit_file"}
    if set(items) != answered_ids:
        raise ValueError("repair worklist IDs must match the explicitly answered audit rows")
    candidates: list[dict[str, Any]] = []
    for rid in sorted(expected_ids):
        candidates.append(_make_candidate(row_by_id[rid], items.get(rid), notes_by_id, songs))
    candidates.sort(key=lambda c: c["review_id"])
    input_bindings = {
        "effective_audit": _binding(effective_path),
        "worklist": _binding(worklist_path),
        "reference_manifest": _binding(reference_path),
        "mapping_overlay": _binding(overlay_path),
        "mapping_evidence": _binding(manual_evidence_path),
    }
    coverage, _ = build_report(effective, worklist, reference, overlay, input_bindings=input_bindings, candidates=candidates)
    proposals = {
        "schema_version": "reference_repair_v3_proposals_v1",
        "generated_at_utc": coverage["generated_at_utc"],
        "scope": coverage["scope"],
        "formal_phone_reference_created": False,
        "starts_training": False,
        "inputs": input_bindings,
        "proposals": candidates,
        "policy": coverage["policy"],
    }
    output = (args.output_dir or ROOT / ".state/runs" / f"reference-repair-v3-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}").resolve()
    if not output.is_relative_to(ROOT / ".state/runs"):
        raise ValueError("output must be under .state/runs")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    proposal_hash = _write_json(output / "repair-proposals.json", proposals)
    coverage["repair_proposals_sha256"] = proposal_hash
    coverage_hash = _write_json(output / "coverage-report.json", coverage)
    results = [
        "# Reference Repair v3",
        "",
        "This is a conservative, replayable proposal report. Original VSQX/reference/mapping files were not modified.",
        "No model, final-test song, or training job was opened.",
        "",
        f"- Historical misaligned training candidates excluded: {coverage['counts']['misaligned_training_excluded']}",
        f"- Correspondence-only candidates: {coverage['counts']['correspondence_candidates']} (formal_phone_reference_created=false)",
        f"- Boundary repair candidates: {coverage['counts']['boundary_repair_candidates']} (edge uncertainty remains unresolved where applicable)",
        f"- B01-B06 dev boundary evidence: {coverage['counts']['dev_boundary_evidence_candidates']} (not full dev windows)",
        "",
        "## Gates",
        "",
        f"- Training gate (>=20 windows / >=3 songs): **false** ({coverage['coverage']['training_windows']} / {coverage['coverage']['training_songs']})",
        f"- Shalong dev gate (>=3 windows / >=100 reference phones): **false** ({coverage['coverage']['shalong_full_dev_windows']} / {coverage['coverage']['shalong_full_dev_reference_phones']})",
        "- formal_phone_reference_created: **false**",
        "- starts_training: **false**",
        "",
        f"Input reference manifest SHA-256: `{coverage['reference_manifest_hash']}`",
        f"Repair proposals SHA-256: `{proposal_hash}`",
        f"Coverage report SHA-256: `{coverage_hash}`",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(results), encoding="utf-8", newline="\n")
    verification = {
        "schema_version": "reference_repair_v3_verification_v1",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": input_bindings,
        "outputs": {"repair_proposals": {"path": "repair-proposals.json", "sha256": proposal_hash}, "coverage_report": {"path": "coverage-report.json", "sha256": coverage_hash}, "results": {"path": "RESULTS.md", "sha256": sha256_file(output / "RESULTS.md")}},
        "checks": {"historical_misaligned_excluded": coverage["counts"]["misaligned_training_excluded"] == len(MISALIGNED_IDS), "c20_risk_not_counted": any(c["review_id"] == "C20" and any("C20_missing" in r for r in c["excluded_reasons"]) for c in candidates), "B_dev_evidence_not_full": all(c["dev_full_window"] is False for c in candidates if c["candidate_kind"] == "dev_boundary_evidence"), "formal_phone_reference_created": False, "starts_training": False, "training_gate_met": False, "shalong_dev_gate_met": False, "original_files_modified": False},
    }
    _write_json(output / "VERIFICATION.json", verification)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build conservative reference-repair-v3 coverage/proposals.")
    parser.add_argument("--accepted-dir", type=Path, default=DEFAULT_ACCEPTED)
    parser.add_argument("--effective-audit", type=Path)
    parser.add_argument("--worklist", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--mapping-overlay", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        output = build(args)
    except (FileNotFoundError, FileExistsError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
