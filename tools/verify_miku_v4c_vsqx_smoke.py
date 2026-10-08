from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import wave
from collections.abc import Mapping
from pathlib import Path
from xml.etree import ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from any2xsampa.adapters.forced_alignment_acceptance import (
    FULL_SONG_SCOPE,
    HUMAN_REVIEW_ROUTE,
    QUALIFIED_MODEL_ROUTE,
    RESOURCE_MANIFEST_SCHEMA_VERSION,
    SHORT_CLIP_SCOPE,
    ForcedPhoneAcceptanceError,
    canonical_phone_inventory_sha256,
    load_forced_phone_acceptance,
)
from any2xsampa.adapters.forced_alignment_acceptance import (
    SCHEMA_VERSION as ACCEPTANCE_SCHEMA_VERSION,
)
from any2xsampa.core.models import TimeRange
from any2xsampa.vocaloid import collect_miku_v4c_policy_issues
from any2xsampa.vocaloid.vsqx import VSQ4_NS


def _configure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


NS = {"v": VSQ4_NS}
EXPECTED_SMOKE = [
    ("\u559c", "x ei"),
    ("\u6b61", "f u n"),
    ("\u4f60", "n ei"),
]
FORBIDDEN_PHONEMES = {"h e i", "n e i", "f UN"}
FORBIDDEN_RAW = ["i\\", "i`"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the Phase 1 Miku V4C import/playback smoke VSQX contract at the XML/artifact level."
    )
    parser.add_argument("vsqx", type=Path)
    parser.add_argument("--expected-smoke", action="store_true", help="Require the Miku V4C smoke phrase x ei / f u n / n ei")
    parser.add_argument("--summary", type=Path, help="Optional export_vsqx.py summary JSON to verify")
    parser.add_argument("--product-provenance", type=Path, help="Require pitch/phone/lyric product evidence gates")
    parser.add_argument("--output", type=Path, help="Optional JSON report output")
    return parser


def verify_vsqx(path: Path, *, expected_smoke: bool) -> dict[str, object]:
    errors: list[str] = []
    if not path.exists():
        return {"ok": False, "errors": [f"missing VSQX: {path}"], "vsqx": str(path)}
    raw = path.read_text(encoding="utf-8-sig")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        return {"ok": False, "errors": [f"XML parse failed: {exc}"], "vsqx": str(path)}
    if root.tag != f"{{{VSQ4_NS}}}vsq4":
        errors.append(f"unexpected root tag: {root.tag}")
    notes = root.findall(".//v:note", NS)
    note_rows = [
        {
            "lyric": note.findtext("v:y", namespaces=NS) or "",
            "phonemes": note.findtext("v:p", namespaces=NS) or "",
            "start_tick": note.findtext("v:t", namespaces=NS) or "",
            "duration_ticks": note.findtext("v:dur", namespaces=NS) or "",
            "pitch": note.findtext("v:n", namespaces=NS) or "",
        }
        for note in notes
    ]
    phoneme_values = [row["phonemes"] for row in note_rows]
    policy_issues = collect_miku_v4c_policy_issues(
        (f"note-{index + 1}", [token for token in row["phonemes"].split() if token])
        for index, row in enumerate(note_rows)
    )
    for row in policy_issues:
        errors.append(
            "known Miku V4C silent/unsafe phoneme policy issue: "
            + json.dumps(row, ensure_ascii=False, sort_keys=True)
        )
    for forbidden in FORBIDDEN_PHONEMES:
        if forbidden in phoneme_values:
            message = f"forbidden known-silent phoneme sequence present: {forbidden}"
            if message not in errors:
                errors.append(message)
    for forbidden in FORBIDDEN_RAW:
        if forbidden in raw:
            message = f"forbidden known-silent raw token present: {forbidden}"
            if message not in errors:
                errors.append(message)
    if "> @N <" in raw or ">@N<" in raw:
        errors.append("bare @N token appears as a serialized text node")
    previous_start = -1
    for index, row in enumerate(note_rows):
        try:
            start_tick = int(row["start_tick"])
            duration_ticks = int(row["duration_ticks"])
            pitch = int(row["pitch"])
        except (TypeError, ValueError):
            errors.append(f"note {index + 1} timing/pitch fields must be integers")
            continue
        if start_tick < 0 or start_tick < previous_start:
            errors.append(f"note {index + 1} start tick is negative or out of order")
        if duration_ticks <= 0:
            errors.append(f"note {index + 1} duration must be positive")
        if not 0 <= pitch <= 127:
            errors.append(f"note {index + 1} pitch is outside MIDI range")
        if not row["lyric"].strip():
            errors.append(f"note {index + 1} lyric is empty")
        previous_start = start_tick
    if expected_smoke:
        observed = [(row["lyric"], row["phonemes"]) for row in note_rows]
        if observed[:3] != EXPECTED_SMOKE:
            errors.append(f"smoke phrase mismatch: expected {EXPECTED_SMOKE!r}, observed {observed[:3]!r}")
    return {
        "ok": not errors,
        "errors": errors,
        "vsqx": str(path),
        "root": root.tag,
        "note_count": len(notes),
        "notes": note_rows,
        "phoneme_policy_issues": policy_issues,
        "manual_playback_required": True,
    }


def verify_summary(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    if not path.exists():
        return {"ok": False, "errors": [f"missing summary: {path}"], "summary": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    errors: list[str] = []
    if payload.get("export_mode") != "template_clone_import_safe":
        errors.append("summary export_mode is not template_clone_import_safe")
    if payload.get("recommended_for_vocaloid4_fe") is not True:
        errors.append("summary recommended_for_vocaloid4_fe is not true")
    return {"ok": not errors, "errors": errors, "summary": str(path), "payload": payload}


def verify_product_provenance(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    if not path.exists():
        return {"ok": False, "errors": [f"missing product provenance: {path}"], "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"ok": False, "errors": [f"invalid product provenance: {exc}"], "path": str(path)}
    if not isinstance(payload, Mapping):
        return {"ok": False, "errors": ["product provenance must be a JSON object"], "path": str(path)}
    errors: list[str] = []
    canonical = payload.get("schema_version") == "miku_v4c_audio_evidence_provenance_v1"
    fusion = payload.get("evidence_fusion")
    if not isinstance(fusion, dict):
        errors.append("product provenance has no evidence_fusion object")
    else:
        try:
            direct_ratio = float(fusion.get("direct_overlap_ratio", 0.0))
            interpolation_ratio = float(fusion.get("interpolation_ratio", 1.0))
        except (TypeError, ValueError, OverflowError) as exc:
            errors.append(f"pitch/phone overlap ratios must be numeric: {exc}")
            direct_ratio = interpolation_ratio = 0.0
        if not math.isfinite(direct_ratio) or not math.isfinite(interpolation_ratio):
            errors.append("pitch/phone overlap ratios must be finite")
        if not 0.0 <= direct_ratio <= 1.0 or not 0.0 <= interpolation_ratio <= 1.0:
            errors.append("pitch/phone overlap ratios must be between 0 and 1")
        interpolations = fusion.get("interpolations")
        if direct_ratio < 0.95:
            errors.append(f"direct pitch/phone overlap ratio is below 0.95: {direct_ratio}")
        if interpolation_ratio > 0.05:
            errors.append(f"pitch/phone interpolation ratio exceeds 0.05: {interpolation_ratio}")
        if not isinstance(interpolations, list):
            errors.append("pitch/phone interpolation audit is missing")
        else:
            try:
                interpolation_count = int(fusion.get("interpolation_count", -1))
            except (TypeError, ValueError, OverflowError):
                interpolation_count = -1
            if len(interpolations) != interpolation_count:
                errors.append("pitch/phone interpolation audit count does not match")
            if canonical:
                _check_count(errors, fusion, "note_count", label="fusion note")
                _check_count(errors, fusion, "direct_overlap_count", label="fusion direct-overlap")
                if _count(fusion, "note_count") is not None and _count(fusion, "direct_overlap_count") is not None:
                    expected_direct = _count(fusion, "note_count") - interpolation_count
                    if _count(fusion, "direct_overlap_count") != expected_direct:
                        errors.append("fusion direct/interpolation counts do not cover note_count")
                _check_threshold(errors, fusion, "direct_overlap_threshold", 0.95, minimum=True)
                _check_threshold(errors, fusion, "maximum_interpolation_ratio", 0.05, minimum=False)
                if abs((direct_ratio + interpolation_ratio) - 1.0) > 1e-6:
                    errors.append("pitch/phone direct and interpolation ratios must sum to 1")
                _verify_fusion_coverage(fusion, errors)
    if not payload.get("lyric_source"):
        errors.append("product provenance has no lyric source")
    phone = payload.get("phone_evidence")
    if not isinstance(phone, dict) or not phone.get("provided"):
        errors.append("product provenance has no phone evidence")
    pitch = payload.get("pitch_evidence")
    if not isinstance(pitch, dict) or not pitch.get("provided"):
        errors.append("product provenance has no pitch evidence")
    if canonical:
        _verify_canonical_audio_and_priority(payload, errors)
        if payload.get("manual_vocaloid4_import_required") is not True:
            errors.append("canonical provenance must require manual Vocaloid4 import")
        if payload.get("manual_acceptance_status") != "pending_user_validation":
            errors.append("canonical provenance acceptance status must remain pending user validation")
    if payload.get("product_mode") == "forced_alignment_product":
        stem = payload.get("vocal_stem")
        if not isinstance(stem, dict) or not stem.get("path") or not _sha256(stem.get("sha256")):
            errors.append("forced product provenance has no hash-bound vocal stem")
        timeline = payload.get("lyrics_timeline_evidence")
        if (
            not isinstance(timeline, Mapping)
            or not timeline.get("provided")
            or (_count(timeline, "item_count") or 0) <= 0
            or not _sha256(timeline.get("audio_sha256"))
            or not _sha256(timeline.get("timeline_sha256"))
        ):
            errors.append("forced product provenance has no lyric timeline evidence")
        g2p = payload.get("g2p_evidence")
        if (
            not isinstance(g2p, Mapping)
            or (_count(g2p, "item_count") or 0) <= 0
            or g2p.get("oov_items")
            or not _sha256(g2p.get("audio_sha256"))
            or not _sha256(g2p.get("lyrics_timeline_sha256"))
            or not _sha256(g2p.get("artifact_sha256"))
        ):
            errors.append("forced product provenance has missing or OOV G2P evidence")
        forced = payload.get("forced_alignment_evidence")
        if not isinstance(forced, Mapping) or not forced.get("provided") or (_count(forced, "covered_lyric_item_count") or 0) <= 0:
            errors.append("forced product provenance has no complete forced alignment evidence")
        elif isinstance(stem, Mapping):
            _verify_forced_acceptance(forced, stem=stem, errors=errors, base_dir=path.parent)
        if payload.get("manual_vocaloid4_import_required") is not True:
            errors.append("forced product provenance must require manual Vocaloid4 import")
        if payload.get("manual_acceptance_status") != "pending_user_validation":
            errors.append("forced product provenance acceptance status must remain pending user validation")
        if canonical:
            _verify_canonical_forced_evidence(payload, errors)
    return {"ok": not errors, "errors": errors, "path": str(path), "payload": payload}


def _count(value: Mapping[str, object], field: str) -> int | None:
    raw = value.get(field)
    if isinstance(raw, bool):
        return None
    try:
        parsed = int(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None


def _check_count(errors: list[str], value: Mapping[str, object], field: str, *, label: str) -> None:
    parsed = _count(value, field)
    if parsed is None:
        errors.append(f"{label} {field} must be a non-negative integer")


def _check_threshold(
    errors: list[str], value: Mapping[str, object], field: str, limit: float, *, minimum: bool
) -> None:
    if field not in value:
        errors.append(f"fusion {field} is missing")
        return
    try:
        parsed = float(value[field])
    except (TypeError, ValueError, OverflowError):
        errors.append(f"fusion {field} must be numeric")
        return
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        errors.append(f"fusion {field} must be finite and between 0 and 1")
    elif (minimum and parsed < limit) or (not minimum and parsed > limit):
        comparator = "at least" if minimum else "at most"
        errors.append(f"fusion {field} must be {comparator} {limit}")


def _verify_canonical_audio_and_priority(payload: Mapping[str, object], errors: list[str]) -> None:
    audio = payload.get("audio")
    if not isinstance(audio, Mapping) or not audio.get("path") or not _sha256(audio.get("sha256")):
        errors.append("canonical provenance has no hash-bound source audio")
    else:
        _verify_file_hash(errors, audio, label="source audio")
    stem = payload.get("vocal_stem")
    stem_hash = stem.get("sha256") if isinstance(stem, Mapping) else None
    if payload.get("product_mode") == "forced_alignment_product" and not _sha256(stem_hash):
        errors.append("canonical forced provenance has no valid vocal-stem hash")
    elif isinstance(stem, Mapping):
        _verify_file_hash(errors, stem, label="vocal stem")

    priority = payload.get("evidence_source_priority")
    expected_prefix = ["user_lyrics", "external_asr", "qwen3_asr"]
    if not isinstance(priority, list) or priority[:3] != expected_prefix or len(set(priority)) != len(priority):
        errors.append("evidence source priority must preserve user lyrics > external ASR > Qwen3 ASR")
    else:
        source = str(payload.get("lyric_source") or "")
        if source in expected_prefix:
            asr = payload.get("asr_evidence")
            asr_available = isinstance(asr, Mapping) and bool(asr.get("provided"))
            available = {
                "user_lyrics": source == "user_lyrics",
                "external_asr": source == "external_asr" or (asr_available and source != "qwen3_asr"),
                "qwen3_asr": source == "qwen3_asr" or (asr_available and source == "qwen3_asr"),
            }
            highest = next((candidate for candidate in expected_prefix if available.get(candidate)), source)
            if source != highest:
                errors.append("selected lyric source is not the highest-priority available evidence")

    pitch = payload.get("pitch_evidence")
    if isinstance(pitch, Mapping) and pitch.get("provided"):
        pitch_path = pitch.get("pitch_audio") or pitch.get("audio_path")
        pitch_hash = _first_hash(pitch, "pitch_audio_sha256", "audio_sha256", "sha256")
        if not pitch_path or not _sha256(pitch_hash):
            errors.append("canonical pitch evidence must include pitch audio path and SHA-256")
        elif isinstance(pitch_path, (str, Path)):
            _verify_file_hash(errors, {"path": pitch_path, "sha256": pitch_hash}, label="pitch audio")
        if pitch.get("note_candidate_count") is None or _count(pitch, "note_candidate_count") is None:
            errors.append("canonical pitch evidence must include note_candidate_count")

    timeline = payload.get("lyrics_timeline_evidence")
    if isinstance(timeline, Mapping) and timeline.get("provided"):
        if _count(timeline, "item_count") is None or _count(timeline, "item_count") == 0:
            errors.append("canonical lyric timeline must include a positive item_count")
        if not _sha256(timeline.get("audio_sha256")) or not _sha256(timeline.get("timeline_sha256")):
            errors.append("canonical lyric timeline must include audio and timeline SHA-256")
        if isinstance(audio, Mapping) and _sha256(audio.get("sha256")) and timeline.get("audio_sha256") != audio.get("sha256"):
            errors.append("lyric timeline audio hash does not match source audio hash")
        timeline_path = payload.get("artifacts")
        if isinstance(timeline_path, Mapping):
            declared_path = timeline_path.get("lyrics_timeline")
            if declared_path:
                _verify_json_artifact_hash(errors, declared_path, timeline.get("timeline_sha256"), label="lyrics timeline")


def _first_hash(value: Mapping[str, object], *fields: str) -> object:
    for field in fields:
        if value.get(field) is not None:
            return value.get(field)
    return None


def _verify_file_hash(errors: list[str], value: Mapping[str, object], *, label: str) -> None:
    raw_path = value.get("path")
    declared = str(value.get("sha256") or "").lower()
    if not _sha256(declared) or not raw_path:
        return
    path = Path(str(raw_path))
    if not path.is_file():
        errors.append(f"canonical {label} path does not exist: {path}")
        return
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        errors.append(f"cannot read canonical {label}: {exc}")
        return
    if digest.hexdigest() != declared:
        errors.append(f"canonical {label} SHA-256 does not match its path")


def _verify_canonical_forced_evidence(payload: Mapping[str, object], errors: list[str]) -> None:
    timeline = payload.get("lyrics_timeline_evidence")
    timeline_hash = timeline.get("timeline_sha256") if isinstance(timeline, Mapping) else None
    timeline_audio_hash = timeline.get("audio_sha256") if isinstance(timeline, Mapping) else None
    timeline_count = _count(timeline, "item_count") if isinstance(timeline, Mapping) else None
    stem = payload.get("vocal_stem")
    stem_hash = stem.get("sha256") if isinstance(stem, Mapping) else None

    g2p = payload.get("g2p_evidence")
    if not isinstance(g2p, Mapping):
        errors.append("canonical forced provenance has no G2P evidence object")
    else:
        g2p_count = _count(g2p, "item_count")
        if g2p_count is None or g2p_count == 0:
            errors.append("canonical G2P evidence must include a positive item_count")
        if timeline_count is not None and g2p_count != timeline_count:
            errors.append("G2P item_count does not match lyric timeline item_count")
        if g2p.get("oov_items") not in ([], None):
            errors.append("canonical G2P evidence contains OOV items")
        for label, value in (("audio", g2p.get("audio_sha256")), ("timeline", g2p.get("lyrics_timeline_sha256")), ("artifact", g2p.get("artifact_sha256"))):
            if not _sha256(value):
                errors.append(f"canonical G2P evidence has no valid {label} SHA-256")
        if _sha256(timeline_hash) and g2p.get("lyrics_timeline_sha256") != timeline_hash:
            errors.append("G2P evidence is bound to a different lyrics timeline")
        if _sha256(timeline_audio_hash) and g2p.get("audio_sha256") != timeline_audio_hash:
            errors.append("G2P evidence is bound to a different source audio")
        _verify_json_artifact_hash(errors, g2p.get("artifact_path"), g2p.get("artifact_sha256"), label="G2P artifact")

    forced = payload.get("forced_alignment_evidence")
    if not isinstance(forced, Mapping) or not forced.get("provided"):
        errors.append("canonical forced provenance has no forced alignment evidence")
        return
    forced_count = _count(forced, "segment_count")
    lyric_count = _count(forced, "lyric_item_count")
    covered_count = _count(forced, "covered_lyric_item_count")
    if forced_count is None or forced_count == 0:
        errors.append("forced alignment must include a positive segment_count")
    if lyric_count is None or covered_count is None or lyric_count == 0 or covered_count != lyric_count:
        errors.append("forced alignment coverage counts are incomplete")
    if timeline_count is not None and (lyric_count != timeline_count or covered_count != timeline_count):
        errors.append("forced alignment coverage does not match lyric timeline count")
    if not _sha256(stem_hash) or forced.get("vocal_stem_sha256") != stem_hash:
        errors.append("forced alignment is not bound to the declared vocal stem")
    if not _sha256(timeline_hash) or forced.get("lyrics_timeline_sha256") != timeline_hash:
        errors.append("forced alignment is not bound to the declared lyrics timeline")
    if not _sha256(timeline_audio_hash) or forced.get("lyrics_audio_sha256") != timeline_audio_hash:
        errors.append("forced alignment is not bound to the declared source audio")
    if not _sha256(_first_hash(forced, "model_hash")):
        errors.append("forced alignment must include a valid model_hash")
    _verify_json_artifact_hash(errors, forced.get("artifact_path"), forced.get("artifact_sha256"), label="forced alignment artifact")

    phone = payload.get("phone_evidence")
    if isinstance(phone, Mapping) and phone.get("provided"):
        if _count(phone, "segment_count") is None or forced_count is not None and _count(phone, "segment_count") != forced_count:
            errors.append("phone evidence segment_count does not match forced alignment")
        if not _sha256(_first_hash(phone, "sha256", "artifact_sha256")):
            errors.append("canonical phone evidence must include an artifact SHA-256")
        _verify_json_artifact_hash(errors, phone.get("artifact_path"), _first_hash(phone, "artifact_sha256", "sha256"), label="phone artifact")
        phone_audio_hash = _first_hash(phone, "audio_sha256", "vocal_stem_sha256")
        if not _sha256(phone_audio_hash) or (_sha256(stem_hash) and phone_audio_hash != stem_hash):
            errors.append("canonical phone evidence is not bound to the declared vocal stem")
        _verify_onnx_identity(phone, errors, label="phone")

    ctc = payload.get("ctc_evidence")
    if isinstance(ctc, Mapping) and ctc.get("provided"):
        if ctc.get("role") != "auxiliary_only":
            errors.append("CTC evidence must remain auxiliary_only")
        comparison = ctc.get("comparison")
        if not isinstance(comparison, Mapping):
            errors.append("provided CTC evidence must include a forced-alignment comparison")
        else:
            if comparison.get("ctc_is_auxiliary_only") is not True:
                errors.append("CTC/forced comparison must declare CTC auxiliary-only")
            for field in (
                "forced_segment_count",
                "ctc_segment_count",
                "overlap_sec",
                "forced_covered_by_ctc_ratio",
                "ctc_covered_by_forced_ratio",
                "duration_difference_sec",
                "duration_difference_ratio",
                "segment_count_difference",
                "phone_disagreement_count",
            ):
                if field not in comparison:
                    errors.append(f"CTC/forced comparison is missing {field}")
            try:
                coverage = float(comparison.get("forced_covered_by_ctc_ratio"))
                if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
                    errors.append("CTC/forced coverage ratio must be finite and between 0 and 1")
            except (TypeError, ValueError, OverflowError):
                errors.append("CTC/forced coverage ratio must be numeric")
            if forced_count is not None and comparison.get("forced_segment_count") != forced_count:
                errors.append("CTC/forced comparison forced_segment_count does not match forced evidence")
            ctc_count = _count(ctc, "segment_count")
            if ctc_count is not None and comparison.get("ctc_segment_count") != ctc_count:
                errors.append("CTC/forced comparison ctc_segment_count does not match CTC evidence")
        if not _sha256(_first_hash(ctc, "comparison_artifact_sha256")):
            errors.append("provided CTC evidence must include comparison_artifact_sha256")
        _verify_json_artifact_hash(
            errors,
            ctc.get("comparison_artifact_path"),
            ctc.get("comparison_artifact_sha256"),
            label="CTC comparison artifact",
        )
        _verify_onnx_identity(ctc, errors, label="CTC")


def _verify_forced_acceptance(
    forced: Mapping[str, object],
    *,
    stem: Mapping[str, object],
    errors: list[str],
    base_dir: Path | None = None,
) -> None:
    acceptance = forced.get("acceptance")
    if not isinstance(acceptance, Mapping):
        errors.append("forced product has no qualifying forced-phone acceptance evidence")
        return

    route = acceptance.get("route")
    if acceptance.get("schema_version") != ACCEPTANCE_SCHEMA_VERSION:
        errors.append("forced-phone acceptance schema is invalid")
    if route not in {QUALIFIED_MODEL_ROUTE, HUMAN_REVIEW_ROUTE}:
        errors.append("forced-phone acceptance route is invalid")

    stem_hash = str(stem.get("sha256") or "").lower()
    stem_path = _resolve_provenance_path(stem.get("path"), base_dir=base_dir)
    acceptance_path = _resolve_provenance_path(
        acceptance.get("acceptance_artifact_path"),
        base_dir=base_dir,
    )
    source_alignment_path = _resolve_provenance_path(forced.get("path"), base_dir=base_dir)
    forced_artifact_path = _resolve_provenance_path(forced.get("artifact_path"), base_dir=base_dir)

    if acceptance_path is None:
        errors.append("forced-phone acceptance artifact path is missing")
    else:
        _verify_required_hash_bound_artifact(
            errors,
            acceptance_path,
            acceptance.get("acceptance_artifact_sha256"),
            label="forced-phone acceptance artifact",
        )

    if source_alignment_path is None:
        errors.append("forced alignment source artifact path is missing")
    else:
        _verify_required_hash_bound_artifact(
            errors,
            source_alignment_path,
            forced.get("source_artifact_sha256"),
            label="forced alignment source artifact",
        )

    if stem_path is None or not _sha256(stem_hash):
        errors.append("forced product provenance has no hash-bound vocal stem")
    elif not stem_path.is_file():
        errors.append(f"forced vocal stem path does not exist: {stem_path}")

    segments = _load_forced_alignment_segments(
        forced_artifact_path,
        expected_stem_hash=stem_hash,
        declared_hash=forced.get("artifact_sha256"),
        errors=errors,
    )
    if acceptance_path is None or source_alignment_path is None or stem_path is None or segments is None:
        return

    try:
        load_forced_phone_acceptance(
            acceptance_path,
            alignment_path=source_alignment_path,
            vocal_stem=stem_path,
            model_hash=str(_first_hash(forced, "model_hash") or ""),
            segments=segments,
        )
    except ForcedPhoneAcceptanceError as exc:
        errors.append(f"forced-phone acceptance loader rejected evidence: {exc}")
    except (OSError, UnicodeError, ValueError) as exc:
        errors.append(f"forced-phone acceptance loader could not read evidence: {exc}")
    _verify_acceptance_inventory(acceptance, errors)
    if route == QUALIFIED_MODEL_ROUTE:
        _verify_qualified_model_acceptance(
            forced,
            acceptance,
            errors,
            base_dir=acceptance_path.parent,
        )
    if route == HUMAN_REVIEW_ROUTE:
        _verify_human_review_acceptance(acceptance, stem=stem, errors=errors)


class _ArtifactSegment:
    """Small adapter preserving the exact PhoneSegment JSON contract for the loader."""

    __slots__ = ("_payload", "normalized_ipa", "time_range")

    def __init__(self, payload: Mapping[str, object], *, start: float, end: float) -> None:
        self._payload = dict(payload)
        self.normalized_ipa = str(payload.get("normalized_ipa") or "").strip()
        self.time_range = TimeRange(start, end)

    def to_dict(self) -> dict[str, object]:
        return dict(self._payload)


def _resolve_provenance_path(value: object, *, base_dir: Path | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value))
    if not path.is_absolute():
        direct_path = path.resolve()
        if base_dir is None or direct_path.exists():
            return direct_path
        path = base_dir / path
    return path.resolve()


def _verify_required_hash_bound_artifact(
    errors: list[str], path: Path, digest: object, *, label: str
) -> None:
    if not path.is_file():
        errors.append(f"{label} path does not exist: {path}")
        return
    if not _sha256(digest):
        errors.append(f"{label} must include a valid SHA-256")
        return
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        errors.append(f"{label} cannot be read: {exc}")
        return
    if actual != str(digest).lower():
        errors.append(f"{label} SHA-256 does not match its path")


def _load_forced_alignment_segments(
    path: Path | None,
    *,
    expected_stem_hash: str,
    declared_hash: object,
    errors: list[str],
) -> list[_ArtifactSegment] | None:
    if path is None:
        errors.append("forced alignment artifact path is missing")
        return None
    if not path.is_file():
        errors.append(f"forced alignment artifact path does not exist: {path}")
        return None
    if not _sha256(declared_hash):
        errors.append("forced alignment artifact must include a valid SHA-256")
        return None
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        errors.append(f"forced alignment artifact cannot be read: {exc}")
        return None
    if hashlib.sha256(raw_bytes).hexdigest() != str(declared_hash).lower():
        errors.append("forced alignment artifact SHA-256 does not match its path")
        return None
    try:
        payload = json.loads(raw_bytes.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"forced alignment artifact is invalid JSON: {exc}")
        return None
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "forced_phone_alignment_v1":
        errors.append("forced alignment artifact schema is invalid")
        return None
    if payload.get("vocal_stem_sha256") != expected_stem_hash:
        errors.append("forced alignment artifact is bound to a different vocal stem")
        return None
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        errors.append("forced alignment artifact has no phone segment list")
        return None

    result: list[_ArtifactSegment] = []
    previous_end = -1.0
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, Mapping):
            errors.append(f"forced alignment segment {index + 1} must be an object")
            continue
        start = _finite_segment_value(raw_segment.get("start_sec"))
        end = _finite_segment_value(raw_segment.get("end_sec"))
        normalized_ipa = str(raw_segment.get("normalized_ipa") or "").strip()
        candidates = raw_segment.get("candidate_phones")
        confidence = _finite_segment_value(raw_segment.get("confidence"))
        if (
            start is None
            or end is None
            or start < 0.0
            or end <= start
            or start < previous_end
            or not normalized_ipa
            or confidence is None
            or not 0.0 <= confidence <= 1.0
            or not isinstance(candidates, list)
            or not candidates
            or not isinstance(raw_segment.get("metadata"), Mapping)
        ):
            errors.append(f"forced alignment segment {index + 1} has invalid phone data")
            continue
        candidate_invalid = False
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or not str(candidate.get("symbol") or "").strip():
                candidate_invalid = True
                break
            candidate_confidence = _finite_segment_value(candidate.get("confidence"))
            if candidate_confidence is None or not 0.0 <= candidate_confidence <= 1.0:
                candidate_invalid = True
                break
        if candidate_invalid:
            errors.append(f"forced alignment segment {index + 1} has invalid phone candidates")
            continue
        canonical = {key: value for key, value in raw_segment.items() if key != "reviewed"}
        try:
            result.append(_ArtifactSegment(canonical, start=start, end=end))
        except ValueError:
            errors.append(f"forced alignment segment {index + 1} has invalid timing")
            continue
        previous_end = end
    return result if result and len(result) == len(raw_segments) else None


def _finite_segment_value(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _verify_acceptance_inventory(acceptance: Mapping[str, object], errors: list[str]) -> None:
    inventory = acceptance.get("phone_inventory")
    if not isinstance(inventory, Mapping):
        errors.append("forced-phone acceptance has no phone inventory")
        return
    symbols = inventory.get("symbols")
    if not isinstance(symbols, list) or not symbols or any(not isinstance(value, str) or not value for value in symbols):
        errors.append("forced-phone acceptance phone inventory is invalid")
        return
    if len(set(symbols)) != len(symbols):
        errors.append("forced-phone acceptance phone inventory contains duplicates")
    if inventory.get("sha256") != canonical_phone_inventory_sha256(symbols):
        errors.append("forced-phone acceptance phone inventory hash is invalid")
    if not str(inventory.get("source") or "").strip():
        errors.append("forced-phone acceptance phone inventory has no source")


def _verify_accepted_phone_segments(
    forced: Mapping[str, object],
    acceptance: Mapping[str, object],
    errors: list[str],
) -> None:
    raw_path = forced.get("artifact_path")
    if not raw_path:
        errors.append("forced alignment artifact path is missing")
        return
    try:
        payload = json.loads(Path(str(raw_path)).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        errors.append("forced alignment artifact cannot be read")
        return
    segments = payload.get("segments") if isinstance(payload, Mapping) else None
    if not isinstance(segments, list):
        errors.append("forced alignment artifact has no phone segment list")
        return
    normalized_segments = [
        {key: value for key, value in item.items() if key != "reviewed"}
        if isinstance(item, Mapping)
        else item
        for item in segments
    ]
    encoded = json.dumps(
        normalized_segments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != acceptance.get("phone_segments_sha256"):
        errors.append("forced-phone acceptance is bound to different phone boundaries")


def _verify_qualified_model_acceptance(
    forced: Mapping[str, object],
    acceptance: Mapping[str, object],
    errors: list[str],
    *,
    base_dir: Path,
) -> None:
    model = acceptance.get("model")
    if not isinstance(model, Mapping):
        errors.append("qualified model acceptance has no model evidence")
        return
    if model.get("sha256") != _first_hash(forced, "model_hash"):
        errors.append("qualified model acceptance has a conflicting model SHA-256")
    if not str(model.get("revision") or "").strip() or model.get("revision_is_immutable") is not True:
        errors.append("qualified model acceptance has no immutable revision")
    license_evidence = model.get("license")
    if (
        not isinstance(license_evidence, Mapping)
        or license_evidence.get("noncommercial_restriction") is not False
        or license_evidence.get("commercial_use") != "allowed"
        or license_evidence.get("applies_to") != "model_weights"
        or not str(license_evidence.get("identifier") or "").strip()
        or not str(license_evidence.get("source") or "").strip()
        or not _sha256(license_evidence.get("source_sha256"))
    ):
        errors.append("qualified model acceptance has no explicit non-NC weights license")
    training = model.get("cantonese_singing_training")
    if (
        not isinstance(training, Mapping)
        or training.get("claim") != "trained_or_finetuned_on_cantonese_singing"
        or training.get("domain") != "singing_voice"
        or training.get("source_type") != "primary"
        or not training.get("datasets")
        or not str(training.get("source") or "").strip()
        or not str(training.get("source_revision") or "").strip()
        or not _sha256(training.get("source_sha256"))
    ):
        errors.append("qualified model acceptance has no primary Cantonese singing training evidence")
    manifest_path = _resolve_provenance_path(
        model.get("resource_manifest_path"),
        base_dir=base_dir,
    )
    if manifest_path is None:
        errors.append("qualified model acceptance has no resource manifest path")
        return
    _verify_required_hash_bound_artifact(
        errors,
        manifest_path,
        model.get("resource_manifest_sha256"),
        label="qualified model resource manifest",
    )
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        errors.append("qualified model resource manifest cannot be read")
        return
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != RESOURCE_MANIFEST_SCHEMA_VERSION:
        errors.append("qualified model resource manifest schema is invalid")
        return
    required = manifest.get("required_artifacts")
    artifacts = manifest.get("artifacts")
    if not isinstance(required, list) or not isinstance(artifacts, Mapping):
        errors.append("qualified model resource manifest has no artifact list")
        return
    for kind in required:
        if not isinstance(kind, str) or not isinstance(artifacts.get(kind), Mapping):
            errors.append(f"qualified model resource manifest is missing artifact: {kind}")
            continue
        artifact = artifacts[kind]
        artifact_path = _resolve_provenance_path(artifact.get("path"), base_dir=manifest_path.parent)
        _verify_required_hash_bound_artifact(
            errors,
            artifact_path if artifact_path is not None else manifest_path.parent / "__missing__",
            artifact.get("sha256"),
            label=f"qualified model {kind} artifact",
        )
        if artifact.get("revision_is_immutable") is not True or not str(artifact.get("revision") or "").strip():
            errors.append(f"qualified model {kind} artifact has no immutable revision")
        license_evidence = artifact.get("license")
        if (
            not isinstance(license_evidence, Mapping)
            or license_evidence.get("noncommercial_restriction") is not False
            or license_evidence.get("commercial_use") != "allowed"
            or license_evidence.get("applies_to") != kind
        ):
            errors.append(f"qualified model {kind} artifact has no explicit non-NC license")
        elif artifact_path is not None:
            license_path = _resolve_provenance_path(
                license_evidence.get("source_path"),
                base_dir=manifest_path.parent,
            )
            _verify_required_hash_bound_artifact(
                errors,
                license_path if license_path is not None else manifest_path.parent / "__missing-license__",
                license_evidence.get("source_sha256"),
                label=f"qualified model {kind} license source",
            )
    inventory = manifest.get("phone_inventory")
    if not isinstance(inventory, Mapping) or not isinstance(inventory.get("symbols"), list):
        errors.append("qualified model resource manifest has no phone inventory")
    else:
        _verify_acceptance_inventory({"phone_inventory": inventory}, errors)
    training_evidence = manifest.get("training_evidence")
    if not isinstance(training_evidence, Mapping):
        errors.append("qualified model resource manifest has no training evidence")
    else:
        training_path = _resolve_provenance_path(training_evidence.get("path"), base_dir=manifest_path.parent)
        _verify_required_hash_bound_artifact(
            errors,
            training_path if training_path is not None else manifest_path.parent / "__missing-training__",
            training_evidence.get("sha256"),
            label="qualified model training evidence",
        )


def _verify_human_review_acceptance(
    acceptance: Mapping[str, object],
    *,
    stem: Mapping[str, object],
    errors: list[str],
) -> None:
    review = acceptance.get("review")
    if (
        not isinstance(review, Mapping)
        or review.get("review_status") != "approved"
        or review.get("reviewer_kind") != "human"
        or review.get("phone_boundaries_reviewed") is not True
        or not str(review.get("reviewer_id") or "").strip()
        or not str(review.get("reviewed_at") or "").strip()
        or not str(review.get("review_method") or "").strip()
        or review.get("scope") not in {SHORT_CLIP_SCOPE, FULL_SONG_SCOPE}
    ):
        errors.append("human forced-phone acceptance has no approved phone-boundary review")
        return
    acceptance_path = Path(str(acceptance.get("acceptance_artifact_path") or ""))
    try:
        artifact = json.loads(acceptance_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        errors.append("human forced-phone acceptance artifact cannot be read")
        return
    if not isinstance(artifact, Mapping):
        errors.append("human forced-phone acceptance artifact must be an object")
        return
    for field in (
        "schema_version",
        "route",
        "alignment_artifact_sha256",
        "vocal_stem_sha256",
        "phone_segments_sha256",
    ):
        if artifact.get(field) != acceptance.get(field):
            errors.append(f"human forced-phone acceptance artifact conflicts on {field}")
    artifact_review = artifact.get("review")
    if not isinstance(artifact_review, Mapping):
        errors.append("human forced-phone acceptance artifact has no review object")
        return
    raw_alignment_path = artifact.get("alignment_artifact_path")
    if not raw_alignment_path:
        errors.append("human acceptance artifact has no alignment_artifact_path")
    else:
        artifact_alignment_path = Path(str(raw_alignment_path))
        if not artifact_alignment_path.is_absolute():
            artifact_alignment_path = acceptance_path.parent / artifact_alignment_path
        if (
            not artifact_alignment_path.is_file()
            or hashlib.sha256(artifact_alignment_path.read_bytes()).hexdigest()
            != artifact.get("alignment_artifact_sha256")
        ):
            errors.append("human acceptance alignment artifact is missing or has the wrong SHA-256")
        else:
            _verify_human_acceptance_with_loader(
                acceptance,
                stem=stem,
                alignment_path=artifact_alignment_path,
                errors=errors,
            )
    bounds = _verified_review_range(artifact_review.get("reviewed_range_sec"), errors)
    if bounds is None:
        return
    start, end = bounds
    scope = artifact_review.get("scope")
    if scope != review.get("scope"):
        errors.append("human forced-phone acceptance artifact scope conflicts with provenance")
    if scope == SHORT_CLIP_SCOPE:
        if not 10.0 <= end - start <= 20.0:
            errors.append("initial human-reviewed clip must be 10-20 seconds")
        return
    if scope != FULL_SONG_SCOPE:
        return
    duration = _verified_wav_duration(stem.get("path"), errors)
    if duration is not None and (start > 0.05 or end < duration - 0.05):
        errors.append("full-song human review range does not cover the complete vocal stem")
    _verify_short_clip_predecessor(
        artifact_review.get("short_clip_predecessor"),
        acceptance_path=acceptance_path,
        stem_hash=stem.get("sha256"),
        errors=errors,
    )


def _verify_human_acceptance_with_loader(
    acceptance: Mapping[str, object],
    *,
    stem: Mapping[str, object],
    alignment_path: Path,
    errors: list[str],
) -> None:
    """Apply the product acceptance schema to a direct human-review check."""

    acceptance_path = _resolve_provenance_path(
        acceptance.get("acceptance_artifact_path"),
        base_dir=alignment_path.parent,
    )
    stem_path = _resolve_provenance_path(stem.get("path"), base_dir=None)
    if acceptance_path is None or stem_path is None:
        return
    segments = _load_forced_alignment_segments(
        alignment_path,
        expected_stem_hash=str(stem.get("sha256") or "").lower(),
        declared_hash=acceptance.get("alignment_artifact_sha256"),
        errors=errors,
    )
    if segments is None:
        return
    try:
        load_forced_phone_acceptance(
            acceptance_path,
            alignment_path=alignment_path,
            vocal_stem=stem_path,
            model_hash="",
            segments=segments,
        )
    except ForcedPhoneAcceptanceError as exc:
        errors.append(f"human forced-phone acceptance loader rejected evidence: {exc}")
    except (OSError, UnicodeError, ValueError) as exc:
        errors.append(f"human forced-phone acceptance loader could not read evidence: {exc}")


def _verified_review_range(raw: object, errors: list[str]) -> tuple[float, float] | None:
    if not isinstance(raw, list) or len(raw) != 2:
        errors.append("human review range must contain start and end")
        return None
    try:
        start, end = float(raw[0]), float(raw[1])
    except (TypeError, ValueError, OverflowError):
        errors.append("human review range must be numeric")
        return None
    if not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end <= start:
        errors.append("human review range is invalid")
        return None
    return start, end


def _verified_wav_duration(raw_path: object, errors: list[str]) -> float | None:
    if not raw_path:
        errors.append("full-song human review has no vocal-stem path")
        return None
    try:
        with wave.open(str(Path(str(raw_path))), "rb") as handle:
            rate = handle.getframerate()
            if rate <= 0:
                raise wave.Error("invalid sample rate")
            return handle.getnframes() / rate
    except (EOFError, OSError, wave.Error):
        errors.append("full-song human review vocal stem is not a readable PCM WAV")
        return None


def _verify_short_clip_predecessor(
    raw: object,
    *,
    acceptance_path: Path,
    stem_hash: object,
    errors: list[str],
) -> None:
    if not isinstance(raw, Mapping):
        errors.append("full-song human review has no short-clip predecessor")
        return
    predecessor_path = Path(str(raw.get("acceptance_path") or ""))
    if not predecessor_path.is_absolute():
        predecessor_path = acceptance_path.parent / predecessor_path
    expected_hash = raw.get("acceptance_sha256")
    if not predecessor_path.is_file() or not _sha256(expected_hash):
        errors.append("full-song short-clip predecessor is missing")
        return
    digest = hashlib.sha256(predecessor_path.read_bytes()).hexdigest()
    if digest != expected_hash:
        errors.append("full-song short-clip predecessor SHA-256 does not match")
        return
    try:
        predecessor = json.loads(predecessor_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        errors.append("full-song short-clip predecessor is invalid JSON")
        return
    predecessor_review = predecessor.get("review") if isinstance(predecessor, Mapping) else None
    if (
        not isinstance(predecessor, Mapping)
        or predecessor.get("schema_version") != ACCEPTANCE_SCHEMA_VERSION
        or predecessor.get("route") != HUMAN_REVIEW_ROUTE
        or predecessor.get("vocal_stem_sha256") != stem_hash
        or not isinstance(predecessor_review, Mapping)
        or predecessor_review.get("review_status") != "approved"
        or predecessor_review.get("reviewer_kind") != "human"
        or predecessor_review.get("phone_boundaries_reviewed") is not True
        or predecessor_review.get("scope") != SHORT_CLIP_SCOPE
    ):
        errors.append("full-song predecessor is not an approved short-clip review for this stem")
        return
    bounds = _verified_review_range(predecessor_review.get("reviewed_range_sec"), errors)
    if bounds is not None and not 10.0 <= bounds[1] - bounds[0] <= 20.0:
        errors.append("full-song predecessor review is not a 10-20 second clip")
    raw_alignment_path = predecessor.get("alignment_artifact_path")
    alignment_path = Path(str(raw_alignment_path or ""))
    if not alignment_path.is_absolute():
        alignment_path = predecessor_path.parent / alignment_path
    if (
        not raw_alignment_path
        or not alignment_path.is_file()
        or hashlib.sha256(alignment_path.read_bytes()).hexdigest()
        != predecessor.get("alignment_artifact_sha256")
    ):
        errors.append("full-song predecessor alignment artifact is missing or has the wrong SHA-256")
        return
    try:
        alignment = json.loads(alignment_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        errors.append("full-song predecessor alignment artifact is invalid JSON")
        return
    raw_segments = alignment.get("segments") if isinstance(alignment, Mapping) else None
    if not isinstance(raw_segments, list) or not raw_segments:
        errors.append("full-song predecessor alignment has no phone segments")
        return
    normalized_segments = [
        {key: value for key, value in item.items() if key != "reviewed"}
        if isinstance(item, Mapping)
        else item
        for item in raw_segments
    ]
    encoded = json.dumps(
        normalized_segments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != predecessor.get("phone_segments_sha256"):
        errors.append("full-song predecessor alignment has different phone boundaries")


def _verify_fusion_coverage(fusion: Mapping[str, object], errors: list[str]) -> None:
    coverage = fusion.get("coverage")
    if coverage is not None and not isinstance(coverage, Mapping):
        errors.append("fusion coverage must be an object")
        return
    if not isinstance(coverage, Mapping):
        return
    for field in (
        "pitch_coverage_ratio",
        "phone_group_coverage_ratio",
        "phone_segment_coverage_ratio",
        "lyric_coverage_ratio",
    ):
        if field not in coverage:
            continue
        try:
            value = float(coverage[field])
        except (TypeError, ValueError, OverflowError):
            errors.append(f"fusion coverage {field} must be numeric")
            continue
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            errors.append(f"fusion coverage {field} must be finite and between 0 and 1")
    if "monotonic" in coverage and coverage.get("monotonic") is not True:
        errors.append("fusion alignment must be monotonic")
    if "monotonic_alignment" in fusion and fusion.get("monotonic_alignment") is not True:
        errors.append("fusion monotonic_alignment must be true")


def _verify_onnx_identity(value: Mapping[str, object], errors: list[str], *, label: str) -> None:
    markers = " ".join(
        str(value.get(field) or "")
        for field in ("source", "backend_kind", "adapter_kind", "model_backend", "real_backend")
    ).lower()
    if "onnx" not in markers:
        return
    if not _sha256(_first_hash(value, "model_sha256", "onnx_model_sha256")):
        errors.append(f"ONNX {label} evidence must include model_sha256")
    if not _sha256(_first_hash(value, "vocab_sha256", "onnx_vocab_sha256")):
        errors.append(f"ONNX {label} evidence must include vocab_sha256")


def _sha256(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _verify_json_artifact_hash(
    errors: list[str], path_value: object, digest_value: object, *, label: str
) -> None:
    if not path_value or not _sha256(digest_value):
        return
    _verify_file_hash(errors, {"path": path_value, "sha256": digest_value}, label=label)


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_stdio()
    args = build_parser().parse_args(argv)
    vsqx_report = verify_vsqx(args.vsqx, expected_smoke=args.expected_smoke)
    summary_report = verify_summary(args.summary)
    provenance_report = verify_product_provenance(args.product_provenance)
    errors = list(vsqx_report.get("errors", []))
    if summary_report is not None:
        errors.extend(summary_report.get("errors", []))
    if provenance_report is not None:
        errors.extend(provenance_report.get("errors", []))
    report = {
        "schema_version": "miku_v4c_vsqx_smoke_verification_v1",
        "ok": not errors,
        "errors": errors,
        "vsqx": vsqx_report,
        "summary": summary_report,
        "product_provenance": provenance_report,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
