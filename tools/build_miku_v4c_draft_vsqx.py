"""Build a conservative Miku V4C draft and template-cloned VSQX."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence
from xml.etree import ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from any2xsampa.cantonese import CantonesePhonemeResolver
from any2xsampa.core.models import PhoneSegment
from any2xsampa.vocaloid import VocalTrackDraft, collect_miku_v4c_policy_issues, export_vsqx_from_template
from any2xsampa.vocaloid.evidence_fusion import EvidenceFusionError, build_pitch_skeleton_payload
from tools.build_vocal_track_draft import build_draft_payload

SUMMARY_SCHEMA_VERSION = "miku_v4c_draft_vsqx_run_v1"


def _failure(kind: str, errors: list[str], artifacts: dict[str, str]) -> tuple[int, dict[str, Any]]:
    return 2, {"schema_version": SUMMARY_SCHEMA_VERSION, "ok": False, "error_kind": kind, "errors": errors, "artifacts": artifacts}


def _validate_miku_template(path: Path) -> None:
    root = ET.parse(path).getroot()
    namespace = root.tag[1:].split("}", 1)[0] if root.tag.startswith("{") else ""
    prefix = f"{{{namespace}}}" if namespace else ""
    names = [
        (node.text or "").strip()
        for node in root.findall(f".//{prefix}vVoice/{prefix}name")
    ]
    if not names or not any("miku" in name.casefold() for name in names):
        raise ValueError("VSQX template does not contain a Miku-compatible voice")


def _finite_number(value: object, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _midi(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer MIDI note")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer MIDI note") from exc
    if not 0 <= result <= 127:
        raise ValueError(f"{field} must be between 0 and 127")
    return result


def _pitch_rows(path: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path is None:
        return [], {"provided": False, "note_candidate_count": 0}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "pitch_note_analysis_v1":
        raise ValueError("pitch analysis schema must be pitch_note_analysis_v1")
    raw_rows = payload.get("note_candidates")
    if not isinstance(raw_rows, list):
        raise ValueError("pitch analysis note_candidates must be a list")
    rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, dict):
            raise ValueError(f"pitch analysis note_candidates[{index}] must be an object")
        row = dict(raw_row)
        row["midi"] = _midi(row.get("midi"), field=f"note_candidates[{index}].midi")
        row["start_sec"] = _finite_number(
            row.get("start_sec"), field=f"note_candidates[{index}].start_sec"
        )
        if row["start_sec"] < 0:
            raise ValueError(f"note_candidates[{index}].start_sec must be non-negative")
        row["duration_sec"] = _finite_number(
            row.get("duration_sec"),
            field=f"note_candidates[{index}].duration_sec",
            positive=True,
        )
        rows.append(row)
    tempo_bpm = _finite_number(payload.get("tempo_bpm", 120.0), field="tempo_bpm", positive=True)
    return rows, {
        "provided": True,
        "path": str(path),
        "schema_version": payload.get("schema_version"),
        "model_output": bool(payload.get("model_output", False)),
        "analysis_output": bool(payload.get("analysis_output", False)),
        "note_candidate_count": len(rows),
        "tempo_bpm": tempo_bpm,
    }


def run_draft_vsqx(
    *,
    lyrics: str,
    template: Path,
    output_root: Path,
    jyutping_override: str | None = None,
    pitch_analysis: Path | None = None,
    pitches: list[int] | None = None,
    tempo: float = 128.0,
    prefix: str = "miku_v4c_draft",
    pretty: bool = False,
    render_pitch_safety: bool = True,
    render_pitch_floor: int = 64,
    phone_segments: Sequence[PhoneSegment] | None = None,
    lyric_source: str = "user_lyrics",
    require_evidence_gate: bool = False,
) -> tuple[int, dict[str, Any]]:
    output_root = Path(output_root)
    draft_path = output_root / f"{prefix}.json"
    vsqx_path = output_root / f"{prefix}.vsqx"
    summary_path = output_root / f"{prefix}_summary.json"
    artifacts = {
        "draft": str(draft_path),
        "vsqx": str(vsqx_path),
        "summary": str(summary_path),
        "product_summary": str(summary_path),
    }
    if not Path(template).is_file():
        return _failure("missing_template", [f"missing VSQX template: {template}"], artifacts)
    try:
        _validate_miku_template(Path(template))
    except ET.ParseError as exc:
        return _failure("invalid_template_xml", [str(exc)], artifacts)
    except (OSError, ValueError) as exc:
        return _failure("incompatible_template_singer", [str(exc)], artifacts)
    if not str(lyrics).strip():
        return _failure("invalid_arguments", ["lyrics must be non-empty"], artifacts)
    if not 0 <= int(render_pitch_floor) <= 127:
        return _failure("invalid_arguments", ["--render-pitch-floor must be between 0 and 127"], artifacts)
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        rows, pitch_summary = _pitch_rows(Path(pitch_analysis) if pitch_analysis else None)
        pitch_values = (
            [_midi(value, field=f"pitches[{index}]") for index, value in enumerate(pitches)]
            if pitches
            else [int(row["midi"]) for row in rows]
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return _failure("invalid_pitch_evidence", [str(exc)], artifacts)
    try:
        payload = build_draft_payload(
            lyrics,
            jyutping_override=jyutping_override,
            tempo=tempo,
            pitches=pitch_values or None,
            resolver=CantonesePhonemeResolver(),
        )
    except (TypeError, ValueError) as exc:
        return _failure("unsupported_lyrics_evidence", [str(exc)], artifacts)
    fusion_summary: dict[str, Any] | None = None
    if require_evidence_gate and not rows:
        return _failure("missing_pitch_evidence", ["product evidence fusion requires pitch note candidates"], artifacts)
    if require_evidence_gate and not phone_segments:
        return _failure("missing_phone_evidence", ["product evidence fusion requires phone segments"], artifacts)
    if phone_segments:
        try:
            effective_tempo = float(pitch_summary.get("tempo_bpm", tempo))
            payload, fusion_summary = build_pitch_skeleton_payload(
                base_payload=payload,
                pitch_rows=rows,
                phone_segments=phone_segments,
                lyric_source=lyric_source,
                tempo_bpm=effective_tempo,
            )
        except EvidenceFusionError as exc:
            return _failure(exc.error_kind, [str(exc)], artifacts)
        except (TypeError, ValueError) as exc:
            return _failure("invalid_evidence_fusion", [str(exc)], artifacts)
    notes = payload["tracks"][0]["notes"]
    if rows and not phone_segments:
        extracted = [int(row.get("midi", 64)) for row in rows]
        transpose = max(0, int(render_pitch_floor) - min(extracted)) if render_pitch_safety and extracted else 0
        if any(value + transpose > 127 for value in extracted):
            return _failure(
                "invalid_pitch_evidence",
                ["pitch safety transposition would exceed MIDI note 127"],
                artifacts,
            )
        ticks_per_quarter = int(payload.get("ticks_per_quarter", 480))
        effective_tempo = float(pitch_summary.get("tempo_bpm", tempo))
        seconds_to_ticks = ticks_per_quarter * effective_tempo / 60.0
        for index, note in enumerate(notes):
            if index >= len(rows):
                break
            row = rows[index]
            raw_midi = int(row.get("midi", note["pitch"]))
            note["pitch"] = raw_midi + transpose
            note["start_tick"] = round(float(row.get("start_sec", 0.0)) * seconds_to_ticks)
            note["duration_ticks"] = max(1, round(float(row.get("duration_sec", 0.5)) * seconds_to_ticks))
            note["provenance"].update({
                "extracted_midi": raw_midi,
                "render_midi": note["pitch"],
                "pitch_transpose_semitones": transpose,
                "pitch_safety_applied": bool(render_pitch_safety),
            })
        payload["provenance"]["pitch_render_policy"] = {
            "enabled": bool(render_pitch_safety), "floor_midi": int(render_pitch_floor), "transpose_semitones": transpose,
        }
        pitch_summary["render_pitch_policy"] = {"floor_midi": int(render_pitch_floor), "floor_note_name": "E4" if int(render_pitch_floor) == 64 else None, "transpose_semitones": transpose}
    issues = collect_miku_v4c_policy_issues((str(note.get("id")), list(note.get("phonemes", []))) for note in notes)
    if issues:
        result = {"schema_version": SUMMARY_SCHEMA_VERSION, "ok": False, "error_kind": "phoneme_policy_failure", "errors": ["known silent or unsafe phoneme sequence"], "phoneme_policy_issues": issues, "artifacts": artifacts}
        summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 1, result
    try:
        draft = VocalTrackDraft.from_mapping(payload)
        draft_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None) + "\n", encoding="utf-8")
        export_vsqx_from_template(draft, vsqx_path, template_path=template)
    except ET.ParseError as exc:
        return _failure("invalid_template_xml", [str(exc)], artifacts)
    except (OSError, TypeError, ValueError) as exc:
        return _failure("vsqx_export_failure", [str(exc)], artifacts)
    export_summary = {"export_mode": "template_clone_import_safe", "template": str(template), "output": str(vsqx_path), "note_count": len(notes)}
    result = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "ok": True,
        "error_kind": None,
        "errors": [],
        "product_stage": "draft_candidate",
        "model_output": False,
        "training_data": False,
        "user_jyutping_required": False,
        "manual_vocaloid_import_required": True,
        "phoneme_empirical_review_required": True,
        "xsampa_role": "export_only_not_internal_truth",
        "pitch_evidence": pitch_summary,
        "evidence_fusion": fusion_summary,
        "export_summary": export_summary,
        "artifacts": artifacts,
    }
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lyrics", required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jyutping-override")
    parser.add_argument("--pitch-analysis", type=Path)
    parser.add_argument("--pitches")
    parser.add_argument("--prefix", default="miku_v4c_draft")
    args = parser.parse_args(argv)
    pitches = [int(value) for value in args.pitches.split(",")] if args.pitches else None
    rc, result = run_draft_vsqx(lyrics=args.lyrics, template=args.template, output_root=args.output_root, jyutping_override=args.jyutping_override, pitch_analysis=args.pitch_analysis, pitches=pitches, prefix=args.prefix)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
