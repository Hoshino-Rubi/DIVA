from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import wave

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from any2xsampa.data.label_provenance import (  # noqa: E402
    human_phone_truth,
    optional_sha256,
    stable_identities,
)


SCHEMA_VERSION = "cantonese_singing_phrase_ctc_dataset_v2"
MODEL_SAMPLE_RATE = 16_000
MAX_WINDOW_SEC = 8.0
MAX_GAP_SEC = 0.5
CONTEXT_SEC = 0.15


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wav(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".wav" or not path.is_file():
        raise ValueError(f"phrase CTC audio must be an existing WAV: {path}")
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        frames = handle.getnframes()
    if sample_rate <= 0 or channels <= 0 or frames <= 0:
        raise ValueError(f"invalid WAV metadata: {path}")
    return {
        "path": str(path),
        "sha256": _sha_file(path),
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frames,
        "duration_sec": frames / sample_rate,
    }


def _audio_path(row: Mapping[str, Any]) -> Path:
    values = [Path(str(value)) for value in row.get("audio_paths", []) if str(value).strip()]
    wavs = [path for path in values if path.suffix.lower() == ".wav"]
    if len(wavs) > 1:
        raise ValueError("QRE phrase rows must resolve to one latest WAV, not multiple candidates")
    if len(wavs) != 1:
        raise ValueError("phrase CTC rows require exactly one WAV audio source")
    return wavs[0]


def _ready(row: Mapping[str, Any]) -> bool:
    target = [str(value).strip() for value in row.get("target_phone_tokens", []) if str(value).strip()]
    return row.get("training_ready") is True and bool(target)


def _stream_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[list[Mapping[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    streams: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    breaks: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: (float(value.get("start_sec") or 0.0), int(value.get("note_index") or 0))):
        note_id = f"{row.get('segment_id')}-n{int(row.get('note_index') or 0):03d}"
        if not _ready(row):
            if current:
                streams.append(current)
                current = []
            detail = {
                "note_id": note_id,
                "reason": "unresolved_note_break",
                "human_v4c_tokens_by_note": [[str(value) for value in row.get("v4c_tokens", [])]],
                "manual_annotation_rejected": False,
            }
            exclusions.append(detail)
            breaks.append(dict(detail))
            continue
        if current:
            gap = float(row.get("start_sec") or 0.0) - float(current[-1].get("end_sec") or 0.0)
            if gap > MAX_GAP_SEC:
                streams.append(current)
                current = []
                breaks.append({"note_id": note_id, "reason": "gap_exceeds_max", "gap_sec": round(gap, 6)})
        current.append(row)
    if current:
        streams.append(current)
    return streams, exclusions, breaks


def _windows(stream: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    if not stream:
        return []
    result: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for row in stream:
        proposed = [*current, row]
        span = float(proposed[-1]["end_sec"]) - float(proposed[0]["start_sec"])
        if current and span > MAX_WINDOW_SEC:
            result.append(current)
            overlap = [current[-1]]
            overlap_span = float(overlap[-1]["end_sec"]) - float(overlap[0]["start_sec"])
            prior_span = float(current[-1]["end_sec"]) - float(current[0]["start_sec"])
            current = overlap if overlap_span / max(prior_span, 1e-9) <= 0.5 + 1e-9 else []
        current.append(row)
    if current:
        result.append(current)
    return result


def _example(rows: Sequence[Mapping[str, Any]], *, index: int, audio: Path, probe: Mapping[str, Any]) -> dict[str, Any]:
    first = rows[0]
    truths = [human_phone_truth(row) for row in rows]
    identities = stable_identities(first)
    if any(stable_identities(row) != identities for row in rows[1:]):
        raise ValueError("phrase window cannot mix singer/song/phrase identities")
    segment_id = str(first.get("segment_id") or "segment")
    note_start = float(rows[0]["start_sec"])
    note_end = float(rows[-1]["end_sec"])
    crop_start = max(0.0, note_start - CONTEXT_SEC)
    crop_end = min(float(probe["duration_sec"]), note_end + CONTEXT_SEC)
    targets = [token for truth in truths for token in truth["target_phone_tokens"]]
    ipa = [token for truth in truths for token in truth["internal_ipa_tokens"]]
    features = [feature for truth in truths for feature in truth["articulatory_features"]]
    note_provenance = [truth["label_provenance"] for truth in truths]
    aggregate_provenance = {
        "schema_version": "human_phone_label_v1",
        "kind": "human",
        "source_id": f"phrase:{first.get('singer_id')}:{first.get('song_id')}:{first.get('phrase_id')}",
        "source_revision": "note-provenance-aggregate-v1",
        "annotation_sha256": _sha_json(note_provenance),
        "policy_version": "phrase-note-provenance-aggregate-v1",
        "annotator_id": "approved-note-annotations",
        "review_status": "approved",
    }
    human = [[str(token) for token in row.get("v4c_tokens", [])] for row in rows]
    note_ids = [f"{row.get('segment_id')}-n{int(row.get('note_index') or 0):03d}" for row in rows]
    return {
        "schema_version": "cantonese_singing_phrase_ctc_example_v2",
        "example_id": f"{segment_id}-phrase-{index:03d}",
        "group_id": str(first.get("variant_group") or segment_id),
        **identities,
        "segment_id": segment_id,
        "note_ids": note_ids,
        "note_count": len(rows),
        "note_start_sec": note_start,
        "note_end_sec": note_end,
        "note_span_duration_sec": round(note_end - note_start, 6),
        "crop_start_sec": round(crop_start, 6),
        "crop_end_sec": round(crop_end, 6),
        "crop_duration_sec": round(crop_end - crop_start, 6),
        "left_context_sec": CONTEXT_SEC,
        "right_context_sec": CONTEXT_SEC,
        "audio_path": str(audio),
        "audio_sha256": probe["sha256"],
        "original_audio_path": first.get("original_audio_path"),
        "original_audio_sha256": optional_sha256(first, "original_audio_sha256"),
        "original_audio_metadata": dict(first.get("original_audio_metadata") or {}),
        "source_audio_sample_rate": probe["sample_rate"],
        "source_audio_channels": probe["channels"],
        "model_audio_sample_rate": MODEL_SAMPLE_RATE,
        "model_audio_channels": 1,
        "target_phone_tokens": targets,
        "internal_ipa_tokens": ipa,
        "articulatory_features": features,
        "label_provenance": aggregate_provenance,
        "label_provenance_by_note": note_provenance,
        "human_v4c_tokens_by_note": human,
        "source_vsqx": first.get("source_vsqx"),
        "source_vsqx_sha256": optional_sha256(first, "source_vsqx_sha256"),
        "source_corpus": first.get("source_corpus"),
        "source_corpus_sha256": optional_sha256(first, "source_corpus_sha256"),
        "human_authored_annotation_evidence": True,
        "manual_annotation_rejected": False,
        "lyrics_used_as_recognizer_input": False,
        "training_data": False,
        "training_data_candidate": True,
    }


def build_dataset(*, manifest_path: Path, output_root: Path, pretty: bool = True) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("rows"), list):
        raise ValueError("phrase manifest must contain a rows list")
    rows = [row for row in manifest["rows"] if isinstance(row, Mapping)]
    if not rows:
        raise ValueError("phrase manifest has no rows")
    paths = {_audio_path(row) for row in rows if _ready(row)}
    if len(paths) != 1:
        raise ValueError("phrase dataset must resolve to one latest WAV")
    audio = next(iter(paths))
    probe = _wav(audio)
    streams, exclusions, breaks = _stream_rows(rows)
    examples: list[dict[str, Any]] = []
    for stream in streams:
        for window in _windows(stream):
            examples.append(_example(window, index=len(examples) + 1, audio=audio, probe=probe))
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest_path": str(manifest_path.resolve(strict=False)),
        "source_manifest_sha256": _sha_file(manifest_path),
        "example_count": len(examples),
        "examples": examples,
        "exclusions": exclusions,
        "continuity_breaks": breaks,
        "audio_read_policy": {"sample_rate": MODEL_SAMPLE_RATE, "channels": 1, "max_window_sec": MAX_WINDOW_SEC},
        "truth_policy": {"manual_v4c_tokens_preserved": True, "internal_ipa_is_training_truth": True, "lyrics_used": False},
        "split_contract": {
            "unit": "connected_component_singer_song_phrase_and_source_hashes",
            "assignment": "required_before_training",
        },
    }
    identity_payload = {
        "schema_version": SCHEMA_VERSION,
        "examples": [_identity_example(row) for row in examples],
        "exclusions": exclusions,
        "continuity_breaks": breaks,
        "audio_read_policy": unsigned["audio_read_policy"],
        "truth_policy": unsigned["truth_policy"],
        "split_contract": unsigned["split_contract"],
    }
    dataset_hash = _sha_json(identity_payload)
    artifact_dir = output_root / dataset_hash
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {**unsigned, "dataset_hash": dataset_hash, "artifact_dir": str(artifact_dir)}
    (artifact_dir / "dataset.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None) + "\n", encoding="utf-8")
    (artifact_dir / "examples.jsonl").write_text("".join(_canonical(row) + "\n" for row in examples), encoding="utf-8")
    lock = {
        "schema_version": "cantonese_singing_phrase_ctc_dataset_lock_v1",
        "dataset_hash": dataset_hash,
        "manifest_sha256": _sha_file(manifest_path),
        "example_count": len(examples),
    }
    (artifact_dir / "dataset_lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return payload


def _identity_example(row: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"audio_path", "source_vsqx", "source_corpus"}
    return {key: row[key] for key in sorted(row) if key not in excluded}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build phrase-level Cantonese singing CTC windows.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = build_dataset(manifest_path=args.manifest, output_root=args.output_root, pretty=args.pretty)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: value for key, value in payload.items() if key != "examples"}, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
