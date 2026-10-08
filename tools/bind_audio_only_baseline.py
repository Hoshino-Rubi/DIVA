from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def bind_baseline(
    manifest_path: Path,
    output_path: Path,
    *,
    song_id: str,
    posterior_path: Path,
    replay_path: Path,
    timing_report_path: Path,
) -> dict[str, Any]:
    source = _json(manifest_path)
    items = source.get("items")
    if not isinstance(items, list):
        raise ValueError("triplet manifest must contain an items list")
    target = next((item for item in items if isinstance(item, Mapping) and item.get("song_id") == song_id), None)
    if target is None:
        raise ValueError(f"song_id not found in triplet manifest: {song_id}")
    posterior = _json(posterior_path)
    if posterior.get("schema_version") != "external_posterior_json_v1":
        raise ValueError("baseline posterior has an unsupported schema")
    producer = posterior.get("producer")
    if not isinstance(producer, Mapping) or producer.get("real_model_output") is not True:
        raise ValueError("baseline posterior is not marked as real model output")
    input_audio = producer.get("input_audio")
    expected_audio = target.get("original_wav")
    if not isinstance(input_audio, Mapping) or not isinstance(expected_audio, Mapping):
        raise ValueError("baseline and triplet are missing input-audio metadata")
    input_hash = str(input_audio.get("sha256") or "").strip().lower()
    expected_hash = str(expected_audio.get("sha256") or "").strip().lower()
    if not input_hash or input_hash != expected_hash:
        raise ValueError("baseline input audio SHA-256 does not match the triplet original WAV")
    if not posterior_path.is_file() or _sha256(posterior_path) == "":
        raise FileNotFoundError(posterior_path)
    replay = _json(replay_path)
    if not isinstance(replay.get("segments"), list):
        raise ValueError("replayed baseline has no segments list")
    timing = _json(timing_report_path)
    phone_segments = timing.get("phone_segments")
    if not isinstance(phone_segments, Mapping):
        raise ValueError("timing report has no phone_segments binding")
    replay_hash = _sha256(replay_path)
    declared_replay_hash = str(phone_segments.get("sha256") or "").strip().lower()
    if declared_replay_hash != replay_hash:
        raise ValueError("timing report replay SHA-256 does not match replay artifact")
    first_segment = posterior.get("segments", [None])[0]
    first_metadata = first_segment.get("metadata") if isinstance(first_segment, Mapping) else {}
    if not isinstance(first_metadata, Mapping):
        first_metadata = {}
    diagnostics = producer.get("model_metadata") if isinstance(producer.get("model_metadata"), Mapping) else {}
    if not diagnostics and isinstance(first_metadata.get("adapter_diagnostics"), Mapping):
        diagnostics = first_metadata["adapter_diagnostics"]
    runtime = producer.get("onnx_espeak_runtime") if isinstance(producer.get("onnx_espeak_runtime"), Mapping) else {}
    if not runtime and isinstance(first_metadata.get("onnx_espeak_runtime"), Mapping):
        runtime = first_metadata["onnx_espeak_runtime"]
    adapter = producer.get("adapter_diagnostics") if isinstance(producer.get("adapter_diagnostics"), Mapping) else {}
    if not adapter and isinstance(first_metadata.get("adapter_diagnostics"), Mapping):
        adapter = first_metadata["adapter_diagnostics"]
    baseline = {
        "posterior_json_path": str(posterior_path.resolve()),
        "posterior_json_sha256": _sha256(posterior_path),
        "replay_segments_path": str(replay_path.resolve()),
        "replay_segments_sha256": replay_hash,
        "timing_report_path": str(timing_report_path.resolve()),
        "timing_report_sha256": _sha256(timing_report_path),
        "segment_count": len(posterior.get("segments", [])),
        "replay_segment_count": len(replay.get("segments", [])),
        "backend": producer.get("backend_kind"),
        "real_model_output": producer.get("real_model_output") is True,
        "input_audio_sha256": input_hash,
        "audio_binding": "verified",
        "device": adapter.get("device"),
        "providers": diagnostics.get("providers"),
        "cuda_active": adapter.get("cuda_active"),
        "provider_fallback_reason": adapter.get("provider_fallback_reason"),
        "runtime_status": runtime.get("runtime_status"),
        "chunk_duration_sec": adapter.get("chunk_duration_sec"),
        "chunk_overlap_sec": adapter.get("chunk_overlap_sec"),
        "cpu_thread_limit": adapter.get("cpu_thread_limit"),
        "lyrics_supplied": producer.get("lyrics_guided") is True,
        "note_context_supplied": producer.get("note_context_path") is not None,
        "rendered_reference_supplied": False,
        "human_annotations_used_as_acoustic_input": False,
    }
    updated = json.loads(json.dumps(source, ensure_ascii=False))
    for item in updated["items"]:
        if isinstance(item, dict) and item.get("song_id") == song_id:
            item["audio_only_baseline"] = baseline
    updated["schema_version"] = "original_route_reference_triplet_v3"
    updated["source_manifest"] = str(manifest_path.resolve())
    updated["source_manifest_sha256"] = _sha256(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bind an audio-only baseline and replay artifacts to one triplet.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--timing-report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        updated = bind_baseline(
            args.manifest,
            args.output,
            song_id=args.song_id,
            posterior_path=args.posterior,
            replay_path=args.replay,
            timing_report_path=args.timing_report,
        )
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"schema_version": updated["schema_version"], "song_id": args.song_id, "output": str(args.output.resolve())}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
