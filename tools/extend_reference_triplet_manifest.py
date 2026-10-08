from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import wave
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from any2xsampa.vocaloid.vsqx_alignment import parse_vsqx_timeline  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audio(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frames = handle.getnframes()
    if sample_rate <= 0 or channels <= 0 or frames <= 0:
        raise ValueError(f"invalid WAV metadata: {path}")
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frames": frames,
        "duration_sec": round(frames / sample_rate, 6),
    }


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
    }
    if path.suffix.casefold() == ".wav":
        record["audio"] = _audio(path)
    return record


def build_manifest(
    base_manifest: Path,
    output: Path,
    *,
    song: str,
    song_id: str,
    original_wav: Path,
    rendered_wav: Path,
    vsqx: Path,
    rhythm_claim: str,
    pitch_claim: str,
) -> dict[str, Any]:
    payload = json.loads(base_manifest.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("base manifest must contain an items list")
    if not song_id.strip() or not song.strip():
        raise ValueError("song and song_id must be non-empty")
    items = [dict(item) for item in payload["items"] if isinstance(item, dict)]
    defaults = ["freedom-flower", "queen-road-east"]
    for index, item in enumerate(items):
        item.setdefault("song_id", defaults[index] if index < len(defaults) else f"song-{index + 1}")
    if any(str(item.get("song_id") or "") == song_id for item in items):
        raise ValueError(f"song_id already exists: {song_id}")
    timeline = parse_vsqx_timeline(vsqx)
    original = _file_record(original_wav)
    rendered = _file_record(rendered_wav)
    vsqx_record = _file_record(vsqx)
    item = {
        "song": song,
        "song_id": song_id,
        "human_claims": {
            "phoneme_correctness": "user_asserted_correct",
            "rhythm_accuracy": rhythm_claim,
            "pitch_accuracy": pitch_claim,
        },
        "pairing_status": "user_declared_final_delivery_pending_machine_pair_verification",
        "original_wav": original,
        "rendered_wav": rendered,
        "human_vsqx": {
            **vsqx_record,
            "normalized_path": vsqx_record["path"],
            "normalized_sha256": vsqx_record["sha256"],
            "normalization_applied": False,
            "note_count": int(timeline["note_count"]),
        },
    }
    items.append(item)
    result = {
        **payload,
        "schema_version": "original_route_reference_triplet_v2",
        "items": items,
        "triplet_count": len(items),
        "new_item": {"song_id": song_id, "song": song},
        "source_manifest": str(base_manifest.resolve()),
        "source_manifest_sha256": _sha256(base_manifest),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extend a reference triplet manifest with one verified song.")
    parser.add_argument("base_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--song", required=True)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--original-wav", type=Path, required=True)
    parser.add_argument("--rendered-wav", type=Path, required=True)
    parser.add_argument("--vsqx", type=Path, required=True)
    parser.add_argument("--rhythm-claim", default="user_asserted_>=0.90")
    parser.add_argument("--pitch-claim", default="user_asserted_correct")
    args = parser.parse_args(argv)
    try:
        payload = build_manifest(
            args.base_manifest,
            args.output,
            song=args.song,
            song_id=args.song_id,
            original_wav=args.original_wav,
            rendered_wav=args.rendered_wav,
            vsqx=args.vsqx,
            rhythm_claim=args.rhythm_claim,
            pitch_claim=args.pitch_claim,
        )
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    # Keep the CLI summary safe on Windows consoles whose active code page
    # cannot encode every user-supplied song title. The manifest itself remains
    # UTF-8 with literal Unicode.
    print(json.dumps({"schema_version": payload["schema_version"], "triplet_count": payload["triplet_count"], "new_item": payload["new_item"]}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
