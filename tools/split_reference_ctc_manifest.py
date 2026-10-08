from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_manifest(source: Path, output_root: Path) -> dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("rows"), list):
        raise ValueError("reference CTC manifest must contain rows")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in payload["rows"]:
        if not isinstance(raw, Mapping):
            continue
        song_id = str(raw.get("song_id") or raw.get("segment_id") or "").strip()
        if not song_id:
            raise ValueError("reference CTC row has no song_id")
        grouped.setdefault(song_id, []).append(dict(raw))
    if len(grouped) < 2:
        raise ValueError("reference CTC manifest must contain at least two songs")
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    for song_id, rows in sorted(grouped.items()):
        out = output_root / f"{song_id}.json"
        song_payload = {
            "schema_version": "cantonese_singing_reference_manifest_song_v1",
            "source_manifest": str(source.resolve()),
            "source_manifest_sha256": _sha256(source),
            "song_id": song_id,
            "row_count": len(rows),
            "rows": rows,
        }
        out.write_text(json.dumps(song_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        outputs.append({"song_id": song_id, "path": str(out.resolve()), "sha256": _sha256(out), "row_count": len(rows)})
    summary = {
        "schema_version": "cantonese_singing_reference_manifest_split_v1",
        "source_manifest": str(source.resolve()),
        "source_manifest_sha256": _sha256(source),
        "song_count": len(outputs),
        "songs": outputs,
    }
    (output_root / "split-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split a reference CTC manifest into one manifest per song.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = split_manifest(args.source, args.output_root)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
