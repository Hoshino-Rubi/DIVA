"""Split the verified application ZIP into GitHub-sized assets without new compression."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def split(archive: Path, output: Path, expected_sha256: str) -> dict:
    manifest_path = output / "release-manifest.json"
    if manifest_path.exists() or any(output.glob(archive.name + ".0*")):
        raise FileExistsError("Release volumes already exist")
    with archive.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != expected_sha256:
            raise ValueError("Input archive SHA256 changed")
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        roots = {e.filename.split("/")[0] for e in entries}
        if len(roots) != 1 or any(e.is_dir() for e in entries):
            raise ValueError("Expected one application root and file-only ZIP")
        expanded = sum(e.file_size for e in entries)
    parts = []
    full = hashlib.sha256()
    with archive.open("rb") as source:
        index = 0
        while source.tell() < archive.stat().st_size:
            index += 1
            name = archive.name + f".{index:03d}"
            h, size = hashlib.sha256(), 0
            with (output / name).open("xb") as target:
                while size < 1024**3:
                    block = source.read(min(1024**2, 1024**3 - size))
                    if not block:
                        break
                    h.update(block); full.update(block); target.write(block); size += len(block)
            parts.append({"name": name, "bytes": size, "sha256": h.hexdigest()})
            print(f"Created {name}: {size} bytes", flush=True)
    if full.hexdigest() != expected_sha256:
        raise ValueError("Input ZIP changed while splitting; do not release")
    value = {"schema_version": "portable_split_release_v1", "root_directory": roots.pop(),
             "archive_bytes": archive.stat().st_size, "archive_sha256": expected_sha256,
             "expanded_bytes": expanded, "file_count": len(entries), "parts": parts}
    manifest_path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return value


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--sha256", required=True)
    args = p.parse_args()
    print(json.dumps(split(args.archive, args.output, args.sha256)))


if __name__ == "__main__":
    main()
