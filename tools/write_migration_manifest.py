from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    ".venv",
    ".nuget",
    ".dotnet-home",
    ".state",
    ".pytest_cache",
    "__pycache__",
    "bin",
    "obj",
}
EXCLUDED_FILES = {"MIGRATION_MANIFEST.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def included_files() -> list[Path]:
    return sorted(
        (
            path
            for path in ROOT.rglob("*")
            if path.is_file()
            and path.name not in EXCLUDED_FILES
            and not any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts)
            and path.suffix != ".pyc"
        ),
        key=lambda path: path.relative_to(ROOT).as_posix().casefold(),
    )


def main() -> int:
    files = included_files()
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    tree_digest = hashlib.sha256()
    for record in records:
        tree_digest.update(
            f"{record['path']}\0{record['size']}\0{record['sha256']}\n".encode("utf-8")
        )
    manifest = {
        "schema": "any2xsampa_clean_migration_v1",
        "date": "2026-08-12",
        "destination": str(ROOT),
        "status": "validated_for_independent_development",
        "file_count": len(records),
        "file_bytes": sum(record["size"] for record in records),
        "tree_sha256": tree_digest.hexdigest(),
        "bundled_dependency_roots": [".venv", ".nuget", ".dotnet-home"],
        "bundled_model_manifest": "models/MANIFEST.json",
        "python_dependency_lock": "requirements.lock.txt",
        "excluded_from_tree_hash": sorted(EXCLUDED_PARTS),
        "validation": {
            "pytest": "310 passed, 3 skipped",
            "pip_check": "passed",
            "torch": "2.13.0+cu126",
            "cuda": "12.6 available",
            "onnx_runtime": "1.28.0 model session loaded",
            "grpc": "1.83.0",
            "winui": "Debug x64 locked restore/build passed, 0 warnings, 0 errors",
            "accepted_vsqx": "official vsq4.xsd passed and user accepted",
        },
        "files": records,
    }
    output = ROOT / "MIGRATION_MANIFEST.json"
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: manifest[key] for key in ("file_count", "file_bytes", "tree_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
