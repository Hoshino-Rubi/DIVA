from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path, PurePosixPath
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new  # noqa: E402


MODEL_ID = "changelinglab/PhoneticXeus"
REVISION = "8d83dee94817a07dc150f87d08f7e0ee01bdb66d"
WEIGHT_SHA256 = "ad58bf20a60e9d0380327bd8b2d0e8e90a9b8de2adccbfb479f9b21ea85eda18"
WEIGHT_SIZE = 2_300_089_432
PROXY = "http://127.0.0.1:10808"


def safe_relative(name: str) -> Path:
    value = PurePosixPath(name)
    if value.is_absolute() or ".." in value.parts or "\\" in name or ":" in name:
        raise ValueError(f"invalid snapshot member: {name}")
    return Path(*value.parts)


def download_weight(destination: Path) -> dict:
    url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/model.safetensors"
    if destination.exists():
        verify_file(destination, WEIGHT_SHA256)
        if destination.stat().st_size != WEIGHT_SIZE:
            raise ValueError("existing weight size mismatch")
        return {"downloaded_this_invocation": False, "url": url}
    partial = destination.with_suffix(destination.suffix + ".partial")
    start_size = partial.stat().st_size if partial.exists() else 0
    if start_size > WEIGHT_SIZE:
        raise ValueError("partial weight exceeds declared size")
    os.environ["NO_PROXY"] = ""
    os.environ["no_proxy"] = ""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}))
    headers = {"User-Agent": "any2xsampa-pinned-snapshot/1", "Accept-Encoding": "identity"}
    if start_size:
        headers["Range"] = f"bytes={start_size}-"
    before = time.monotonic()
    with opener.open(urllib.request.Request(url, headers=headers), timeout=90) as response:
        resumed = response.status == 206 and start_size > 0
        if resumed and not str(response.headers.get("Content-Range", "")).startswith(f"bytes {start_size}-"):
            raise ValueError("server returned an inconsistent resume range")
        received = start_size if resumed else 0
        last_progress = before
        with partial.open("ab" if resumed else "wb") as stream:
            while block := response.read(4 * 1024 * 1024):
                received += len(block)
                if received > WEIGHT_SIZE:
                    raise ValueError("weight download exceeds pinned byte count")
                stream.write(block)
                now = time.monotonic()
                if now - last_progress >= 15:
                    print(f"weight_bytes={received}/{WEIGHT_SIZE}; elapsed_seconds={now - before:.1f}", flush=True)
                    last_progress = now
    if partial.stat().st_size != WEIGHT_SIZE:
        raise ValueError("incomplete pinned weight download; partial retained for bounded resume")
    verify_file(partial, WEIGHT_SHA256)
    if destination.exists():
        raise FileExistsError(destination)
    partial.rename(destination)
    return {"downloaded_this_invocation": True, "url": url, "elapsed_seconds": time.monotonic() - before, "resumed": resumed}


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire only the authorized immutable PhoneticXeus snapshot.")
    parser.add_argument("--research-lock", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if not destination.is_relative_to(ROOT / ".state/runs"):
        parser.error("snapshot must remain under .state/runs")
    lock_path = destination / "snapshot-lock.json"
    if lock_path.exists():
        parser.error("snapshot already finalized; verify it instead of overwriting")
    research = read_json(args.research_lock)
    candidates = [row for row in research["candidates"] if row["id"] == MODEL_ID]
    if len(candidates) != 1:
        raise ValueError("research lock does not identify exactly one authorized candidate")
    candidate = candidates[0]
    if candidate["immutable_revision"] != REVISION or candidate["declared_license"] != "apache-2.0":
        raise ValueError("revision/license differ from the approved candidate")
    weight = next(row for row in candidate["weight_files_not_downloaded"] if row["file"] == "model.safetensors")
    if weight["upstream_lfs"]["sha256"] != WEIGHT_SHA256 or weight["bytes"] != WEIGHT_SIZE:
        raise ValueError("weight declaration differs from the approved lock")
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for row in candidate["source_files"]:
        source = Path(row["path"])
        verify_file(source, row["sha256"])
        target = destination / safe_relative(row["file"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            verify_file(target, row["sha256"])
        else:
            with target.open("xb") as stream:
                stream.write(source.read_bytes())
            verify_file(target, row["sha256"])
        files.append({"file": row["file"], "sha256": row["sha256"], "bytes": target.stat().st_size})
    download = download_weight(destination / "model.safetensors")
    files.append({"file": "model.safetensors", "sha256": WEIGHT_SHA256, "bytes": WEIGHT_SIZE})
    lock = {
        "schema_version": "phoneticxeus_snapshot_lock_v1",
        "model_id": MODEL_ID, "revision": REVISION, "license": "apache-2.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_lock_sha256": sha256_file(args.research_lock),
        "proxy": PROXY, "direct_network_fallback": False,
        "files": files, "weight_download": download,
        "all_file_hashes_verified_locally": True, "remote_code_executed": False,
    }
    digest = write_json_new(lock_path, lock)
    print(f"snapshot_lock_sha256={digest}; verified_files={len(files)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
