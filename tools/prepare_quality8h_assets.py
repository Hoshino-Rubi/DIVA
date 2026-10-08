"""Retrieve only the approved separator; verify existing Qwen assets upstream."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new

PROXY = "http://127.0.0.1:10808"


def opener() -> urllib.request.OpenerDirector:
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = ""
    return urllib.request.build_opener(urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}))


def retrieve(url: str, target: Path, *, maximum: int, expected_hash: str | None = None, expected_bytes: int | None = None) -> dict:
    if target.exists():
        if expected_hash:
            verify_file(target, expected_hash)
        elif target.stat().st_size > maximum:
            raise ValueError("existing download exceeds its limit")
        if expected_bytes is not None and target.stat().st_size != expected_bytes:
            raise ValueError("existing file size differs from lock")
        return {"url": url, "path": str(target.resolve()), "sha256": sha256_file(target), "bytes": target.stat().st_size, "reused": True}
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    initial = partial.stat().st_size if partial.exists() and expected_hash else 0
    if initial > maximum:
        raise ValueError("partial download exceeds its limit")
    headers = {"User-Agent": "any2xsampa-quality8h/1", "Accept-Encoding": "identity"}
    if initial:
        headers["Range"] = f"bytes={initial}-"
    before = last_progress = time.monotonic()
    with opener().open(urllib.request.Request(url, headers=headers), timeout=90) as response:
        resumed = initial > 0 and response.status == 206
        if resumed and not str(response.headers.get("Content-Range", "")).startswith(f"bytes {initial}-"):
            raise ValueError("incorrect resume range")
        received = initial if resumed else 0
        with partial.open("ab" if resumed else "wb") as stream:
            while block := response.read(2 * 1024 ** 2):
                received += len(block)
                if received > maximum or time.monotonic() - before > 2700:
                    raise RuntimeError("bounded download limit exceeded")
                stream.write(block)
                if time.monotonic() - last_progress > 10:
                    print(f"download={target.name}; bytes={received}; seconds={time.monotonic() - before:.1f}", flush=True)
                    last_progress = time.monotonic()
    if expected_bytes is not None and partial.stat().st_size != expected_bytes:
        raise ValueError("incomplete download; partial retained")
    if expected_hash:
        verify_file(partial, expected_hash)
    partial.rename(target)
    return {"url": url, "path": str(target.resolve()), "sha256": sha256_file(target), "bytes": target.stat().st_size, "elapsed_seconds": time.monotonic() - before, "reused": False}


def unpack_source(archive: Path, target: Path) -> list[dict]:
    records = []
    with zipfile.ZipFile(archive) as source:
        members = [m for m in source.infolist() if not m.is_dir()]
        if len(members) > 200 or sum(m.file_size for m in members) > 20 * 1024 ** 2:
            raise ValueError("unexpected source archive size")
        for member in members:
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts or len(name.parts) < 2 or "\\" in member.filename or ":" in member.filename:
                raise ValueError("unsafe source archive member")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("source archive symlinks are forbidden")
            relative = Path(*name.parts[1:])
            path = target / relative
            data = source.read(member)
            digest = hashlib.sha256(data).hexdigest()
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                verify_file(path, digest)
            else:
                with path.open("xb") as stream:
                    stream.write(data)
            records.append({"file": relative.as_posix(), "bytes": len(data), "sha256": digest})
    return records


def verify_qwen(spec: dict, metadata: dict) -> dict:
    if metadata.get("sha") != spec["revision"] or metadata.get("cardData", {}).get("license") != spec["license"]:
        raise ValueError("Qwen upstream revision/license mismatch")
    members = {m["rfilename"]: m for m in metadata["siblings"]}
    root = ROOT / spec["root"]
    files = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        member = members.get(path.name)
        if member is None:
            raise ValueError(f"undeclared local Qwen file: {path.name}")
        if member.get("lfs"):
            verify_file(path, member["lfs"]["sha256"])
        else:
            value = path.read_bytes()
            git_hash = hashlib.sha1(b"blob " + str(len(value)).encode() + b"\0" + value).hexdigest()
            if git_hash != member["blobId"]:
                raise ValueError(f"local Qwen file differs from pinned upstream blob: {path.name}")
        files.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if not {"model.safetensors", "config.json", "tokenizer_config.json", "preprocessor_config.json"} <= {r["file"] for r in files}:
        raise ValueError("incomplete local Qwen snapshot")
    return {"model": spec, "root": str(root.resolve()), "files": files, "all_existing_bytes_matched_upstream": True}


def prepare(run: Path) -> dict:
    run = run.resolve()
    if not run.is_relative_to(ROOT / ".state/runs"):
        raise ValueError("assets must remain under .state/runs")
    start = read_json(run / "execution-start.json")
    verify_file(run / "policy.json", start["policy_sha256"])
    policy = read_json(run / "policy.json")
    assets = run / "assets"
    report = {"schema_version": "quality8h_assets_v1", "policy_sha256": start["policy_sha256"], "proxy": PROXY,
        "direct_network_fallback": False, "remote_code_executed": False, "downloads": [], "models": {}}
    for role, spec in policy["models"].items():
        url = f"https://huggingface.co/api/models/{spec['id']}/revision/{spec['revision']}?blobs=true"
        target = assets / f"{role}-upstream.json"
        report["downloads"].append(retrieve(url, target, maximum=2 * 1024 ** 2))
        metadata = read_json(target)
        if role != "separator":
            report["models"][role] = verify_qwen(spec, metadata)
            continue
        if metadata["sha"] != spec["revision"] or metadata["cardData"]["license"] != spec["license"]:
            raise ValueError("separator upstream revision/license mismatch")
        weight = next(m for m in metadata["siblings"] if m["rfilename"] == spec["weight_file"])
        if weight["lfs"]["sha256"] != spec["weight_sha256"] or weight["lfs"]["size"] != spec["weight_bytes"]:
            raise ValueError("separator weight declaration differs from frozen policy")
        url = f"https://codeload.github.com/{spec['code_repository']}/zip/{spec['code_revision']}"
        archive = assets / "separator-code.zip"
        report["downloads"].append(retrieve(url, archive, maximum=20 * 1024 ** 2))
        code_files = unpack_source(archive, assets / "separator/code")
        code_lock = {"repository": spec["code_repository"], "revision": spec["code_revision"], "archive_sha256": sha256_file(archive), "files": code_files, "remote_code_executed": False}
        code_lock_path = assets / "separator/code-lock.json"
        if not code_lock_path.exists():
            write_json_new(code_lock_path, code_lock)
        url = f"https://huggingface.co/{spec['id']}/resolve/{spec['revision']}/{spec['weight_file']}"
        weight_path = assets / "separator" / spec["weight_file"]
        report["downloads"].append(retrieve(url, weight_path, maximum=spec["weight_bytes"], expected_bytes=spec["weight_bytes"], expected_hash=spec["weight_sha256"]))
        report["models"][role] = {"model": spec, "weight_path": str(weight_path.resolve()), "code_root": str((assets / "separator/code").resolve()), "code_lock_sha256": sha256_file(code_lock_path)}
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_new(assets / "assets-lock.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    report = prepare(args.run_root)
    print(json.dumps({"models": list(report["models"]), "downloads": len(report["downloads"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
