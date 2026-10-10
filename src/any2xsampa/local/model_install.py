"""Pinned model installation. Network exists only in this explicit setup path."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import urllib.request
from urllib.parse import urlsplit
import zipfile

from .common import ROOT, check_cancel, read, sha
from .storage import model_cache, state_dir

BLOCK = 1024 * 1024


def inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("Invalid installation path")
    parts = PurePosixPath(relative).parts
    if relative.startswith("/") or any(p in {".", ".."} for p in parts):
        raise ValueError("Unsafe installation path")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Installation path escapes root")
    return target


def _entry(row: dict) -> None:
    if type(row.get("bytes")) is not int or row["bytes"] < 0:
        raise ValueError("Invalid pinned file size")
    if not re.fullmatch("[0-9a-f]{64}", row.get("sha256", "")):
        raise ValueError("Invalid pinned SHA256")


def catalog(root: Path = ROOT) -> dict | None:
    file = root / "model-downloads.json"
    if not file.is_file():
        return None
    data = read(file)
    if data.get("schema_version") != "local_model_downloads_v1":
        raise ValueError("Unsupported download catalog")
    seen, ids = set(), set()
    for bundle in data["bundles"]:
        if bundle["id"] in ids:
            raise ValueError("Duplicate model id")
        ids.add(bundle["id"])
        known = {}
        for row in bundle["files"]:
            _entry(row)
            target = inside(root, row["path"])
            if not target.is_relative_to((root / "models").resolve()) or row["path"].casefold() in seen:
                raise ValueError("Duplicate or non-model installation target")
            seen.add(row["path"].casefold())
            known[row["path"]] = row
        downloaded = set()
        for item in bundle["downloads"]:
            _entry(item)
            url = urlsplit(item["url"])
            if url.scheme != "https" or not url.hostname or url.username or url.password or url.fragment:
                raise ValueError("Model downloads require HTTPS without credentials")
            if "target" in item:
                row = known.get(item["target"])
                if not row or any(row[k] != item[k] for k in ("bytes", "sha256")):
                    raise ValueError("Download identity does not match target")
                targets = [item["target"]]
            else:
                targets = []
                archives = set()
                for member in item["members"]:
                    inside(root, member["archive_path"])
                    if member["archive_path"] in archives or member["target"] not in known:
                        raise ValueError("Invalid archive member binding")
                    archives.add(member["archive_path"])
                    targets.append(member["target"])
            if any(t in downloaded for t in targets):
                raise ValueError("Repeated download target")
            downloaded.update(targets)
            parts = item.get("offline_parts", [])
            for part in parts:
                _entry(part)
                inside(root, part["name"])
            if parts and sum(p["bytes"] for p in parts) != item["bytes"]:
                raise ValueError("Offline parts have wrong total size")
            inside(root, item["offline_name"])
    return data


def matches(path: Path, row: dict) -> bool:
    return path.is_file() and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"]


def status(root: Path = ROOT) -> dict:
    data = catalog(root)
    if data is None:
        return {"enabled": False, "ready": False, "auto_download": False, "bundles": []}
    rows = []
    for bundle in data["bundles"]:
        missing = [r["path"] for r in bundle["files"] if not matches(inside(root, r["path"]), r)]
        rows.append({"id": bundle["id"], "name": bundle["name"], "license": bundle["license"],
                     "ready": not missing, "missing_files": len(missing),
                     "download_bytes": sum(r["bytes"] for r in bundle["downloads"])})
    return {"enabled": True, "auto_download": data.get("auto_download", False),
            "ready": all(r["ready"] for r in rows), "bundles": rows}


@contextmanager
def install_lock(root: Path):
    directory = state_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model-install.lock"
    with path.open("a+b") as stream:
        if not path.stat().st_size:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another model installation is already running") from exc
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _copy_verified(source, destination: Path, expected: dict, cancel, progress) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".installing")
    digest, count = hashlib.sha256(), 0
    try:
        with tmp.open("wb") as out:
            while block := source.read(BLOCK):
                check_cancel(cancel)
                count += len(block)
                if count > expected["bytes"]:
                    raise ValueError("Installation source exceeds pinned size")
                digest.update(block)
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        check_cancel(cancel)
        if count != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
            raise ValueError("Installation SHA256/size mismatch")
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)


def _download(item: dict, cache: Path, cancel, progress, opener) -> Path:
    complete = cache / (item["sha256"] + ".blob")
    if matches(complete, item):
        return complete
    partial = cache / (item["sha256"] + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > item["bytes"]:
        raise ValueError("Partial model exceeds pinned size; use reset_downloads")
    if offset == item["bytes"]:
        if not matches(partial, item):
            partial.replace(cache / (item["sha256"] + ".bad"))
            raise ValueError("Downloaded model SHA256 mismatch; nothing installed")
        os.replace(partial, complete)
        return complete
    headers = {"User-Agent": "Any2Xsampa-ModelInstaller/1", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(item["url"], headers=headers)
    with opener.open(request, timeout=5) as response:
        if urlsplit(response.geturl()).scheme != "https":
            raise ValueError("Model download redirected outside HTTPS")
        code = response.status
        if offset:
            expected_range = f"bytes {offset}-{item['bytes'] - 1}/{item['bytes']}"
            if code != 206 or response.headers.get("Content-Range") != expected_range:
                raise ValueError("Server refused exact resume range; partial retained. Use reset_downloads to restart.")
        elif code != 200:
            raise ValueError(f"Unexpected model HTTP status: {code}")
        length = response.headers.get("Content-Length")
        if length is not None and int(length) != item["bytes"] - offset:
            raise ValueError("Remote model length changed")
        with partial.open("ab" if offset else "wb") as out:
            while block := response.read(BLOCK):
                check_cancel(cancel)
                if offset + len(block) > item["bytes"]:
                    raise ValueError("Remote model exceeds pinned size")
                out.write(block)
                offset += len(block)
                progress({"stage": "model_download", "file": item["offline_name"],
                          "downloaded_bytes": offset, "total_bytes": item["bytes"]})
    check_cancel(cancel)
    if not matches(partial, item):
        if partial.stat().st_size == item["bytes"]:
            partial.replace(cache / (item["sha256"] + ".bad"))
        raise ValueError("Incomplete download or SHA256 mismatch; model not installed")
    os.replace(partial, complete)
    return complete


def _offline(item: dict, directory: Path, cache: Path, cancel, progress) -> Path:
    whole = inside(directory, item["offline_name"])
    if whole.is_file():
        if not matches(whole, item):
            raise ValueError("Offline model SHA256 mismatch: " + whole.name)
        return whole
    parts = item.get("offline_parts", [])
    if not parts:
        raise FileNotFoundError("Offline model missing: " + whole.name)
    target = cache / (item["sha256"] + ".blob")
    if matches(target, item):
        return target
    tmp = cache / (item["sha256"] + ".joining")
    try:
        with tmp.open("wb") as out:
            for row in parts:
                file = inside(directory, row["name"])
                if not matches(file, row):
                    raise ValueError("Offline part missing or hash mismatch: " + row["name"])
                with file.open("rb") as source:
                    while block := source.read(BLOCK):
                        check_cancel(cancel)
                        out.write(block)
                progress({"stage": "model_import", "file": row["name"]})
        if not matches(tmp, item):
            raise ValueError("Joined model SHA256 mismatch")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def _install_blob(item, blob, root, known, cancel, progress):
    if "target" in item:
        with blob.open("rb") as source:
            _copy_verified(source, inside(root, item["target"]), known[item["target"]], cancel, progress)
        return
    wanted = {m["archive_path"]: m["target"] for m in item["members"]}
    with zipfile.ZipFile(blob) as archive:
        names = []
        for entry in archive.infolist():
            inside(root, entry.filename.rstrip("/"))
            if entry.is_dir():
                continue
            if entry.filename not in wanted or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unexpected or symlink archive member")
            names.append(entry.filename)
        if len(names) != len(set(names)) or set(names) != set(wanted):
            raise ValueError("Archive members differ from pinned list")
        for name, target in wanted.items():
            check_cancel(cancel)
            if archive.getinfo(name).file_size != known[target]["bytes"]:
                raise ValueError("Archive expanded size changed")
            with archive.open(name) as source:
                _copy_verified(source, inside(root, target), known[target], cancel, progress)


def install(params: dict, cancel, progress, root: Path = ROOT, *, opener=None) -> dict:
    if set(params) - {"proxy", "offline_dir", "reset_downloads"}:
        raise ValueError("Unexpected model installation parameters")
    data = catalog(root)
    if data is None:
        raise ValueError("This build has no model download catalog")
    cache = model_cache(root)
    cache.mkdir(parents=True, exist_ok=True)
    proxy = params.get("proxy", "").strip()
    if proxy and (urlsplit(proxy).scheme not in {"http", "https"} or not urlsplit(proxy).hostname):
        raise ValueError("Proxy must be an HTTP(S) URL")
    offline = Path(params["offline_dir"]).resolve() if params.get("offline_dir") else None
    if opener is None and offline is None:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}) if proxy
                                            else urllib.request.ProxyHandler())
    with install_lock(root):
        for bundle in data["bundles"]:
            check_cancel(cancel)
            known = {r["path"]: r for r in bundle["files"]}
            for item in bundle["downloads"]:
                targets = [item["target"]] if "target" in item else [r["target"] for r in item["members"]]
                if all(matches(inside(root, t), known[t]) for t in targets):
                    continue
                if params.get("reset_downloads"):
                    (cache / (item["sha256"] + ".part")).unlink(missing_ok=True)
                needed = sum(known[t]["bytes"] for t in targets if not matches(inside(root, t), known[t]))
                if offline is None and not matches(cache / (item["sha256"] + ".blob"), item):
                    needed += item["bytes"]
                if shutil.disk_usage(root).free < needed + 64 * BLOCK:
                    raise OSError("Not enough disk space for verified model installation")
                progress({"stage": "model_download", "model": bundle["name"],
                          "file": item["offline_name"], "downloaded_bytes": 0, "total_bytes": item["bytes"]})
                blob = _offline(item, offline, cache, cancel, progress) if offline else _download(item, cache, cancel, progress, opener)
                _install_blob(item, blob, root, known, cancel, progress)
            missing = [t for t, row in known.items() if not matches(inside(root, t), row)]
            if missing:
                raise ValueError("Bundled model metadata damaged; re-extract the application: " + ", ".join(missing))
        progress({"stage": "model_ready"})
        return status(root)
