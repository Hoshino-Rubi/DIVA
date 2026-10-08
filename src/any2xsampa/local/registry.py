"""Hash-bound, offline model packages. Runnable is separate from quality."""
from __future__ import annotations
from pathlib import Path
import os
import sys
from .common import ROOT, read, sha

def registry_path() -> Path:
    return Path(os.environ.get("ANY2XSAMPA_MODELS", ROOT / "local-models.json"))

def package(model_id: str, *, verify: bool = True) -> dict:
    data = read(registry_path())
    if data.get("schema_version") != "local_model_registry_v1":
        raise ValueError("Unsupported model registry")
    entries = [x for x in data["models"] if x["id"] == model_id]
    if len(entries) != 1:
        raise ValueError("Model not configured: " + model_id)
    spec = entries[0]
    path = (registry_path().parent / spec["package"]).resolve()
    if sha(path) != spec["sha256"]:
        raise ValueError("Model package hash mismatch: " + model_id)
    p = read(path)
    if p.get("schema_version") != "local_model_package_v1" or p.get("id") != model_id:
        raise ValueError("Unsupported model package")
    if p.get("backend") not in {"onnx-espeak", "guarded-phoneticxeus"}:
        raise ValueError("Not an allowed real backend")
    base = (path.parent / p.get("root", ".")).resolve()
    def resolve(name):
        return str((base / name).resolve())
    result = {**p, "package_path": str(path), "package_sha256": spec["sha256"], "base": str(base)}
    result["python"] = resolve(p["python"])
    result["paths"] = {k: resolve(v) for k, v in p["paths"].items()}
    if verify:
        if not p.get("bindings") or not p.get("review"):
            raise ValueError("Missing hash-bound code review")
        for b in p["bindings"]:
            if sha(Path(resolve(b["path"]))) != b["sha256"]:
                raise ValueError("Package binding changed: " + b["path"])
        review = p["review"]
        if sha(Path(resolve(review["path"]))) != review["sha256"]:
            raise ValueError("Code review changed")
    return result

def models(*, verify: bool = True) -> list[dict]:
    if not registry_path().is_file():
        return []
    entries = read(registry_path()).get("models", [])
    result = []
    for entry in entries:
        try:
            p = package(entry["id"], verify=verify)
            result.append({"id": p["id"], "backend": p["backend"], "available": True,
                           "quality": p["quality"], "devices": p["devices"], "max_duration": 20,
                           "python": p["python"], "package_sha256": p["package_sha256"]})
        except Exception as e:
            result.append({"id": entry["id"], "available": False, "reason": str(e)})
    return result

def doctor() -> dict:
    return {"schema_version": "local_doctor_v1", "python": sys.executable,
            "root": str(ROOT), "models": models(), "transport": "stdio",
            "requires_server": False, "model_loaded": False, "network_required": False}
