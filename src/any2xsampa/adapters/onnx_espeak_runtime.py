from __future__ import annotations

"""Runtime and bundle diagnostics for the isolated ONNX/eSpeak producer.

This module deliberately does not run inference.  It reports the local model,
provider and external-worker state so callers can preserve a fail-closed
producer contract.
"""

from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


BUNDLE_SCHEMA_VERSION = "onnx_espeak_runtime_bundle_v1"
REQUIRED_DEPENDENCIES = ("numpy", "onnxruntime", "huggingface_hub")
PYTHON_REQUIRED_VERSION = "3.11"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODEL_ROOT = _PROJECT_ROOT / "models" / "onnx-espeak"
_MODEL_MANIFEST = _PROJECT_ROOT / "models" / "MANIFEST.json"
_MODEL_MANIFEST_IDS = {
    "model": "wav2vec2-lv-60-espeak-cv-ft-onnx-int8",
    "vocab": "wav2vec2-lv-60-espeak-cv-ft-tokenizer",
}
_MODEL_RELATIVE_PATHS = {
    "model": Path("model") / "onnx" / "model_int8.onnx",
    "vocab": Path("tokenizer") / "vocab.json",
}
_OPTIONAL_MODEL_RELATIVE_PATHS = {
    "preprocessor_config": Path("tokenizer") / "preprocessor_config.json",
}

# The bundle worker is a launcher, not a second copy of the producer.  Keeping
# this small wrapper in the bundle preserves the single producer implementation
# while making the external process independent of the caller's PYTHONPATH.
_BUNDLE_WORKER_SOURCE = '''from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    configured = os.environ.get("ANY2XSAMPA_REPO_ROOT")
    if configured:
        roots.append(Path(configured))
    roots.append(Path.cwd())
    roots.extend(Path(__file__).resolve().parents)
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def main() -> None:
    for root in _candidate_roots():
        producer = root / "tools" / "posterior_json_producer.py"
        source_root = root / "src"
        if not producer.is_file():
            continue
        if source_root.is_dir() and str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        runpy.run_path(str(producer), run_name="__main__")
        return
    raise SystemExit(
        "Unable to locate tools/posterior_json_producer.py. "
        "Set ANY2XSAMPA_REPO_ROOT to the repository root or run from it."
    )


if __name__ == "__main__":
    main()
'''


@dataclass(slots=True)
class OnnxEspeakRuntimeCandidate:
    """Description of one external Python worker candidate.

    The historical recognizer imported this value object from the runtime
    probe module.  Keeping it small and mapping-friendly lets callers use the
    probe without coupling to the subprocess implementation.
    """

    command: tuple[str, ...] = ()
    source: str = "external_worker_fallback"
    label: str | None = None
    runtime_root: str | None = None
    worker_entrypoint: str | None = None
    env_overrides: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "source": self.source,
            "label": self.label,
            "runtime_root": self.runtime_root,
            "worker_entrypoint": self.worker_entrypoint,
            "env_overrides": dict(self.env_overrides),
        }


def utc_timestamp() -> str:
    """Return a JSON-safe UTC timestamp used by diagnostics sidecars."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_relative_path(path: Path | str, root: Path | str) -> str:
    """Render a path relative to *root* without allowing ``..`` escapes."""

    candidate = Path(path).resolve(strict=False)
    root_path = Path(root).resolve(strict=False)
    try:
        return candidate.relative_to(root_path).as_posix()
    except ValueError:
        return str(candidate)


def resolve_bundle_relative_path(runtime_root: Path | str, relative_path: str | Path) -> Path:
    """Resolve a manifest path while rejecting paths outside the bundle."""

    root = Path(runtime_root).resolve(strict=False)
    candidate = (root / Path(relative_path)).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"bundle path escapes runtime root: {relative_path}") from exc
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _manifest_resource_snapshot() -> tuple[dict[str, dict[str, Any]], str | None]:
    """Read the immutable model manifest without making diagnostics fatal.

    The ONNX adapter treats a malformed or missing manifest as a hard resource
    error.  Runtime diagnostics must remain serializable so callers can report
    that failure and stay fail-closed instead of crashing while preparing a
    bundle status file.
    """

    try:
        raw = json.loads(_MODEL_MANIFEST.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    if not isinstance(raw, Mapping) or not isinstance(raw.get("models"), list):
        return {}, "ValueError: model manifest must contain a models list"
    resources: dict[str, dict[str, Any]] = {}
    for item in raw["models"]:
        if not isinstance(item, Mapping):
            continue
        model_id = str(item.get("id", ""))
        for resource, expected_id in _MODEL_MANIFEST_IDS.items():
            if model_id != expected_id:
                continue
            digest = item.get("primary_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                return {}, f"ValueError: invalid primary_sha256 for {expected_id}"
            try:
                int(digest, 16)
            except ValueError:
                return {}, f"ValueError: invalid primary_sha256 for {expected_id}"
            resources[resource] = {
                "manifest_id": expected_id,
                "expected_sha256": digest.lower(),
                "expected_size": item.get("primary_size"),
                "source": item.get("source"),
                "revision": item.get("revision"),
            }
    missing = sorted(set(_MODEL_MANIFEST_IDS) - set(resources))
    if missing:
        return {}, "ValueError: model manifest missing " + ", ".join(missing)
    return resources, None


def _model_resource_snapshot() -> dict[str, Any]:
    """Return model/vocabulary bytes and manifest verification evidence."""

    manifest_resources, manifest_error = _manifest_resource_snapshot()
    files: list[dict[str, Any]] = []
    resource_paths = {**_MODEL_RELATIVE_PATHS, **_OPTIONAL_MODEL_RELATIVE_PATHS}
    for resource, relative in resource_paths.items():
        path = _MODEL_ROOT / relative
        manifest_item = manifest_resources.get(resource, {})
        repo_id = (
            "onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX"
            if resource == "model"
            else "facebook/wav2vec2-lv-60-espeak-cv-ft"
        )
        filename = "onnx/model_int8.onnx" if resource == "model" else "vocab.json"
        record: dict[str, Any] = {
            "resource": resource,
            "required": resource in _MODEL_RELATIVE_PATHS,
            "repo_id": repo_id,
            "filename": filename,
            "manifest_id": manifest_item.get("manifest_id"),
            "local_path": str(path),
            "exists": path.is_file(),
            "expected_sha256": manifest_item.get("expected_sha256"),
            "expected_size": manifest_item.get("expected_size"),
            "sha256": None,
            "size_bytes": None,
            "hash_matches": False,
            "size_matches": False,
            "cache_state": "missing",
            "cache_error": None,
            "prefetch_error": None,
            "remote_state": None,
            "remote_http_status": None,
            "remote_url": None,
            "remote_error": None,
            "source": manifest_item.get("source"),
            "revision": manifest_item.get("revision"),
        }
        if path.is_file():
            try:
                digest = _sha256(path)
                size = path.stat().st_size
            except OSError as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            else:
                record["sha256"] = digest
                record["size_bytes"] = size
                record["size"] = size
                record["hash_matches"] = (
                    not record["required"]
                    or (
                        bool(record["expected_sha256"])
                        and digest == record["expected_sha256"]
                    )
                )
                expected_size = record["expected_size"]
                record["size_matches"] = (
                    expected_size is None
                    or (isinstance(expected_size, int) and size == expected_size)
                )
                if record["hash_matches"] and record["size_matches"]:
                    record["cache_state"] = "cached"
                elif not record["hash_matches"]:
                    record["cache_state"] = "hash_mismatch"
                else:
                    record["cache_state"] = "size_mismatch"
        files.append(record)
    missing = [
        item["resource"]
        for item in files
        if item["required"] and not item["exists"]
    ]
    mismatched = [
        item["resource"]
        for item in files
        if item["required"] and item["exists"] and not item["hash_matches"]
    ]
    size_mismatched = [
        item["resource"]
        for item in files
        if item["required"]
        and item["exists"]
        and item["hash_matches"]
        and not item["size_matches"]
    ]
    if manifest_error:
        status = "manifest_invalid"
        failure_category = "manifest_invalid"
        failure_reason = manifest_error
    elif missing:
        status = "missing"
        failure_category = "missing_model_resources"
        failure_reason = "missing model resource(s): " + ", ".join(missing)
    elif mismatched:
        status = "invalid"
        failure_category = "model_hash_mismatch"
        failure_reason = "SHA-256 mismatch for: " + ", ".join(mismatched)
    elif size_mismatched:
        status = "invalid"
        failure_category = "model_size_mismatch"
        failure_reason = "size mismatch for: " + ", ".join(size_mismatched)
    else:
        status = "ready"
        failure_category = None
        failure_reason = None
    return {
        "status": status,
        "ready": status == "ready",
        "failure_category": failure_category,
        "failure_reason": failure_reason,
        "failure_message": (
            "onnx-espeak model resources are cached locally."
            if status == "ready"
            else failure_reason
        ),
        "remediation_message": (
            None
            if status == "ready"
            else "Restore models/onnx-espeak resources and verify models/MANIFEST.json."
        ),
        "cached": status == "ready",
        "fetchable": False,
        "manifest_path": str(_MODEL_MANIFEST),
        "manifest_verified": manifest_error is None and status == "ready",
        "local_files": files,
        "model_repo": "onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX",
        "model_file": "onnx/model_int8.onnx",
        "vocab_repo": "facebook/wav2vec2-lv-60-espeak-cv-ft",
    }


def _version_info(version: str | None) -> list[int] | None:
    if not version:
        return None
    parts: list[int] = []
    for item in str(version).split("."):
        digits = "".join(character for character in item if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return parts or None


def _dependency_snapshot() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_DEPENDENCIES:
        try:
            module = importlib.import_module(name)
            result[name] = {
                "available": True,
                "version": getattr(module, "__version__", None),
                "error": None,
                "module": name,
            }
        except Exception as exc:  # optional runtime dependencies are diagnosed, not raised
            result[name] = {
                "available": False,
                "version": None,
                "error": f"{type(exc).__name__}: {exc}",
                "module": name,
            }
    return result


def _provider_snapshot() -> list[str]:
    try:
        runtime = importlib.import_module("onnxruntime")
        providers = runtime.get_available_providers()
    except Exception:
        return []
    return [str(provider) for provider in providers]


def _external_command_available(command: Sequence[str]) -> bool:
    if not command:
        return False
    executable = str(command[0])
    if any(separator in executable for separator in ("/", "\\")):
        return Path(executable).is_file()
    return shutil.which(executable) is not None


def _unavailable_external_probe(command: Sequence[str], *, reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "ready": False,
        "python_version": None,
        "python_version_info": None,
        "python_version_matches": False,
        "dependencies": {
            name: {"available": False, "version": None, "error": None, "module": name}
            for name in REQUIRED_DEPENDENCIES
        },
        "available_providers": [],
        "provider_ready": False,
        "missing_providers": ["CPUExecutionProvider", "CUDAExecutionProvider"],
        "missing_dependencies": list(REQUIRED_DEPENDENCIES),
        "failure_reason": reason,
        "stderr": None,
        "error": None,
        "command": list(command),
    }


def _probe_external_python(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    timeout: float = 8.0,
) -> dict[str, Any]:
    if not _external_command_available(command):
        return _unavailable_external_probe(command, reason="worker_not_found")
    probe = (
        "import importlib, json, sys; "
        "names=('numpy','onnxruntime','huggingface_hub'); "
        "out={'python_version':'.'.join(map(str,sys.version_info[:3])),'python_version_info':list(sys.version_info[:3]),'dependencies':{},'available_providers':[]}; "
        "\nfor n in names:\n"
        "  try:\n"
        "    m=importlib.import_module(n); out['dependencies'][n]={'available':True,'version':getattr(m,'__version__',None),'error':None,'module':n}\n"
        "  except Exception as e: out['dependencies'][n]={'available':False,'version':None,'error':f'{type(e).__name__}: {e}','module':n}\n"
        "try:\n"
        "  out['available_providers']=[str(p) for p in importlib.import_module('onnxruntime').get_available_providers()]\n"
        "except Exception:\n"
        "  out['available_providers']=[]\n"
        "print(json.dumps(out))"
    )
    try:
        completed = subprocess.run(
            [*command, "-c", probe],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            shell=False,
            env=dict(env) if env is not None else None,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        result = _unavailable_external_probe(command, reason="worker_not_found")
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        payload = {}
    version_info = payload.get("python_version_info")
    dependencies = payload.get("dependencies")
    if not isinstance(dependencies, dict):
        dependencies = {}
    missing = [
        name for name in REQUIRED_DEPENDENCIES
        if not isinstance(dependencies.get(name), Mapping) or not dependencies[name].get("available")
    ]
    python_version = payload.get("python_version")
    version_matches = isinstance(version_info, list) and version_info[:2] == [3, 11]
    available_providers = payload.get("available_providers")
    if not isinstance(available_providers, list):
        available_providers = []
    available_providers = [str(provider) for provider in available_providers]
    provider_ready = bool(
        {"CPUExecutionProvider", "CUDAExecutionProvider"}.intersection(available_providers)
    )
    failure_reason = None
    if completed.returncode != 0:
        failure_reason = "worker_probe_failed"
    elif not version_matches:
        failure_reason = "python_version_mismatch"
    elif missing:
        failure_reason = "missing_dependencies"
    elif not provider_ready:
        failure_reason = "providers_unavailable"
    return {
        "available": completed.returncode == 0,
        "ready": completed.returncode == 0 and version_matches and not missing and provider_ready,
        "python_version": python_version,
        "python_version_info": version_info,
        "python_version_matches": version_matches,
        "dependencies": dependencies,
        "available_providers": available_providers,
        "provider_ready": provider_ready,
        "missing_providers": [] if provider_ready else ["CPUExecutionProvider", "CUDAExecutionProvider"],
        "missing_dependencies": missing,
        "failure_reason": failure_reason,
        "stderr": completed.stderr.strip() or None,
        "error": None if completed.returncode == 0 else f"exit code {completed.returncode}",
    }


def normalize_dependency_probe(value: Mapping[str, Any] | object) -> dict[str, dict[str, Any]]:
    """Normalize dependency rows from an external probe or fixture payload."""

    source: Mapping[str, Any]
    if isinstance(value, Mapping) and isinstance(value.get("dependencies"), Mapping):
        source = value["dependencies"]
    elif isinstance(value, Mapping):
        source = value
    else:
        source = {}
    normalized: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_DEPENDENCIES:
        row = source.get(name)
        if isinstance(row, Mapping):
            available = bool(row.get("available", False))
            version = row.get("version")
            error = row.get("error")
            module = str(row.get("module") or name)
        else:
            available = bool(row) if isinstance(row, bool) else False
            version = None
            error = None
            module = name
        normalized[name] = {
            "available": available,
            "version": str(version) if version is not None else None,
            "error": str(error) if error else None,
            "module": module,
        }
    return normalized


def normalize_python_probe(value: Mapping[str, Any] | object) -> dict[str, Any]:
    """Normalize a Python/dependency probe to the stable diagnostics shape."""

    payload = dict(value) if isinstance(value, Mapping) else {}
    raw_info = payload.get("python_version_info")
    if isinstance(raw_info, str):
        raw_info = _version_info(raw_info)
    version_info = list(raw_info) if isinstance(raw_info, (list, tuple)) else None
    version_text = payload.get("python_version")
    if version_text is None and version_info:
        version_text = ".".join(str(part) for part in version_info)
    dependencies = normalize_dependency_probe(payload)
    missing = [name for name, row in dependencies.items() if not row["available"]]
    version_matches = bool(version_info and list(version_info[:2]) == [3, 11])
    available = bool(payload.get("available", payload.get("python_version") is not None))
    ready = bool(payload.get("ready", available and version_matches and not missing))
    failure_reason = payload.get("failure_reason")
    if not failure_reason:
        if not available:
            failure_reason = "worker_not_found"
        elif not version_matches:
            failure_reason = "python_version_mismatch"
        elif missing:
            failure_reason = "missing_dependencies"
    return {
        "available": available,
        "ready": ready,
        "python_version": str(version_text) if version_text is not None else None,
        "python_version_info": version_info,
        "python_version_matches": version_matches,
        "dependencies": dependencies,
        "available_providers": [str(item) for item in payload.get("available_providers", [])] if isinstance(payload.get("available_providers"), list) else [],
        "provider_ready": bool(payload.get("provider_ready", False)),
        "missing_providers": list(payload.get("missing_providers", [])) if isinstance(payload.get("missing_providers"), list) else [],
        "missing_dependencies": missing,
        "failure_reason": failure_reason,
        "stderr": payload.get("stderr"),
        "error": payload.get("error"),
    }


def probe_external_python_runtime(
    command: Sequence[str] | str,
    *,
    timeout: float = 8.0,
) -> dict[str, Any]:
    """Probe one external Python executable without importing project code."""

    argv = [command] if isinstance(command, str) else list(command)
    if not argv:
        return normalize_python_probe({"failure_reason": "worker_not_found"})
    return normalize_python_probe(_probe_external_python(argv, timeout=timeout))


def probe_external_python_runtime_with_env(
    command: Sequence[str] | str,
    env_overrides: Mapping[str, str] | None = None,
    *,
    timeout: float = 8.0,
) -> dict[str, Any]:
    """Probe an external Python candidate with bundle-specific environment."""

    argv = [command] if isinstance(command, str) else list(command)
    environment = os.environ.copy()
    if env_overrides:
        environment.update({str(key): str(value) for key, value in env_overrides.items()})
    if not _external_command_available(argv):
        return normalize_python_probe({"failure_reason": "worker_not_found"})
    return normalize_python_probe(_probe_external_python(argv, env=environment, timeout=timeout))


def probe_onnx_espeak_runtime_candidate(
    candidate: OnnxEspeakRuntimeCandidate | Mapping[str, Any] | Sequence[str] | str,
    *,
    timeout: float = 8.0,
) -> dict[str, Any]:
    """Probe and decorate a candidate for diagnostics consumers."""

    if isinstance(candidate, OnnxEspeakRuntimeCandidate):
        descriptor = candidate.to_dict()
    elif isinstance(candidate, Mapping):
        descriptor = dict(candidate)
    else:
        command = [candidate] if isinstance(candidate, str) else list(candidate)
        descriptor = {"command": command}
    command_value = descriptor.get("command", [])
    command = [command_value] if isinstance(command_value, str) else list(command_value)
    env_overrides = descriptor.get("env_overrides")
    probe = probe_external_python_runtime_with_env(command, env_overrides, timeout=timeout)
    return {
        **probe,
        **descriptor,
        "command": command,
        "source": descriptor.get("source", "external_worker_fallback"),
        "label": descriptor.get("label"),
        "python_path": command[0] if len(command) == 1 else None,
    }


def _python_candidates(runtime_root: Path) -> list[tuple[list[str], str]]:
    candidates: list[tuple[list[str], str]] = []
    configured = os.environ.get("ANY2XSAMPA_ONNX_ESPEAK_PYTHON")
    if configured:
        candidates.append(([configured], "configured_external_python"))
    for relative in ("Scripts/python.exe", "bin/python", "python.exe", "python"):
        path = runtime_root / relative
        if path not in [Path(item[0][0]) for item in candidates if len(item[0]) == 1]:
            candidates.append(([str(path)], f"bundle:{relative}"))
    for command, label in ((["python3.11"], "python3.11"), (["py", "-3.11"], "py-3.11")):
        candidates.append((command, label))
    return candidates


def _bundle_layout(runtime_root: Path) -> dict[str, Any]:
    runtime_root = runtime_root.resolve(strict=False)
    manifest_path = runtime_root / "manifest.json"
    worker_dir = runtime_root / "worker"
    worker_entrypoint = worker_dir / "posterior_json_producer.py"
    cache_dir = runtime_root / "cache" / "huggingface"
    model_dir = runtime_root / "models" / "onnx-espeak"
    status_file = runtime_root / "bundle_state.json"
    manifest_present = manifest_path.is_file()
    manifest_valid = False
    manifest_error: str | None = None
    manifest_payload: dict[str, Any] = {}
    if manifest_present:
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if not isinstance(raw, dict) or raw.get("schema_version") != BUNDLE_SCHEMA_VERSION:
                raise ValueError(f"expected schema {BUNDLE_SCHEMA_VERSION}")
            manifest_payload = raw
            manifest_valid = True
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            manifest_error = f"{type(exc).__name__}: {exc}"
    python_relative_paths = ("Scripts/python.exe", "bin/python", "python.exe", "python")
    python_candidates = [
        {"relative_path": relative, "path": str(runtime_root / relative), "exists": (runtime_root / relative).is_file()}
        for relative in python_relative_paths
    ]
    selected = next((item for item in python_candidates if item["exists"]), None)
    ready_components: list[str] = []
    if manifest_valid:
        ready_components.append("manifest")
    if worker_entrypoint.is_file():
        ready_components.append("worker")
    if cache_dir.is_dir():
        ready_components.append("cache_dir")
    if model_dir.is_dir():
        ready_components.append("model_dir")
    if selected is not None:
        ready_components.append("runtime_python")
    missing_components = [
        component for component in ("manifest", "worker", "cache_dir", "model_dir", "runtime_python")
        if component not in ready_components
    ]
    detected = bool(ready_components or manifest_present or worker_dir.exists() or runtime_root.exists())
    if not detected:
        status = "not_bundle_layout"
        failure_category = None
    elif not missing_components:
        status = "ready"
        failure_category = None
    elif missing_components == ["runtime_python"]:
        status = "bundle_missing_runtime"
        # Keep the historical category: without an interpreter the worker
        # cannot be launched, even when the worker script itself is present.
        failure_category = "missing_worker"
    else:
        status = "bundle_incomplete"
        failure_category = "missing_component"
    return {
        "runtime_root": str(runtime_root),
        "detected": detected,
        "bundle_ready": not missing_components and manifest_valid,
        "status": status,
        "bundle_status": status,
        "failure_category": failure_category,
        "ready_components": ready_components,
        "missing_components": missing_components,
        "manifest_path": str(manifest_path),
        "manifest_present": manifest_present,
        "manifest_valid": manifest_valid,
        "manifest_error": manifest_error,
        "manifest_schema_version": manifest_payload.get("schema_version") if manifest_valid else None,
        "worker_dir": str(worker_dir),
        "worker_dir_present": worker_dir.is_dir(),
        "worker_entrypoint": str(worker_entrypoint),
        "worker_entrypoint_present": worker_entrypoint.is_file(),
        "dependency_probe_entrypoint": str(worker_entrypoint),
        "cache_dir": str(cache_dir),
        "cache_dir_present": cache_dir.is_dir(),
        "model_dir": str(model_dir),
        "model_dir_present": model_dir.is_dir(),
        "status_file": str(status_file),
        "status_file_present": status_file.is_file(),
        "python_required_version": PYTHON_REQUIRED_VERSION,
        "python_relative_paths": list(python_relative_paths),
        "python_candidates": python_candidates,
        "runtime_python_present": selected is not None,
        "selected_python_candidate": selected,
        "required_dependencies": list(REQUIRED_DEPENDENCIES),
        "manifest_summary": {
            "bundle_name": manifest_payload.get("bundle_name", "onnx-espeak-runtime"),
            "backend_kind": manifest_payload.get("backend_kind", "onnx-espeak"),
            "worker_entrypoint": manifest_payload.get("worker_entrypoint", "worker/posterior_json_producer.py"),
            "cache_root": manifest_payload.get("cache_root", "cache/huggingface"),
            "model_root": manifest_payload.get("model_root", "models/onnx-espeak"),
            "status_file": manifest_payload.get("status_file", "bundle_state.json"),
        },
    }


def inspect_onnx_espeak_runtime_bundle(runtime_root: Path | str) -> dict[str, Any]:
    """Inspect a prepared bundle without creating or mutating it."""

    return _bundle_layout(Path(runtime_root))


def onnx_espeak_bundle_status(layout: Mapping[str, Any] | Path | str) -> str:
    """Return the stable bundle status from a layout payload or root path."""

    if isinstance(layout, Mapping):
        value = layout.get("bundle_status", layout.get("status"))
        return str(value) if value not in {None, ""} else "not_bundle_layout"
    return str(_bundle_layout(Path(layout)).get("bundle_status", "not_bundle_layout"))


def classify_onnx_espeak_runtime_failure(
    diagnostics: Mapping[str, Any] | None = None,
    *,
    bundle_layout: Mapping[str, Any] | None = None,
    model_resources: Mapping[str, Any] | None = None,
) -> dict[str, str | None]:
    """Classify a runtime failure for machine-readable remediation output."""

    payload = diagnostics or {}
    layout = bundle_layout or (
        payload.get("runtime_path", {}).get("primary_bundle_layout", {})
        if isinstance(payload.get("runtime_path"), Mapping)
        else {}
    )
    resources = model_resources or (
        payload.get("model_resources", {}) if isinstance(payload.get("model_resources"), Mapping) else {}
    )
    status = str(payload.get("runtime_status", ""))
    bundle_status = onnx_espeak_bundle_status(layout)
    category = payload.get("failure_category")
    reason = payload.get("failure_reason")
    if not category:
        if resources.get("failure_category"):
            category = resources.get("failure_category")
            reason = reason or resources.get("failure_reason")
        elif bundle_status in {"bundle_missing_runtime", "bundle_incomplete"}:
            category = layout.get("failure_category") or "missing_worker"
            reason = reason or f"runtime bundle status is {bundle_status}"
        elif status in {"unavailable", ""}:
            category = "runtime_unavailable"
            reason = reason or "no ready in-process or external worker runtime"
    return {
        "failure_category": str(category) if category not in {None, ""} else None,
        "failure_reason": str(reason) if reason not in {None, ""} else None,
    }


def resolve_onnx_espeak_runtime_status(
    in_process: Mapping[str, Any] | None = None,
    external_worker: Mapping[str, Any] | None = None,
    model_resources: Mapping[str, Any] | None = None,
) -> str:
    """Resolve dispatch precedence: in-process, external worker, unavailable."""

    if isinstance(in_process, Mapping) and bool(in_process.get("ready")):
        return "in_process_ready"
    if isinstance(external_worker, Mapping) and bool(
        external_worker.get("ready", external_worker.get("worker_available"))
    ):
        return "external_worker_ready"
    if isinstance(model_resources, Mapping) and model_resources.get("ready") is False:
        return "unavailable"
    return "unavailable"


def prepare_onnx_espeak_runtime_bundle(runtime_root: Path | str, *, bundle_name: str = "onnx-espeak-runtime") -> dict[str, str]:
    root = Path(runtime_root).resolve(strict=False)
    for relative in ("worker", "cache/huggingface", "models/onnx-espeak"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    worker_path = root / "worker" / "posterior_json_producer.py"
    # Keep one producer implementation.  The bundle gets a launcher that
    # resolves the source tree at runtime instead of a stale copy of the CLI,
    # whose imports otherwise fail in an external worker without PYTHONPATH.
    worker_path.write_text(_BUNDLE_WORKER_SOURCE, encoding="utf-8")
    manifest_path = root / "manifest.json"
    payload = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle_name": bundle_name,
        "backend_kind": "onnx-espeak",
        "worker_entrypoint": "worker/posterior_json_producer.py",
        "dependency_probe_entrypoint": "worker/posterior_json_producer.py",
        "cache_root": "cache/huggingface",
        "model_root": "models/onnx-espeak",
        "status_file": "bundle_state.json",
        "python_required_version": PYTHON_REQUIRED_VERSION,
        "required_dependencies": list(REQUIRED_DEPENDENCIES),
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "runtime_root": str(root),
        "manifest_path": str(manifest_path),
        "worker_entrypoint": str(worker_path),
        "status_file": str(root / "bundle_state.json"),
    }


def describe_onnx_espeak_runtime(*, prefetch: bool = False) -> dict[str, Any]:
    configured_root = os.environ.get("ANY2XSAMPA_ONNX_ESPEAK_RUNTIME")
    primary_root = Path(configured_root) if configured_root else _PROJECT_ROOT / "runtime" / "onnx-espeak"
    preferred_roots = [primary_root, _PROJECT_ROOT / ".runtime" / "onnx-espeak"]
    layouts = [_bundle_layout(root) for root in preferred_roots]
    primary_layout = layouts[0]
    in_process_dependencies = _dependency_snapshot()
    in_process_missing = [name for name, item in in_process_dependencies.items() if not item["available"]]
    in_process_version_info = list(sys.version_info[:3])
    in_process_providers = _provider_snapshot()
    provider_ready = bool(
        {"CPUExecutionProvider", "CUDAExecutionProvider"}.intersection(in_process_providers)
    )
    in_process = {
        "command": [sys.executable],
        "available": True,
        "ready": not in_process_missing and provider_ready,
        "python_version": ".".join(map(str, in_process_version_info)),
        "python_version_info": in_process_version_info,
        "python_version_matches": in_process_version_info[:2] == [3, 11],
        "dependencies": in_process_dependencies,
        "available_providers": in_process_providers,
        "provider_ready": provider_ready,
        "missing_providers": (
            [] if provider_ready else ["CPUExecutionProvider", "CUDAExecutionProvider"]
        ),
        "missing_dependencies": in_process_missing,
        "failure_reason": (
            "missing_dependencies"
            if in_process_missing
            else "providers_unavailable"
            if not provider_ready
            else None
        ),
        "stderr": None,
        "error": None,
    }
    external_candidates: list[dict[str, Any]] = []
    selected_external: dict[str, Any] | None = None
    for command, label in _python_candidates(primary_root):
        probe = _probe_external_python(command)
        candidate = {
            **probe,
            "command": command,
            "source": "external_worker_fallback",
            "label": label,
            "python_path": command[0] if len(command) == 1 else None,
            "runtime_root": str(primary_root.resolve(strict=False)),
            "worker_entrypoint": str(primary_root.resolve(strict=False) / "worker" / "posterior_json_producer.py"),
            "env_overrides": {
                "ANY2XSAMPA_ONNX_ESPEAK_RUNTIME": str(primary_root.resolve(strict=False)),
                "HF_HOME": str(primary_root.resolve(strict=False) / "cache" / "huggingface"),
            },
            "bundle_layout": primary_layout,
        }
        external_candidates.append(candidate)
        if candidate["ready"] and selected_external is None and primary_layout["worker_entrypoint_present"]:
            selected_external = candidate
    model_resources = _model_resource_snapshot()
    fetchable = importlib.util.find_spec("huggingface_hub") is not None
    missing_files = [
        item["resource"]
        for item in model_resources["local_files"]
        if item.get("required", True) and not item["exists"]
    ]
    model_resources.update(
        {
            "prefetch_requested": bool(prefetch),
            "prefetch_performed": False,
            "checked_via": "in_process" if in_process["ready"] else "none",
            "checked_with_command": [sys.executable] if in_process["ready"] else None,
            "fetchable": fetchable,
            "missing_files": missing_files,
            "fetchable_files": list(missing_files) if fetchable else [],
            "unavailable_files": [] if fetchable else list(missing_files),
            "unknown_files": [],
            "resources": model_resources["local_files"],
        }
    )
    if in_process["ready"] and model_resources["ready"]:
        runtime_status = "in_process_ready"
        dispatch_mode = "in_process"
    elif selected_external is not None and model_resources["ready"]:
        runtime_status = "external_worker_ready"
        dispatch_mode = "external_worker"
    else:
        runtime_status = "unavailable"
        dispatch_mode = "unavailable"
    external_payload = selected_external or {
        "worker_available": False,
        "ready": False,
        "selected_command": None,
        "selected_source": None,
        "selected_label": None,
        "selected_runtime_root": None,
        "selected_python_path": None,
        "selected_worker_entrypoint": None,
        "selected_env_overrides": {},
        "bundle_layout_hit": primary_layout["detected"],
        "bundle_layout": primary_layout,
        "runtime_status": runtime_status,
        "python_version": None,
        "python_version_matches": False,
        "missing_dependencies": list(REQUIRED_DEPENDENCIES),
        "failure_reason": "worker_not_found",
        "candidates": external_candidates,
    }
    if selected_external is not None:
        external_payload = {
            **selected_external,
            "worker_available": True,
            "selected_command": selected_external["command"],
            "selected_source": selected_external["source"],
            "selected_label": selected_external["label"],
            "selected_runtime_root": selected_external["runtime_root"],
            "selected_python_path": selected_external["python_path"],
            "selected_worker_entrypoint": selected_external["worker_entrypoint"],
            "selected_env_overrides": selected_external["env_overrides"],
            "bundle_layout_hit": True,
            "runtime_status": runtime_status,
            "candidates": external_candidates,
        }
    return {
        "backend_kind": "onnx-espeak",
        "bundle_status": primary_layout["bundle_status"],
        "runtime_status": runtime_status,
        "ready": runtime_status != "unavailable" and model_resources["ready"],
        "dispatch_mode": dispatch_mode,
        "failure_category": None if runtime_status != "unavailable" else "runtime_unavailable",
        "failure_reason": None if runtime_status != "unavailable" else "no ready in-process or external worker runtime",
        "failure_message": None,
        "remediation_message": None if runtime_status != "unavailable" else "Install runtime dependencies or provide a valid bundle runtime.",
        "in_process": in_process,
        "external_worker": external_payload,
        "model_resources": model_resources,
        "runtime_path": {
            "repo_root": str(_PROJECT_ROOT),
            "worker_entrypoint": str((_PROJECT_ROOT / "tools" / "posterior_json_producer.py").resolve(strict=False)),
            "worker_cwd": str(_PROJECT_ROOT),
            "worker_pythonpath_entry": str(_PROJECT_ROOT / "src"),
            "python_env_var": "ANY2XSAMPA_ONNX_ESPEAK_PYTHON",
            "runtime_root_env_var": "ANY2XSAMPA_ONNX_ESPEAK_RUNTIME",
            "configured_python": os.environ.get("ANY2XSAMPA_ONNX_ESPEAK_PYTHON"),
            "configured_runtime_root": configured_root,
            "preferred_runtime_roots": [str(root.resolve(strict=False)) for root in preferred_roots],
            "preferred_bundle_layouts": layouts,
            "primary_bundle_layout": primary_layout,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def build_onnx_espeak_runtime_diagnostics_snapshot(runtime_diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": "onnx_espeak_runtime_diagnostics_v1", **dict(runtime_diagnostics)}


def write_onnx_espeak_runtime_diagnostics(
    path: Path | str,
    *,
    runtime_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_onnx_espeak_runtime_diagnostics_snapshot(
        runtime_diagnostics or describe_onnx_espeak_runtime()
    )
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def write_onnx_espeak_bundle_state(
    runtime_root: Path | str,
    *,
    runtime_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(runtime_root).resolve(strict=False)
    layout = _bundle_layout(root)
    diagnostics = runtime_diagnostics or describe_onnx_espeak_runtime()
    runtime_status = str(diagnostics.get("runtime_status", "unavailable"))
    bundle_state = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_root": str(root),
        "bundle_status": layout["bundle_status"],
        "runtime_status": runtime_status,
        "bundle_ready": layout["bundle_ready"],
        "ready_components": layout["ready_components"],
        "missing_components": layout["missing_components"],
        "external_runtime": runtime_status == "external_worker_ready",
        "model_resources": diagnostics.get("model_resources", {}),
        "runtime_diagnostics": dict(diagnostics),
    }
    status_file = root / "bundle_state.json"
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(json.dumps(bundle_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return bundle_state


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "OnnxEspeakRuntimeCandidate",
    "build_onnx_espeak_runtime_diagnostics_snapshot",
    "classify_onnx_espeak_runtime_failure",
    "describe_onnx_espeak_runtime",
    "inspect_onnx_espeak_runtime_bundle",
    "normalize_dependency_probe",
    "normalize_python_probe",
    "onnx_espeak_bundle_status",
    "prepare_onnx_espeak_runtime_bundle",
    "probe_external_python_runtime",
    "probe_external_python_runtime_with_env",
    "probe_onnx_espeak_runtime_candidate",
    "resolve_bundle_relative_path",
    "resolve_onnx_espeak_runtime_status",
    "safe_relative_path",
    "utc_timestamp",
    "write_onnx_espeak_bundle_state",
    "write_onnx_espeak_runtime_diagnostics",
]
