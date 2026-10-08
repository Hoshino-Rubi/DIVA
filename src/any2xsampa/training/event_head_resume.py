"""Hash-bound epoch state snapshots; loading never grants a training restart."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file


def _cpu_copy(torch: Any, value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_copy(torch, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(torch, v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(torch, v) for v in value)
    return value


def rng_state(torch: Any) -> dict:
    name, keys, position, gaussian, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (name, keys.tolist(), int(position), int(gaussian), float(cached)),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [s.cpu().clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [],
    }


def save_epoch_state(torch: Any, head: Any, optimizer: Any, folder: Path, *,
                     epoch: int, steps: int, binding: dict) -> Path:
    if epoch < 1 or steps < 1:
        raise ValueError("only completed epoch boundaries can be saved")
    folder.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": "event_head_epoch_state_v1", "epoch": epoch,
        "optimizer_steps": steps, "binding": binding, "rng": rng_state(torch),
        "head": _cpu_copy(torch, head.state_dict()),
        "optimizer": _cpu_copy(torch, optimizer.state_dict()),
    }
    temporary = folder / "state.pt.tmp"
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    state = folder / "state.pt"
    temporary.rename(state)
    commit = {
        "schema_version": "event_head_epoch_commit_v1", "epoch": epoch,
        "optimizer_steps": steps, "binding": binding,
        "state_file": state.name, "state_sha256": sha256_file(state),
        "safe_load_weights_only": True, "restart_requires_separate_authorization": True,
    }
    pending = folder / "COMMIT.json.tmp"
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(commit, stream, ensure_ascii=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    final = folder / "COMMIT.json"
    pending.rename(final)  # A killed writer without this final marker is unusable.
    return final


def restore_epoch_state(torch: Any, head: Any, optimizer: Any, commit_path: Path, *,
                        commit_sha256: str, expected_binding: dict) -> dict:
    verify_file(commit_path, commit_sha256)
    commit = read_json(commit_path)
    if commit.get("schema_version") != "event_head_epoch_commit_v1" or commit["binding"] != expected_binding:
        raise ValueError("checkpoint binding differs from approved experiment")
    state = (commit_path.parent / commit["state_file"]).resolve()
    if not state.is_relative_to(commit_path.parent.resolve()) or state.name != "state.pt":
        raise ValueError("unsafe checkpoint member")
    verify_file(state, commit["state_sha256"])
    payload = torch.load(state, map_location="cpu", weights_only=True)
    if (payload.get("schema_version") != "event_head_epoch_state_v1"
            or payload["binding"] != expected_binding
            or payload["epoch"] != commit["epoch"]
            or payload["optimizer_steps"] != commit["optimizer_steps"]):
        raise ValueError("checkpoint metadata differs")
    saved = payload["rng"]
    if saved["torch_cuda"] and len(saved["torch_cuda"]) != torch.cuda.device_count():
        raise ValueError("CUDA device geometry changed")
    head.load_state_dict(payload["head"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    random.setstate(saved["python"])
    name, keys, position, gaussian, cached = saved["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(saved["torch_cpu"])
    if saved["torch_cuda"]:
        torch.cuda.set_rng_state_all(saved["torch_cuda"])
    return {"epoch": payload["epoch"], "optimizer_steps": payload["optimizer_steps"]}
