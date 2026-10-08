"""Check actual loaded tensors, including explicitly shared tied aliases."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def verify_loaded_tensors(torch: Any, model: Any, weight_path: Path) -> dict:
    from safetensors import safe_open

    state = model.state_dict()
    if not state or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("loaded model must expose a nonempty tensor state dictionary")
    aliases = {}
    count = 0
    with safe_open(str(weight_path), framework="pt", device="cpu") as source:
        names = set(source.keys())
        if names - set(state):
            raise ValueError("checkpoint contains tensors absent from the loaded model")
        for name in sorted(names):
            expected = source.get_tensor(name)
            actual = state[name]
            if tuple(actual.shape) != tuple(expected.shape):
                raise ValueError(f"loaded tensor shape differs from checkpoint: {name}")
            if not torch.equal(actual.detach().cpu(), expected.to(dtype=actual.dtype)):
                raise ValueError(f"loaded tensor values differ from checkpoint: {name}")
            count += actual.numel()
        for name in sorted(set(state) - names):
            value = state[name]
            matches = [other for other in names if value.numel() > 0 and value.shape == state[other].shape
                       and value.dtype == state[other].dtype and value.device == state[other].device
                       and value.data_ptr() == state[other].data_ptr() and value.stride() == state[other].stride()]
            if not matches:
                raise ValueError(f"loaded model has an unverified, non-tied tensor: {name}")
            aliases[name] = min(matches)
    return {"checkpoint_tensors_verified": len(names), "checkpoint_elements_verified": count,
            "loaded_state_tensors": len(state), "tied_aliases": aliases,
            "all_loaded_tensors_match_checkpoint_at_runtime_dtype": True,
            "missing_randomly_initialized_tensors": [], "checkpoint_values_modified": False}
