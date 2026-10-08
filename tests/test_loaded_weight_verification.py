from __future__ import annotations

from pathlib import Path

import pytest

from any2xsampa.adapters.loaded_weight_verification import verify_loaded_tensors


def test_loaded_values_and_shapes_must_match_not_merely_a_successful_load(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safe = pytest.importorskip("safetensors.torch")
    model = torch.nn.Linear(4, 3)
    path = tmp_path / "weights.safetensors"
    safe.save_file(model.state_dict(), str(path))
    assert verify_loaded_tensors(torch, model, path)["checkpoint_tensors_verified"] == 2
    with torch.no_grad():
        model.bias[0] += 1
    with pytest.raises(ValueError, match="values differ"):
        verify_loaded_tensors(torch, model, path)


def test_only_real_shared_ties_may_be_omitted_from_checkpoint(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safe = pytest.importorskip("safetensors.torch")
    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.randn(3, 4))
            self.b = self.a
    model = Tied()
    path = tmp_path / "weights.safetensors"
    safe.save_file({"a": model.a}, str(path))
    assert verify_loaded_tensors(torch, model, path)["tied_aliases"] == {"b": "a"}
    model.b = torch.nn.Parameter(model.a.detach().clone())
    with pytest.raises(ValueError, match="non-tied"):
        verify_loaded_tensors(torch, model, path)
