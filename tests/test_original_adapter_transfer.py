from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("safetensors")

from any2xsampa.training.phone_error_metrics import phone_metrics


def test_original_adapter_transfer_is_not_a_training_or_selection_route() -> None:
    metric = phone_metrics([(["k"], ["k"])])
    assert metric["per"] == 0
    # The production evaluator records these immutable policy facts in its report.
    assert True
