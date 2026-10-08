from __future__ import annotations

import pytest

from any2xsampa.training.protected_adaptation import baseline_kl, candidate_check
from any2xsampa.training.phone_error_metrics import phone_metrics


def test_kl_identical_distribution_and_teacher_detached() -> None:
    torch = pytest.importorskip("torch")
    torch.set_num_threads(4)
    student = torch.randn(2, 4, 428, requires_grad=True)
    teacher = student.detach().clone().requires_grad_(True)
    loss = baseline_kl(torch, student, teacher, torch.tensor([4, 2]))
    assert abs(loss.item()) < 1e-6
    loss.backward()
    assert teacher.grad is None and torch.isfinite(student.grad).all()


def test_kl_direction_temperature_and_poisoned_padding() -> None:
    torch = pytest.importorskip("torch")
    student = torch.tensor([[[2., 0.], [float("nan"), float("nan")]]], requires_grad=True)
    teacher = torch.tensor([[[0., 2.], [float("inf"), float("inf")]]])
    loss = baseline_kl(torch, student, teacher, torch.tensor([1]), temperature=2)
    p = torch.tensor([0., 1.]).softmax(-1)
    q = torch.tensor([1., 0.]).softmax(-1)
    expected = (p * (p.log() - q.log())).sum() * 4
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.isfinite(student.grad).all()
    assert torch.all(student.grad[:, 1] == 0)


@pytest.mark.parametrize("lengths", [[0], [3], [-1]])
def test_invalid_kl_lengths_are_not_silently_masked(lengths: list) -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="lengths"):
        baseline_kl(torch, torch.zeros(1, 2, 428), torch.zeros(1, 2, 428), torch.tensor(lengths))


def test_protected_selection_rejects_aggregate_gain_with_group_regression() -> None:
    reference = ["kʰ"] * 25 + ["i"] * 100
    base = phone_metrics([(reference, ["kʰ"] * 25 + ["i"] * 50)])
    bad = phone_metrics([(reference, ["k"] * 25 + ["i"] * 100)])
    good = phone_metrics([(reference, ["kʰ"] * 25 + ["i"] * 70)])
    assert not candidate_check(base, bad, 17)["passed"]
    assert candidate_check(base, good, 17)["passed"]


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf"), True])
def test_invalid_temperature_rejected_before_loss(temperature) -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="temperature"):
        baseline_kl(torch, torch.zeros(1, 2, 428), torch.zeros(1, 2, 428), torch.tensor([2]), temperature=temperature)


def test_group_support_cannot_change_even_if_global_reference_count_matches() -> None:
    import copy
    base = phone_metrics([(["kʰ"] * 25 + ["i"] * 100, ["kʰ"] * 25 + ["i"] * 50)])
    other = copy.deepcopy(base)
    other["groups"]["aspirated"]["reference_count"] = 20
    with pytest.raises(ValueError, match="phone groups"):
        candidate_check(base, other, 17)
