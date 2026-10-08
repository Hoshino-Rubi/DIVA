from pathlib import Path
import ast
import random
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from event_head_completion_common import check_attempt, load_completion
from any2xsampa.training.limited_data_artifacts import write_json_new, read_json, sha256_file
from any2xsampa.training.event_head_resume import save_epoch_state, restore_epoch_state


def assert_tree_equal(torch, a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: assert_tree_equal(torch, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_tree_equal(torch, x, y)
    else:
        assert a == b


def test_resumed_real_event_head_and_optimizer_equal_uninterrupted(tmp_path):
    torch = pytest.importorskip("torch")
    from any2xsampa.training.event_head_training import make_head
    from any2xsampa.training.segmental_event_exact import exact_sequence_nll
    torch.set_num_threads(4);torch.manual_seed(17)
    state = {"weight": torch.randn(428, 1024) * .005, "bias": torch.zeros(428)}
    binding = {"held_song": "fixture_only", "schedule_sha256": "fixed", "worker_sha256": "fixture"}

    def initialize():
        head = make_head(torch, state, "E")
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4, foreach=False)
        random.seed(29);np.random.seed(29);torch.manual_seed(29)
        return head, optimizer

    def step(head, optimizer):
        # Exercise all RNG sources using synthetic fixtures, no acoustic encoder.
        x = torch.randn(5, 1024) * (random.random() + float(np.random.rand()))
        optimizer.zero_grad(set_to_none=True)
        q, b = head(x)
        loss = exact_sequence_nll(torch, q, b, [1, 2]) / 2
        loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()

    uninterrupted, fullopt = initialize()
    for _ in range(3): step(uninterrupted, fullopt)
    full_rng = (random.random(), float(np.random.rand()), torch.rand(3))
    split, splitopt = initialize();step(split, splitopt)
    before = {k: v.detach().clone() for k, v in split.state_dict().items()}
    commit = save_epoch_state(torch, split, splitopt, tmp_path / "epoch-01", epoch=1, steps=1, binding=binding)
    assert_tree_equal(torch, before, split.state_dict())
    restored, restoredopt = initialize()
    info = restore_epoch_state(torch, restored, restoredopt, commit,
        commit_sha256=sha256_file(commit), expected_binding=binding)
    assert info == {"epoch": 1, "optimizer_steps": 1}
    for _ in range(2): step(restored, restoredopt)
    assert_tree_equal(torch, uninterrupted.state_dict(), restored.state_dict())
    assert_tree_equal(torch, fullopt.state_dict(), restoredopt.state_dict())
    resumed_rng = (random.random(), float(np.random.rand()), torch.rand(3))
    assert_tree_equal(torch, full_rng, resumed_rng)
    with pytest.raises(FileExistsError):
        save_epoch_state(torch, restored, restoredopt, tmp_path / "epoch-01", epoch=1, steps=1, binding=binding)


def test_checkpoint_hash_binding_and_incomplete_writes_fail_closed(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    head = torch.nn.Linear(2, 2);optimizer = torch.optim.AdamW(head.parameters(), foreach=False)
    commit = save_epoch_state(torch, head, optimizer, tmp_path / "good", epoch=1, steps=1, binding={"fold": "a"})
    with pytest.raises(ValueError, match="binding"):
        restore_epoch_state(torch, head, optimizer, commit, commit_sha256=sha256_file(commit), expected_binding={"fold": "b"})
    (commit.parent / "state.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        restore_epoch_state(torch, head, optimizer, commit, commit_sha256=sha256_file(commit), expected_binding={"fold": "a"})
    def fail_save(payload, stream):
        stream.write(b"partial");raise OSError("simulated interrupted write")
    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError):
        save_epoch_state(torch, head, optimizer, tmp_path / "partial", epoch=1, steps=1, binding={})
    assert not (tmp_path / "partial/COMMIT.json").exists()


def test_executed_optimizer_loop_is_ast_identical_to_original():
    def nodes(path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
        loop = next(n for n in ast.walk(train) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "offset")
        optimizer = next(n for n in ast.walk(train) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "optimizer" for t in n.targets))
        return ast.dump(loop, include_attributes=False), ast.dump(optimizer, include_attributes=False)
    assert nodes(ROOT / "tools/train_event_head_comparison.py") == nodes(ROOT / "tools/train_event_head_completion.py")


def test_global_budget_whitelist_and_duplicate_start(tmp_path):
    auth = {"allowed_jobs": ["E--promise", "E--queen-road-east"], "prior_quality_starts": 9}
    assert check_attempt(tmp_path, "promise", auth) == 10
    write_json_new(tmp_path / "commands/E--promise.command.json", {})
    assert check_attempt(tmp_path, "promise", auth, worker=True) == 10
    assert check_attempt(tmp_path, "queen-road-east", auth) == 11
    write_json_new(tmp_path / "screen/E/promise/START.json", {})
    with pytest.raises(ValueError, match="already started"):
        check_attempt(tmp_path, "promise", auth, worker=True)
    with pytest.raises(ValueError, match="whitelist"):
        check_attempt(tmp_path, "like-you", auth)
    write_json_new(tmp_path / "commands/E--queen-road-east.command.json", {})
    with pytest.raises(ValueError, match="budget"):
        check_attempt(tmp_path, "queen-road-east", auth)


def test_authorization_cannot_move_to_reset_budget(tmp_path):
    write_json_new(tmp_path / "AUTHORIZATION.json", {"completion_run": str(tmp_path / "other")})
    with pytest.raises(ValueError, match="copied"):
        load_completion(tmp_path, "unused.json")
