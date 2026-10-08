from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from any2xsampa.training.limited_data_artifacts import read_json
from any2xsampa.training.limited_data_reference import load_mapping, project_tokens
from any2xsampa.training.original_reference_mapping import OVERLAY_PATH, documented_compound_mapping


def inputs() -> tuple[dict, dict]:
    return load_mapping(Path(__file__).resolve().parents[1] / "src/any2xsampa/config/v4c_reference_ipa_v4.json"), read_json(OVERLAY_PATH)


def test_documented_compound_preserves_base_and_distinctions() -> None:
    base, overlay = inputs()
    old = deepcopy(base)
    new = documented_compound_mapping(base, overlay)
    assert base == old
    assert len(new["tokens"]) == len(base["tokens"]) + 1
    assert new["tokens"]["ua_n"]["ipa"] == ["u", "\u0250", "n"]
    assert all(new["tokens"][key] == value for key, value in old["tokens"].items())
    assert not new["acoustic_realization_verified"]


def test_other_unknowns_are_not_split_by_guesswork() -> None:
    base, overlay = inputs()
    new = documented_compound_mapping(base, overlay)
    vocab = {"<blank>": 0, "u": 1, "\u0250": 2, "n": 3}
    value = project_tokens(["ua_n", "uAN"], new, vocab)
    assert value["unresolved_tokens"] == ["uAN"]
    assert value["model_target_ipa"] == []


def test_overlay_cannot_introduce_arbitrary_aliases() -> None:
    base, overlay = inputs()
    overlay["rules"]["uo"] = {"source_tokens": ["u", "o"], "ipa": ["u", "o"]}
    with pytest.raises(ValueError, match="only the documented"):
        documented_compound_mapping(base, overlay)


def test_damaged_or_changed_constituents_rejected() -> None:
    base, overlay = inputs()
    base["tokens"]["ua"]["ipa"] = ["u", "?"]
    with pytest.raises(ValueError, match="constituents"):
        documented_compound_mapping(base, overlay)
