from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import wave


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_reference_ctc_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_reference_ctc_manifest", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_mapping_covers_current_delivery_inventory() -> None:
    mapping = json.loads(MODULE.DEFAULT_MAPPING.read_text(encoding="utf-8"))
    tokens = {
        "@N", "@U", "@_n", "@`", "AN", "AU", "a", "aI", "a_n", "ei",
        "f", "i", "i@U", "iAN", "iAU", "iN", "i_n", "ia", "k", "k_h",
        "l", "m", "n", "o", "p", "p_h", "s", "t", "t_h", "ts", "ts_h",
        "u", "u@N", "u@_n", "ua", "uaI", "uei", "x", "y", "y_n",
    }
    assert tokens == set(mapping["tokens"])
    for token in sorted(tokens):
        phones, ipa, features, _ = MODULE._map_tokens([token], mapping, "a" * 64)
        assert phones and len(phones) == len(ipa) == len(features)


def test_unknown_v4c_token_fails_instead_of_guessing() -> None:
    mapping = json.loads(MODULE.DEFAULT_MAPPING.read_text(encoding="utf-8"))
    try:
        MODULE._map_tokens(["UNKNOWN"], mapping, "a" * 64)
    except ValueError as exc:
        assert "no recognizer-contract mapping" in str(exc)
    else:
        raise AssertionError("unknown V4C token must be rejected")
