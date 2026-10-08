from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from any2xsampa.adapters.forced_aligner import ForcedPhoneAlignerError
from any2xsampa.adapters.mfa_forced_aligner import _canonical_phone


_SOURCE_PHONE_SET = frozenset(
    {
        "<eps>",
        "sil",
        "spn",
        "O",
        "a",
        "a:",
        "a:a:",
        "a:i",
        "a:u",
        "b",
        "d",
        "e",
        "ei",
        "f",
        "g",
        "h",
        "i:",
        "i:a",
        "i:u",
        "j",
        "k",
        "k\u02b0",
        "k\u02b7",
        "k\u02b7\u02b0",
        "l",
        "m",
        "m\u0329",
        "n",
        "nn",
        "ou",
        "p",
        "pp",
        "p\u02b0",
        "r",
        "s",
        "t",
        "ts",
        "ts\u02b0",
        "t\u02b0",
        "u",
        "u:",
        "u:i",
        "v",
        "w",
        "y:",
        "\u0103",
        "\u014b",
        "\u014b\u0329",
        "\u0153:",
        "\u0250",
        "\u0250i",
        "\u0250u",
        "\u0254:",
        "\u0254:i",
        "\u025b:",
        "\u025b:u",
        "\u0261",
        "\u026a",
        "\u0275",
        "\u0275y",
        "\u0281",
        "\u028a",
        "\u02c8i",
        "\u02c8\u025b:",
        "\u02cce",
        "\u02cci",
    }
)


def _mapping() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "src/any2xsampa/config/mfa_cv19_v3_phone_map.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_cv19_v3_source_phones_have_complete_non_overlapping_coverage() -> None:
    mapping = _mapping()
    phones = set(mapping["phones"])
    silence = set(mapping["silence"])
    oov = set(mapping["oov"])

    assert len(_SOURCE_PHONE_SET) == 66
    assert phones | silence | oov == _SOURCE_PHONE_SET
    assert phones.isdisjoint(silence)
    assert phones.isdisjoint(oov)
    assert silence.isdisjoint(oov)
    assert len(phones) == 62
    assert mapping["phones"].get("i") is None
    assert "i:" in phones

    model_phones_path = (
        Path(__file__).resolve().parents[1]
        / ".state/runs/forced-aligner-resources/HKCantonese_models/cv19_v3/canto_model_v3/phones.txt"
    )
    if model_phones_path.is_file():
        model_phones = {
            line.split("\t", 1)[0]
            for line in model_phones_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        assert model_phones == _SOURCE_PHONE_SET


def test_cv19_v3_ipa_mapping_preserves_source_contrasts() -> None:
    phones = _mapping()["phones"]

    expected = {
        "ts": "ts",
        "ts\u02b0": "ts\u02b0",
        "k": "k",
        "k\u02b0": "k\u02b0",
        "k\u02b7": "k\u02b7",
        "k\u02b7\u02b0": "k\u02b7\u02b0",
        "p": "p",
        "p\u02b0": "p\u02b0",
        "t": "t",
        "t\u02b0": "t\u02b0",
        "i:": "i\u02d0",
        "a:": "a\u02d0",
        "u:": "u\u02d0",
        "\u0153:": "\u0153\u02d0",
        "\u0254:": "\u0254\u02d0",
        "\u025b:": "\u025b\u02d0",
        "nn": "n\u02d0",
        "pp": "p\u02d0",
    }
    for source, canonical in expected.items():
        assert phones[source] == canonical

    for sources in (
        ("k", "k\u02b0", "k\u02b7", "k\u02b7\u02b0"),
        ("p", "p\u02b0"),
        ("t", "t\u02b0"),
        ("ts", "ts\u02b0"),
        ("i:", "a:", "u:", "\u0153:", "\u0254:", "\u025b:"),
        ("n", "nn"),
        ("p", "pp"),
    ):
        assert len({phones[source] for source in sources}) == len(sources)


def test_cv19_v3_map_is_internal_ipa_not_xsampa() -> None:
    mapping = _mapping()
    phones = mapping["phones"]

    assert mapping["canonical_layer"] == "phones_ipa_articulatory_features"
    assert mapping["source_phone_set"] == "IPA_cv19_v3"
    assert mapping["normalization"]["length_mark"] == {
        "source": ":",
        "canonical": "\u02d0",
        "lossless": True,
    }
    assert phones["\u014b"] == "\u014b"
    assert phones["g"] == phones["\u0261"] == "\u0261"
    assert phones["\u0261"] == "\u0261"
    assert phones["\u0281"] == "\u0281"
    assert phones["\u0250"] == "\u0250"
    assert phones["\u014b"] != "N"
    assert phones["\u0281"] != "R"
    assert phones["\u0250"] != "6"


def test_cv19_v3_silence_and_oov_are_not_ipa_truth() -> None:
    mapping = _mapping()

    assert mapping["silence"] == ["<eps>", "sil", "O"]
    assert mapping["oov"] == ["spn"]
    assert _canonical_phone("sil", mapping) is None
    assert _canonical_phone("spn", mapping) is None
    assert all(label not in mapping["phones"] for label in mapping["silence"] + mapping["oov"])


def test_unknown_source_phone_is_a_structured_hard_failure() -> None:
    with pytest.raises(ForcedPhoneAlignerError) as exc_info:
        _canonical_phone("X-SAMPA-UNKNOWN", _mapping())

    assert exc_info.value.error_kind == "unsupported_forced_phone_set"
    assert "not present in the checked-in phone map" in str(exc_info.value)
