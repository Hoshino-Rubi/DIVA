"""Synthetic contract tests; no models, audio, or reference labels."""

from copy import deepcopy
import json

import pytest

from any2xsampa.adapters.cantonese4_lexicon import (
    choose_prediction,
    completion_gate,
    convert_jyutping_output,
    project,
)


COMPLETE = {"status": "ok", "eos_observed": True,
            "truncated_or_incomplete": False, "truncation_status": "complete_with_eos"}


@pytest.mark.parametrize("change", [
    {"status": "failed"}, {"status": "unknown"},
    {"eos_observed": False}, {"eos_observed": None}, {"eos_observed": 1},
    {"truncated_or_incomplete": True}, {"truncated_or_incomplete": None},
    {"truncated_or_incomplete": 0}, {"truncation_status": None},
    {"truncation_status": "unknown_if_forced_eos_at_limit"},
    {"truncation_status": "cap_reached"},
])
def test_completion_rejects_failure_unknown_and_non_boolean_flags(change):
    raw = {**COMPLETE, **change, "decoded": "la la HnG"}
    assert not completion_gate(raw)
    choice = choose_prediction(raw, COMPLETE)
    assert choice["selected_model"] == "base"
    assert choice["reason"] == "fallback_base_complete"
    assert choice["completion_gate"]["small"] == {
        "passed": False, "observed": {key: raw[key] for key in COMPLETE},
    }


@pytest.mark.parametrize("missing", COMPLETE)
def test_no_missing_completion_field_is_assumed(missing):
    raw = dict(COMPLETE)
    del raw[missing]
    assert not completion_gate(raw)
    assert choose_prediction(raw, None)["selected_model"] is None


@pytest.mark.parametrize("raw", [None, {}, [], "ok", True])
def test_gate_rejects_absent_or_non_object_raw(raw):
    assert completion_gate(raw) is False


def test_small_precedence_ignores_content_quality_and_keeps_inputs():
    primary = {**COMPLETE, "decoded": "la la HnG", "score": -100}
    base = {**COMPLETE, "decoded": "nei hou", "score": 100}
    before = deepcopy((primary, base))
    result = choose_prediction(primary, base)
    assert result == {
        "selected_model": "small", "reason": "primary_complete",
        "completion_gate": {
            "small": {"passed": True, "observed": COMPLETE},
            "base": {"passed": True, "observed": COMPLETE},
        },
    }
    assert (primary, base) == before
    result["completion_gate"]["small"]["observed"]["status"] = "changed"
    assert primary["status"] == "ok"


def test_complete_empty_decode_is_still_primary():
    assert completion_gate({**COMPLETE, "decoded": ""})
    assert choose_prediction({**COMPLETE, "decoded": ""}, COMPLETE)["selected_model"] == "small"


@pytest.mark.parametrize("primary,base", [
    (None, None), ({}, {}),
    ({**COMPLETE, "status": "failed"}, {**COMPLETE, "eos_observed": None}),
])
def test_both_rejected_are_explicit_no_complete_prediction(primary, base):
    choice = choose_prediction(primary, base)
    assert choice["selected_model"] is None
    assert choice["reason"] == "no_complete_prediction"
    assert all(not side["passed"] for side in choice["completion_gate"].values())


def test_original_diagnostic_schema_and_untoned_evidence():
    result = convert_jyutping_output("NeI")
    assert set(result) == {"schema_version", "mapping_version", "mapping_sources", "input",
                           "status", "counts", "syllables", "unknown"}
    assert result["schema_version"] == "jyutping_acoustic.v1"
    assert result["mapping_version"] == "jyutping_acoustic_ipa.v1"
    assert result["status"] == "partial"
    assert result["counts"] == {"tokens": 1, "mapped": 1, "unknown": 0, "unknown_tone": 1}
    item, = result["syllables"]
    assert item == {
        "native_syllable": "NeI", "span": {"start_char": 0, "end_char": 3},
        "normalized": "nei", "status": "mapped", "canonicalIPA": "nei",
        "phones": ["n", "e", "i"],
        "parts": {"onset": "n", "rime": "ei", "syllabic_nasal": False},
        "tone": "unknown", "sourceIPA": {"pycantonese": "nei"},
        "phonetic_variants": [], "acoustic_repetition_evidence": None,
        "unknown": [{"reason": "tone_absent", "evidence": "NeI"}],
    }


def test_repetitions_keep_every_syllable_and_phone_occurrence():
    result = project("La la la")
    assert result["scored_tokens"] == ["lɐ"] * 3
    assert result["phones"] == ["l", "ɐ"] * 3
    assert [s["native"] for s in result["syllables"]] == ["La", "la", "la"]
    for item in result["conversion"]["syllables"]:
        assert item["status"] == "mapped"
        assert item["tone"] == item["acoustic_repetition_evidence"] == "unknown"
        assert item["unknown"][-1] == {
            "reason": "acoustic_repetition_unresolved", "evidence": [0, 1, 2],
        }


def test_punctuation_and_unknown_retain_original_token_indices_and_spans():
    text = "NeI，HnG hou! 3 + si"
    result = project(text)
    assert set(result) == {"syllables", "scored_tokens", "phones", "unknown_tokens",
                           "punctuation", "conversion"}
    assert result["punctuation"] == [{"index": 1, "native": "，"}, {"index": 4, "native": "!"}]
    assert result["scored_tokens"] == ["nei", "<UNK:2>", "hou", "<UNK:5>", "<UNK:6>", "siː"]
    assert result["unknown_tokens"] == ["<UNK:2>", "<UNK:5>", "<UNK:6>"]
    assert result["phones"] == ["n", "e", "i", "<UNK:2>", "h", "o", "u",
                                 "<UNK:5>", "<UNK:6>", "s", "iː"]
    for item in result["conversion"]["syllables"]:
        span = item["span"]
        assert text[span["start_char"]:span["end_char"]] == item["native_syllable"]


@pytest.mark.parametrize("text,punctuation,unknown", [
    ("，!?", [{"index": 0, "native": "，!?"}], []),
    ("!+", [], ["<UNK:0>"]), ("🙂", [], ["<UNK:0>"]),
])
def test_only_wholly_unicode_punctuation_tokens_are_exempt(text, punctuation, unknown):
    result = project(text)
    assert result["punctuation"] == punctuation
    assert result["unknown_tokens"] == result["phones"] == unknown
    assert result["conversion"]["syllables"][0]["native_syllable"] == text


def test_only_explicit_valid_tones_split_connected_runs():
    result = project("NeI5HoU2 sik1 si1si1")
    assert result["scored_tokens"] == ["nei", "hou", "sɪk̚", "siː", "siː"]
    assert [s["tone"] for s in result["conversion"]["syllables"]] == [5, 2, 1, 1, 1]


@pytest.mark.parametrize("text", ["neihou", "sisi", "si7", "si11", "你", "nei５", "ne\u0301i"])
def test_unsupported_tokens_are_not_repaired_or_silently_dropped(text):
    result = project(text)
    assert result["scored_tokens"] == result["unknown_tokens"] == ["<UNK:0>"]
    assert result["syllables"][0]["native"] == text
    assert result["conversion"]["syllables"][0]["canonicalIPA"] is None


@pytest.mark.parametrize("text", ["", " \t\n"])
def test_empty_projection_retains_empty_output_diagnostic(text):
    result = project(text)
    assert result["syllables"] == result["phones"] == result["scored_tokens"] == []
    assert result["conversion"]["unknown"] == [{"reason": "empty_output", "evidence": text}]


def test_variants_are_diagnostics_not_extra_phones_and_outputs_are_independent():
    first = project("cing")
    expected = deepcopy(first)
    assert first["phones"] == ["t͡sʰ", "ɪ", "ŋ"]
    item = first["conversion"]["syllables"][0]
    assert item["sourceIPA"] == {"pycantonese": "tsʰɪŋ"}
    assert [v["IPA"] for v in item["phonetic_variants"]] == ["t͡ʃʰɪŋ", "t͡sʰeŋ"]
    first["phones"].clear()
    item["phones"].clear()
    item["phonetic_variants"][0]["phones"].clear()
    first["conversion"]["mapping_sources"].clear()
    assert project("cing") == expected
    assert json.loads(json.dumps(expected)) == expected
    assert "duration" not in json.dumps(expected)
    assert set(item["span"]) == {"start_char", "end_char"}


@pytest.mark.parametrize("convert", [project, convert_jyutping_output])
def test_non_string_fails_explicitly(convert):
    with pytest.raises(TypeError, match="text must be str"):
        convert(None)
