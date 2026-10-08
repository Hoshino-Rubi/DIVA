from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path

import pytest

from any2xsampa.adapters.forced_aligner import (
    MACHINE_REVIEW_DRAFT_STATUS,
    PENDING_HUMAN_PHONE_REVIEW_STATUS,
    ForcedPhoneAlignerError,
    parse_forced_phone_alignment,
)
from any2xsampa.adapters.mfa_forced_aligner import (
    MFA_MANIFEST_SCHEMA_VERSION,
    MFA_TIMING_POLICY,
    build_mfa_full_song_review_draft,
    build_mfa_timeline_constrained_review_draft,
    convert_mfa_alignment_review_draft,
    load_and_convert_mfa_alignment,
    phone_map_sha256,
)
from any2xsampa.cantonese.g2p import (
    ReviewedLexiconCantoneseG2PAdapter,
    resolve_lyrics_timeline,
)
from any2xsampa.lyrics.timeline import (
    LyricsTimeline,
    LyricsTimelineItem,
    canonical_timeline_sha256,
    sha256_file,
)
from tools.build_mfa_phone_review_draft import build_review_draft


def _fixture_paths() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    return (
        root / ".state/runs/forced-aligner-smoke/clip.wav",
        root / ".state/runs/forced-aligner-smoke/mfa_alignment.json",
        root / ".state/runs/forced-aligner-resources/HKCantonese_models/cv19_v3/canto_model_v3/final.mdl",
    )


def _dictionary_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / ".state/runs/forced-aligner-resources/HKCantonese_models/cv19_val_lexicon_v3_project_reviewed.dict"


def _strict_alignment_payload(alignment: Path, lyric_count: int = 10) -> dict[str, object]:
    """Trim the recovered smoke artifact to its declared lyric timeline."""
    payload = json.loads(alignment.read_text(encoding="utf-8"))
    silence = {"<eps>", "sil", "O"}
    words = payload["tiers"]["words"]["entries"]
    kept_words: list[list[object]] = []
    usable_count = 0
    last_lyric_end = 0.0
    for entry in words:
        if entry[2] in silence:
            kept_words.append(entry)
        elif usable_count < lyric_count:
            kept_words.append(entry)
            usable_count += 1
            last_lyric_end = float(entry[1])
    if usable_count != lyric_count:
        raise AssertionError(f"fixture has {usable_count} usable words, expected {lyric_count}")
    phones = payload["tiers"]["phones"]["entries"]
    payload["tiers"]["words"]["entries"] = kept_words
    payload["tiers"]["phones"]["entries"] = [
        entry
        for entry in phones
        if entry[2] in silence or float(entry[1]) <= last_lyric_end + 1e-9
    ]
    return payload


def _timeline(audio: Path, alignment: Path) -> object:
    payload = json.loads(alignment.read_text(encoding="utf-8"))
    words = [
        entry
        for entry in payload["tiers"]["words"]["entries"]
        if entry[2] not in {"<eps>", "sil", "spn", "O"}
    ][:10]
    source = json.loads(
        (audio.parents[0] / "mfa_clip_timeline.json").read_text(encoding="utf-8")
    )["items"]
    return LyricsTimeline(
        hashlib.sha256(audio.read_bytes()).hexdigest(),
        "manual_alignment",
        tuple(
            LyricsTimelineItem(
                index=index,
                text=source[index]["text"],
                start_sec=float(entry[0]),
                end_sec=float(entry[1]),
                jyutping=source[index]["jyutping"],
            )
            for index, entry in enumerate(words)
        ),
        {"test": "mfa"},
    )


def _manifest(
    alignment: Path,
    audio: Path,
    timeline: LyricsTimeline,
    model: Path,
    dictionary: Path,
    revision: str,
) -> dict[str, object]:
    return {
        "schema_version": MFA_MANIFEST_SCHEMA_VERSION,
        "alignment_json_sha256": sha256_file(alignment),
        "vocal_stem_sha256": sha256_file(audio),
        "lyrics_timeline_sha256": canonical_timeline_sha256(timeline),
        "model_sha256": sha256_file(model),
        "model_revision": revision,
        "dictionary_sha256": sha256_file(dictionary),
        "phone_map_schema": "mfa_cv19_v3_phone_map_v1",
        "phone_map_sha256": phone_map_sha256(),
    }


def test_mfa_json_converts_to_hash_bound_forced_phone_evidence(tmp_path: Path) -> None:
    audio, alignment, model = _fixture_paths()
    strict_alignment = tmp_path / "strict.json"
    strict_alignment.write_text(
        json.dumps(_strict_alignment_payload(alignment), ensure_ascii=False), encoding="utf-8"
    )
    timeline = _timeline(audio, strict_alignment)
    resolved, _ = resolve_lyrics_timeline(timeline, adapter=ReviewedLexiconCantoneseG2PAdapter())
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    result = load_and_convert_mfa_alignment(
        strict_alignment,
        vocal_stem=audio,
        timeline=timeline,
        resolved_items=resolved,
        model_hash=model_hash,
        model_revision="HKCantonese_models@fixture/cv19_v3",
    )
    assert result.metadata["alignment_source"] == "mfa_hkcantonese_cv19_v3"
    assert result.metadata["source_mode"] == "fixture"
    assert result.metadata["trusted"] is False
    assert result.to_dict()["model_output"] is False
    assert result.metadata["covered_lyric_item_count"] == 10
    assert len(result.segments) > 0
    assert result.segments[0].metadata["source_phone_set"] == "HKCantonese_models_cv19_v3"
    assert len(phone_map_sha256()) == 64


def test_full_song_review_draft_preserves_raw_phone_timing_and_stays_pending(
    tmp_path: Path,
) -> None:
    audio, alignment, model = _fixture_paths()
    payload = _strict_alignment_payload(alignment)
    timeline = _timeline(audio, alignment)
    first = timeline.items[0]
    drifted_timeline = LyricsTimeline(
        timeline.audio_sha256,
        timeline.source,
        (
            LyricsTimelineItem(
                first.index,
                first.text,
                first.start_sec + 0.01,
                first.end_sec,
                first.jyutping,
            ),
            *timeline.items[1:],
        ),
        timeline.provenance,
    )

    draft = build_mfa_full_song_review_draft(
        payload,
        vocal_stem=audio,
        timeline=drifted_timeline,
        model_hash=sha256_file(model),
        model_revision="HKCantonese_models@fixture/cv19_v3",
        alignment_sha256="a" * 64,
    )

    raw_phone = next(
        entry
        for entry in payload["tiers"]["phones"]["entries"]  # type: ignore[index]
        if entry[2] not in {"<eps>", "sil", "O", "spn"}
    )
    first_segment = draft["segments"][0]
    assert first_segment["start_sec"] == raw_phone[0]
    assert first_segment["end_sec"] == raw_phone[1]
    assert first_segment["metadata"]["source_phone"] == raw_phone[2]
    assert draft["model_output"] is True
    assert draft["status"] == MACHINE_REVIEW_DRAFT_STATUS
    assert draft["product_accepted"] is False
    assert draft["machine_suggestion"] is True
    assert draft["provenance"]["product_accepted"] is False
    assert draft["provenance"]["status"] == MACHINE_REVIEW_DRAFT_STATUS
    assert draft["provenance"]["boundary_transform_applied"] is False
    assert draft["provenance"]["timing_comparison"]["within_policy"] is False
    assert draft["review"]["status"] == PENDING_HUMAN_PHONE_REVIEW_STATUS
    assert len(draft["segments"]) > 0


def test_timeline_constrained_review_draft_maps_phones_and_keeps_raw_provenance() -> None:
    audio, alignment, model = _fixture_paths()
    payload = _strict_alignment_payload(alignment)
    source_items = json.loads(
        (audio.parent / "mfa_clip_timeline.json").read_text(encoding="utf-8")
    )["items"]
    source_timeline = _timeline(audio, alignment)
    timeline = LyricsTimeline(
        source_timeline.audio_sha256,
        source_timeline.source,
        tuple(
            LyricsTimelineItem(
                index=index,
                text=item["text"],
                start_sec=float(item["start_sec"]),
                end_sec=float(item["end_sec"]),
                jyutping=item["jyutping"],
            )
            for index, item in enumerate(source_items)
        ),
        source_timeline.provenance,
    )
    dictionary = _dictionary_path()
    draft = build_mfa_timeline_constrained_review_draft(
        payload,
        vocal_stem=audio,
        timeline=timeline,
        model_hash=sha256_file(model),
        model_revision="HKCantonese_models@fixture/cv19_v3",
        dictionary_path=dictionary,
        alignment_sha256="a" * 64,
    )

    raw_phones = [
        entry
        for entry in payload["tiers"]["phones"]["entries"]  # type: ignore[index]
        if entry[2] not in {"<eps>", "sil", "O", "spn"}
    ]
    segments = draft["segments"]
    assert [segment["metadata"]["source_phone"] for segment in segments] == [
        entry[2] for entry in raw_phones
    ]
    assert [segment["metadata"]["raw_start_sec"] for segment in segments] == [
        entry[0] for entry in raw_phones
    ]
    assert [segment["metadata"]["raw_end_sec"] for segment in segments] == [
        entry[1] for entry in raw_phones
    ]

    first_raw = raw_phones[0]
    first_word = payload["tiers"]["words"]["entries"][1]  # type: ignore[index]
    target_item = timeline.items[0]
    source_duration = first_word[1] - first_word[0]
    target_duration = target_item.end_sec - target_item.start_sec
    expected_start = target_item.start_sec + (first_raw[0] - first_word[0]) / source_duration * target_duration
    expected_end = target_item.start_sec + (first_raw[1] - first_word[0]) / source_duration * target_duration
    assert segments[0]["start_sec"] == pytest.approx(expected_start)
    assert segments[0]["end_sec"] == pytest.approx(expected_end)
    assert segments[0]["start_sec"] != first_raw[0]

    assert draft["model_output"] is True
    assert draft["status"] == MACHINE_REVIEW_DRAFT_STATUS
    assert draft["product_accepted"] is False
    assert draft["review"]["status"] == PENDING_HUMAN_PHONE_REVIEW_STATUS
    assert draft["review"]["machine_suggestion"] is True
    assert draft["boundary_transform_applied"] is True
    assert draft["machine_suggestion"] is True
    provenance = draft["provenance"]
    assert provenance["product_accepted"] is False
    assert provenance["status"] == MACHINE_REVIEW_DRAFT_STATUS
    assert provenance["boundary_transform_applied"] is True
    assert provenance["machine_suggestion"] is True
    assert provenance["raw_mfa_alignment_sha256"] == "a" * 64
    assert provenance["raw_alignment_sha256"] == "a" * 64
    assert provenance["timing_comparison"]["within_policy"] is False
    assert provenance["target_timing_comparison"]["within_policy"] is True
    assert draft["vocal_stem_sha256"] == sha256_file(audio)
    assert draft["lyrics_timeline_sha256"] == canonical_timeline_sha256(timeline)
    assert draft["model_hash"] == sha256_file(model)
    assert provenance["dictionary_sha256"] == sha256_file(dictionary)
    assert provenance["phone_map_sha256"] == phone_map_sha256()

    for previous, current in pairwise(segments):
        assert previous["end_sec"] <= current["start_sec"] + 1e-9

    resolved, _ = resolve_lyrics_timeline(
        timeline, adapter=ReviewedLexiconCantoneseG2PAdapter()
    )
    with pytest.raises(
        ForcedPhoneAlignerError,
        match="machine phone-review drafts cannot be parsed",
    ):
        parse_forced_phone_alignment(
            draft,
            vocal_stem=audio,
            timeline=timeline,
            resolved_items=resolved,
        )


def test_convert_mfa_alignment_review_draft_alias_preserves_pending_contract() -> None:
    audio, alignment, model = _fixture_paths()
    payload = _strict_alignment_payload(alignment)
    timeline = _timeline(audio, alignment)

    draft = convert_mfa_alignment_review_draft(
        payload,
        vocal_stem=audio,
        timeline=timeline,
        model_hash=sha256_file(model),
        model_revision="HKCantonese_models@fixture/cv19_v3",
        alignment_sha256="b" * 64,
    )

    assert draft["status"] == MACHINE_REVIEW_DRAFT_STATUS
    assert draft["product_accepted"] is False
    assert draft["review"]["status"] == PENDING_HUMAN_PHONE_REVIEW_STATUS
    assert draft["provenance"]["raw_mfa_alignment_sha256"] == "b" * 64


def test_review_draft_cli_keeps_raw_boundaries_by_default(tmp_path: Path) -> None:
    audio, alignment, model = _fixture_paths()
    payload = _strict_alignment_payload(alignment)
    alignment_path = tmp_path / "strict.json"
    alignment_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    timeline = LyricsTimeline(
        _timeline(audio, alignment).audio_sha256,
        "manual_alignment",
        tuple(
            LyricsTimelineItem(
                index=index,
                text=item["text"],
                start_sec=float(item["start_sec"]),
                end_sec=float(item["end_sec"]),
                jyutping=item["jyutping"],
            )
            for index, item in enumerate(
                json.loads(
                    (audio.parent / "mfa_clip_timeline.json").read_text(encoding="utf-8")
                )["items"]
            )
        ),
        {"test": "mfa-cli"},
    )
    timeline_path = tmp_path / "timeline.json"
    timeline_path.write_text(json.dumps(timeline.to_dict(), ensure_ascii=False), encoding="utf-8")
    model_revision = "HKCantonese_models@fixture/cv19_v3"
    common = {
        "alignment_path": alignment_path,
        "vocal_stem_path": audio,
        "timeline_path": timeline_path,
        "model_path": model,
        "model_revision": model_revision,
        "dictionary_path": _dictionary_path(),
    }
    raw_output = tmp_path / "raw.json"
    constrained_output = tmp_path / "constrained.json"
    build_review_draft(**common, output_path=raw_output)
    build_review_draft(**common, output_path=constrained_output, timeline_constrained=True)
    raw_draft = json.loads(raw_output.read_text(encoding="utf-8"))
    constrained_draft = json.loads(constrained_output.read_text(encoding="utf-8"))
    raw_phone = next(
        entry
        for entry in payload["tiers"]["phones"]["entries"]  # type: ignore[index]
        if entry[2] not in {"<eps>", "sil", "O", "spn"}
    )
    assert raw_draft["segments"][0]["start_sec"] == raw_phone[0]
    assert raw_draft["provenance"]["boundary_transform_applied"] is False
    assert constrained_draft["provenance"]["boundary_transform_applied"] is True
    assert constrained_draft["segments"][0]["start_sec"] != raw_phone[0]


def test_mfa_trusted_manifest_binds_alignment_and_resources(tmp_path: Path) -> None:
    audio, alignment, model = _fixture_paths()
    strict_alignment = tmp_path / "strict-trusted.json"
    strict_alignment.write_text(
        json.dumps(_strict_alignment_payload(alignment), ensure_ascii=False), encoding="utf-8"
    )
    timeline = _timeline(audio, strict_alignment)
    resolved, _ = resolve_lyrics_timeline(timeline, adapter=ReviewedLexiconCantoneseG2PAdapter())
    dictionary = _dictionary_path()
    revision = "HKCantonese_models@fixture/cv19_v3"
    manifest = _manifest(strict_alignment, audio, timeline, model, dictionary, revision)
    result = load_and_convert_mfa_alignment(
        strict_alignment,
        vocal_stem=audio,
        timeline=timeline,
        resolved_items=resolved,
        model_hash=sha256_file(model),
        model_revision=revision,
        manifest=manifest,
        trusted=True,
        dictionary_path=dictionary,
    )
    assert result.metadata["source_mode"] == "real"
    assert result.metadata["trusted"] is True
    assert result.to_dict()["model_output"] is True
    assert result.metadata["provenance"]["manifest_schema"] == MFA_MANIFEST_SCHEMA_VERSION
    assert result.metadata["model"]["repository_license"] == "MIT"
    assert result.metadata["model"]["license_status"] == "local_only_unverified_model_data_rights"
    assert result.metadata["model"]["distribution_allowed"] is False
    assert "license_id" not in result.metadata["model"]


def test_timeline_constrained_mfa_policy_is_validated_and_preserved(tmp_path: Path) -> None:
    audio, alignment, model = _fixture_paths()
    payload = _strict_alignment_payload(alignment)
    payload["provenance"] = {
        "timeline_constraint": "manual_lyrics_item_bounds_with_mfa_internal_phone_boundaries",
        "boundary_policy": "close_first_phone_to_item_start_and_last_phone_to_item_end",
        "interpolation_policy": "edge_closure_only_explicitly_recorded",
    }
    constrained_alignment = tmp_path / "timeline-constrained.json"
    constrained_alignment.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    timeline = _timeline(audio, constrained_alignment)
    resolved, _ = resolve_lyrics_timeline(
        timeline, adapter=ReviewedLexiconCantoneseG2PAdapter()
    )
    dictionary = _dictionary_path()
    revision = "HKCantonese_models@fixture/cv19_v3"
    manifest = _manifest(
        constrained_alignment, audio, timeline, model, dictionary, revision
    )
    manifest.update(
        {
            "manifest_kind": "trusted_timeline_constrained_mfa_alignment",
            "validation": {
                "exact_lyric_order": True,
                "timeline_constrained": True,
                "boundary_policy": "close_first_phone_to_item_start_and_last_phone_to_item_end",
            },
        }
    )

    result = load_and_convert_mfa_alignment(
        constrained_alignment,
        vocal_stem=audio,
        timeline=timeline,
        resolved_items=resolved,
        model_hash=sha256_file(model),
        model_revision=revision,
        manifest=manifest,
        trusted=True,
        dictionary_path=dictionary,
    )

    transform = result.metadata["provenance"]["alignment_transform"]
    assert transform["boundary_transform_applied"] is True
    assert transform["uniform_duration_inference"] is False
    assert transform["review_status"] == "mechanically_validated_pending_listening_review"
    assert result.to_dict()["provenance"]["alignment_transform"] == transform


def test_timeline_constrained_mfa_policy_mismatch_is_a_hard_failure(tmp_path: Path) -> None:
    audio, alignment, model = _fixture_paths()
    payload = _strict_alignment_payload(alignment)
    payload["provenance"] = {
        "timeline_constraint": "manual_lyrics_item_bounds_with_mfa_internal_phone_boundaries",
        "boundary_policy": "unsupported_boundary_policy",
        "interpolation_policy": "edge_closure_only_explicitly_recorded",
    }
    constrained_alignment = tmp_path / "bad-timeline-constraint.json"
    constrained_alignment.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    timeline = _timeline(audio, constrained_alignment)
    resolved, _ = resolve_lyrics_timeline(
        timeline, adapter=ReviewedLexiconCantoneseG2PAdapter()
    )
    dictionary = _dictionary_path()
    revision = "fixture"
    manifest = _manifest(
        constrained_alignment, audio, timeline, model, dictionary, revision
    )
    manifest.update(
        {
            "manifest_kind": "trusted_timeline_constrained_mfa_alignment",
            "validation": {
                "exact_lyric_order": True,
                "timeline_constrained": True,
                "boundary_policy": "close_first_phone_to_item_start_and_last_phone_to_item_end",
            },
        }
    )

    with pytest.raises(ForcedPhoneAlignerError, match="complete supported boundary policy"):
        load_and_convert_mfa_alignment(
            constrained_alignment,
            vocal_stem=audio,
            timeline=timeline,
            resolved_items=resolved,
            model_hash=sha256_file(model),
            model_revision=revision,
            manifest=manifest,
            trusted=True,
            dictionary_path=dictionary,
        )


def test_mfa_trusted_mode_requires_manifest(tmp_path: Path) -> None:
    audio, alignment, model = _fixture_paths()
    strict_alignment = tmp_path / "strict-missing-manifest.json"
    strict_alignment.write_text(
        json.dumps(_strict_alignment_payload(alignment), ensure_ascii=False), encoding="utf-8"
    )
    timeline = _timeline(audio, strict_alignment)
    resolved, _ = resolve_lyrics_timeline(timeline, adapter=ReviewedLexiconCantoneseG2PAdapter())
    with pytest.raises(ForcedPhoneAlignerError, match="manifest"):
        load_and_convert_mfa_alignment(
            strict_alignment,
            vocal_stem=audio,
            timeline=timeline,
            resolved_items=resolved,
            model_hash=sha256_file(model),
            model_revision="fixture",
            trusted=True,
        )


def test_mfa_manifest_hash_and_required_fields_are_hard_failures(tmp_path: Path) -> None:
    audio, alignment, model = _fixture_paths()
    strict_alignment = tmp_path / "strict-manifest-checks.json"
    strict_alignment.write_text(
        json.dumps(_strict_alignment_payload(alignment), ensure_ascii=False), encoding="utf-8"
    )
    timeline = _timeline(audio, strict_alignment)
    resolved, _ = resolve_lyrics_timeline(timeline, adapter=ReviewedLexiconCantoneseG2PAdapter())
    dictionary = _dictionary_path()
    revision = "fixture"
    base = _manifest(strict_alignment, audio, timeline, model, dictionary, revision)
    base["alignment_json_sha256"] = "0" * 64
    with pytest.raises(ForcedPhoneAlignerError, match="alignment_json_sha256"):
        load_and_convert_mfa_alignment(
            strict_alignment,
            vocal_stem=audio,
            timeline=timeline,
            resolved_items=resolved,
            model_hash=sha256_file(model),
            model_revision=revision,
            manifest=base,
            trusted=True,
            dictionary_path=dictionary,
        )

    missing_dictionary = _manifest(strict_alignment, audio, timeline, model, dictionary, revision)
    missing_dictionary.pop("dictionary_sha256")
    with pytest.raises(ForcedPhoneAlignerError, match="dictionary_sha256"):
        load_and_convert_mfa_alignment(
            strict_alignment,
            vocal_stem=audio,
            timeline=timeline,
            resolved_items=resolved,
            model_hash=sha256_file(model),
            model_revision=revision,
            manifest=missing_dictionary,
            trusted=True,
            dictionary_path=dictionary,
        )

    conflicting_map = _manifest(strict_alignment, audio, timeline, model, dictionary, revision)
    conflicting_map["phone_map"] = {"schema": "wrong", "sha256": phone_map_sha256()}
    with pytest.raises(ForcedPhoneAlignerError, match="phone_map_schema"):
        load_and_convert_mfa_alignment(
            strict_alignment,
            vocal_stem=audio,
            timeline=timeline,
            resolved_items=resolved,
            model_hash=sha256_file(model),
            model_revision=revision,
            manifest=conflicting_map,
            trusted=True,
            dictionary_path=dictionary,
        )


def test_mfa_unknown_phone_is_a_structured_gate_failure(tmp_path: Path) -> None:
    audio, alignment, model = _fixture_paths()
    payload = _strict_alignment_payload(alignment)
    payload["tiers"]["phones"]["entries"][1][2] = "unknown-phone"
    altered = tmp_path / "unknown.json"
    altered.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    timeline = _timeline(audio, alignment)
    resolved, _ = resolve_lyrics_timeline(timeline, adapter=ReviewedLexiconCantoneseG2PAdapter())
    with pytest.raises(ForcedPhoneAlignerError, match="not present in the checked-in phone map"):
        load_and_convert_mfa_alignment(
            altered,
            vocal_stem=audio,
            timeline=timeline,
            resolved_items=resolved,
            model_hash=hashlib.sha256(model.read_bytes()).hexdigest(),
            model_revision="fixture",
        )


def test_mfa_extra_word_token_is_a_hard_failure(tmp_path: Path) -> None:
    audio, alignment, model = _fixture_paths()
    payload = json.loads(alignment.read_text(encoding="utf-8"))
    altered = tmp_path / "extra-word.json"
    altered.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    timeline = _timeline(audio, altered)
    resolved, _ = resolve_lyrics_timeline(timeline, adapter=ReviewedLexiconCantoneseG2PAdapter())
    with pytest.raises(ForcedPhoneAlignerError) as exc_info:
        load_and_convert_mfa_alignment(
            altered,
            vocal_stem=audio,
            timeline=timeline,
            resolved_items=resolved,
            model_hash=hashlib.sha256(model.read_bytes()).hexdigest(),
            model_revision="fixture",
        )
    assert exc_info.value.error_kind == "invalid_forced_phone_evidence"
    assert "extra usable token" in str(exc_info.value)


def test_mfa_unknown_word_token_is_a_hard_failure(tmp_path: Path) -> None:
    audio, alignment, model = _fixture_paths()
    payload = _strict_alignment_payload(alignment)
    payload["tiers"]["words"]["entries"][1][2] = "<unk>"
    altered = tmp_path / "unknown-word.json"
    altered.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    timeline = _timeline(audio, alignment)
    resolved, _ = resolve_lyrics_timeline(timeline, adapter=ReviewedLexiconCantoneseG2PAdapter())
    with pytest.raises(ForcedPhoneAlignerError, match="unknown token"):
        load_and_convert_mfa_alignment(
            altered,
            vocal_stem=audio,
            timeline=timeline,
            resolved_items=resolved,
            model_hash=hashlib.sha256(model.read_bytes()).hexdigest(),
            model_revision="fixture",
        )


def test_mfa_metadata_hash_and_word_timing_are_hard_failures(tmp_path: Path) -> None:
    audio, alignment, model = _fixture_paths()
    payload = _strict_alignment_payload(alignment)
    payload["schema_version"] = "mfa_alignment_json_v0"
    altered = tmp_path / "bad-schema.json"
    altered.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    timeline = _timeline(audio, alignment)
    resolved, _ = resolve_lyrics_timeline(timeline, adapter=ReviewedLexiconCantoneseG2PAdapter())
    with pytest.raises(ForcedPhoneAlignerError, match="schema"):
        load_and_convert_mfa_alignment(
            altered,
            vocal_stem=audio,
            timeline=timeline,
            resolved_items=resolved,
            model_hash=hashlib.sha256(model.read_bytes()).hexdigest(),
            model_revision="fixture",
        )

    payload = _strict_alignment_payload(alignment)
    payload["tiers"]["words"]["entries"][1][1] = 0.129
    altered = tmp_path / "bad-word-timing.json"
    altered.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ForcedPhoneAlignerError, match="word timing"):
        load_and_convert_mfa_alignment(
            altered,
            vocal_stem=audio,
            timeline=timeline,
            resolved_items=resolved,
            model_hash=hashlib.sha256(model.read_bytes()).hexdigest(),
            model_revision="fixture",
        )


def test_mfa_silence_phone_inside_lyric_word_is_not_dropped(tmp_path: Path) -> None:
    audio, alignment, model = _fixture_paths()
    payload = _strict_alignment_payload(alignment)
    payload["tiers"]["phones"]["entries"][1][2] = "sil"
    altered = tmp_path / "silence-in-word.json"
    altered.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    timeline = _timeline(audio, altered)
    resolved, _ = resolve_lyrics_timeline(timeline, adapter=ReviewedLexiconCantoneseG2PAdapter())
    with pytest.raises(ForcedPhoneAlignerError, match="silence/OOV phone"):
        load_and_convert_mfa_alignment(
            altered,
            vocal_stem=audio,
            timeline=timeline,
            resolved_items=resolved,
            model_hash=hashlib.sha256(model.read_bytes()).hexdigest(),
            model_revision="fixture",
        )


def test_mfa_declared_source_offset_must_match_call(tmp_path: Path) -> None:
    audio, alignment, model = _fixture_paths()
    payload = _strict_alignment_payload(alignment)
    payload["source_offset_sec"] = 1.0
    altered = tmp_path / "bad-offset.json"
    altered.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    timeline = _timeline(audio, altered)
    resolved, _ = resolve_lyrics_timeline(timeline, adapter=ReviewedLexiconCantoneseG2PAdapter())
    with pytest.raises(ForcedPhoneAlignerError, match="source offset") as exc_info:
        load_and_convert_mfa_alignment(
            altered,
            vocal_stem=audio,
            timeline=timeline,
            resolved_items=resolved,
            model_hash=hashlib.sha256(model.read_bytes()).hexdigest(),
            model_revision="fixture",
        )
    assert exc_info.value.details["timing_policy"] == MFA_TIMING_POLICY.to_dict()
    assert exc_info.value.details["transform_applied"] is False
    assert exc_info.value.details["source_offset_delta_sec"] == pytest.approx(1.0)


def test_mfa_timing_policy_records_comparison_and_allows_float_noise(tmp_path: Path) -> None:
    audio, alignment, model = _fixture_paths()
    payload = _strict_alignment_payload(alignment)
    # MFA JSON round-tripping can move a boundary by a fraction of a
    # microsecond; this stays within the documented representation bound.
    payload["tiers"]["words"]["entries"][1][0] += 5e-7
    altered = tmp_path / "float-noise.json"
    altered.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    timeline = _timeline(audio, alignment)
    resolved, _ = resolve_lyrics_timeline(timeline, adapter=ReviewedLexiconCantoneseG2PAdapter())

    result = load_and_convert_mfa_alignment(
        altered,
        vocal_stem=audio,
        timeline=timeline,
        resolved_items=resolved,
        model_hash=hashlib.sha256(model.read_bytes()).hexdigest(),
        model_revision="fixture",
    )

    comparison = result.metadata["timing_comparison"]
    assert result.metadata["timing_policy"] == MFA_TIMING_POLICY.to_dict()
    assert result.metadata["provenance"]["timing_policy"] == MFA_TIMING_POLICY.to_dict()
    assert comparison["within_policy"] is True
    assert comparison["mismatch_count"] == 0
    assert comparison["word_count"] == 10
    assert len(comparison["rows"]) == 10
    assert comparison["max_abs_start_delta_sec"] == pytest.approx(5e-7)


def test_mfa_timing_policy_rejects_large_drift_with_structured_differences(tmp_path: Path) -> None:
    audio, alignment, model = _fixture_paths()
    strict_alignment = tmp_path / "strict-large-drift.json"
    strict_alignment.write_text(
        json.dumps(_strict_alignment_payload(alignment), ensure_ascii=False), encoding="utf-8"
    )
    timeline = _timeline(audio, strict_alignment)
    resolved, _ = resolve_lyrics_timeline(timeline, adapter=ReviewedLexiconCantoneseG2PAdapter())

    with pytest.raises(ForcedPhoneAlignerError) as exc_info:
        load_and_convert_mfa_alignment(
            strict_alignment,
            vocal_stem=audio,
            timeline=timeline,
            resolved_items=resolved,
            model_hash=hashlib.sha256(model.read_bytes()).hexdigest(),
            model_revision="fixture",
            source_offset_sec=14.0,
        )

    error = exc_info.value
    comparison = error.details["timing_comparison"]
    assert error.error_kind == "invalid_forced_phone_evidence"
    assert error.details["timing_policy"] == MFA_TIMING_POLICY.to_dict()
    assert comparison["within_policy"] is False
    assert comparison["transform_applied"] is False
    assert comparison["mismatch_count"] == 10
    assert comparison["mismatch_lyric_item_indices"] == list(range(10))
    assert comparison["max_abs_start_delta_sec"] > 13.0
    assert comparison["max_abs_end_delta_sec"] > 13.0
    assert len(comparison["rows"]) == 10
    assert error.details["lyric_item_index"] == 0
