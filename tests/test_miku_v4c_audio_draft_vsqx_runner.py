from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path
from xml.etree import ElementTree as ET

import tools.build_miku_v4c_audio_draft_vsqx as audio_runner
from any2xsampa.adapters.forced_aligner import (
    FixtureForcedPhoneAligner,
    ForcedAlignmentResult,
    lyrics_timeline_sha256,
)
from any2xsampa.adapters.forced_alignment_acceptance import (
    QUALIFIED_MODEL_ROUTE,
    canonical_phone_inventory_sha256,
    canonical_phone_segments_sha256,
)
from any2xsampa.adapters.forced_alignment_acceptance import (
    SCHEMA_VERSION as ACCEPTANCE_SCHEMA_VERSION,
)
from any2xsampa.adapters.mfa_forced_aligner import (
    MFA_MANIFEST_SCHEMA_VERSION,
    phone_map_sha256,
)
from any2xsampa.cantonese.g2p import resolve_lyrics_timeline
from any2xsampa.lyrics.timeline import load_lyrics_timeline, sha256_file
from any2xsampa.vocaloid.vsqx import VSQ4_NS
from tools.build_miku_v4c_audio_draft_vsqx import run_audio_draft_vsqx
from tools.review_forced_phone_alignment import (
    FULL_SONG_SCOPE,
    load_review_inputs,
    save_review,
)
from tools.verify_miku_v4c_vsqx_smoke import verify_product_provenance

REPO_ROOT = Path(__file__).resolve().parents[1]
NS = {"v": VSQ4_NS}


def _template(path: Path) -> Path:
    root = ET.Element(f"{{{VSQ4_NS}}}vsq4")
    ET.SubElement(root, f"{{{VSQ4_NS}}}vender").text = "Yamaha corporation"
    ET.SubElement(root, f"{{{VSQ4_NS}}}version").text = "4.0.0.3"
    table = ET.SubElement(root, f"{{{VSQ4_NS}}}vVoiceTable")
    voice = ET.SubElement(table, f"{{{VSQ4_NS}}}vVoice")
    ET.SubElement(voice, f"{{{VSQ4_NS}}}bs").text = "1"
    ET.SubElement(voice, f"{{{VSQ4_NS}}}pc").text = "11"
    ET.SubElement(voice, f"{{{VSQ4_NS}}}id").text = "template"
    ET.SubElement(voice, f"{{{VSQ4_NS}}}name").text = "MIKU_V4X_Original_EVEC"
    ET.SubElement(root, f"{{{VSQ4_NS}}}mixer")
    master = ET.SubElement(root, f"{{{VSQ4_NS}}}masterTrack")
    ET.SubElement(master, f"{{{VSQ4_NS}}}seqName").text = "template"
    ET.SubElement(master, f"{{{VSQ4_NS}}}comment").text = "template"
    ET.SubElement(master, f"{{{VSQ4_NS}}}resolution").text = "480"
    ET.SubElement(master, f"{{{VSQ4_NS}}}preMeasure").text = "1"
    ET.SubElement(master, f"{{{VSQ4_NS}}}timeSig")
    ET.SubElement(master, f"{{{VSQ4_NS}}}tempo")
    track = ET.SubElement(root, f"{{{VSQ4_NS}}}vsTrack")
    ET.SubElement(track, f"{{{VSQ4_NS}}}tNo").text = "0"
    ET.SubElement(track, f"{{{VSQ4_NS}}}name").text = "Track"
    ET.SubElement(track, f"{{{VSQ4_NS}}}comment").text = "Track"
    part = ET.SubElement(track, f"{{{VSQ4_NS}}}vsPart")
    ET.SubElement(part, f"{{{VSQ4_NS}}}t").text = "3840"
    ET.SubElement(part, f"{{{VSQ4_NS}}}playTime").text = "1920"
    ET.SubElement(part, f"{{{VSQ4_NS}}}name").text = "Part"
    ET.SubElement(part, f"{{{VSQ4_NS}}}comment").text = "Part"
    ET.SubElement(part, f"{{{VSQ4_NS}}}sPlug")
    ET.SubElement(part, f"{{{VSQ4_NS}}}singer")
    ET.SubElement(part, f"{{{VSQ4_NS}}}plane").text = "0"
    ET.SubElement(root, f"{{{VSQ4_NS}}}monoTrack")
    ET.SubElement(root, f"{{{VSQ4_NS}}}stTrack")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def _strict_template(path: Path) -> Path:
    path = _template(path)
    root = ET.parse(path).getroot()
    part = root.find(".//v:vsPart", NS)
    assert part is not None
    plane = part.find("v:plane", NS)
    assert plane is not None
    note = ET.Element(f"{{{VSQ4_NS}}}note")
    for name, value in (
        ("t", "0"),
        ("dur", "480"),
        ("n", "60"),
        ("v", "64"),
        ("y", "la"),
        ("p", "l a"),
    ):
        child = ET.SubElement(note, f"{{{VSQ4_NS}}}{name}")
        child.text = value
        if name == "p":
            child.set("lock", "1")
    ET.SubElement(note, f"{{{VSQ4_NS}}}nStyle")
    part.insert(list(part).index(plane), note)
    master = root.find("v:masterTrack", NS)
    assert master is not None
    tempo = master.find("v:tempo", NS)
    assert tempo is not None
    ET.SubElement(tempo, f"{{{VSQ4_NS}}}t").text = "0"
    ET.SubElement(tempo, f"{{{VSQ4_NS}}}v").text = "15000"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def _audio(path: Path) -> Path:
    path.write_bytes(b"RIFFfake-audio")
    return path


def _real_wav(path: Path, seconds: float = 2.0) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


def _strict_timeline(path: Path, audio: Path) -> Path:
    payload = {
        "schema_version": "lyrics_timeline_v1",
        "audio_sha256": sha256_file(audio),
        "source": "manual_alignment",
        "items": [
            {"index": 0, "text": "喜", "start_sec": 0.0, "end_sec": 0.5, "jyutping": "hei2"},
            {"index": 1, "text": "歡", "start_sec": 0.5, "end_sec": 1.0, "jyutping": "fun1"},
            {"index": 2, "text": "你", "start_sec": 1.0, "end_sec": 1.5, "jyutping": "nei5"},
        ],
        "provenance": {"timestamp_source": "manual_review"},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _mfa_fixture_paths() -> tuple[Path, Path, Path, Path]:
    resources = REPO_ROOT / ".state/runs/forced-aligner-resources/HKCantonese_models"
    smoke = REPO_ROOT / ".state/runs/forced-aligner-smoke"
    return (
        smoke / "clip.wav",
        smoke / "mfa_alignment.json",
        resources / "cv19_v3/canto_model_v3/final.mdl",
        resources / "cv19_val_lexicon_v3_project_reviewed.dict",
    )


def _mfa_alignment_for_cli(path: Path, source: Path) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    silence = {"<eps>", "sil", "O"}
    words = payload["tiers"]["words"]["entries"]
    kept_words: list[list[object]] = []
    usable_count = 0
    last_word_end = 0.0
    for entry in words:
        if entry[2] in silence:
            kept_words.append(entry)
        elif usable_count < 10:
            kept_words.append(entry)
            usable_count += 1
            last_word_end = float(entry[1])
    assert usable_count == 10
    payload["tiers"]["words"]["entries"] = kept_words
    payload["tiers"]["phones"]["entries"] = [
        entry
        for entry in payload["tiers"]["phones"]["entries"]
        if entry[2] in silence or float(entry[1]) <= last_word_end + 1e-9
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _mfa_manifest(
    path: Path,
    *,
    alignment: Path,
    audio: Path,
    timeline: Path,
    model: Path,
    dictionary: Path,
) -> Path:
    timeline_payload = load_lyrics_timeline(timeline, audio_path=audio)
    payload = {
        "schema_version": MFA_MANIFEST_SCHEMA_VERSION,
        "alignment_json_sha256": sha256_file(alignment),
        "vocal_stem_sha256": sha256_file(audio),
        "lyrics_timeline_sha256": lyrics_timeline_sha256(timeline_payload),
        "model_sha256": sha256_file(model),
        "model_revision": "HKCantonese_models@fixture/cv19_v3",
        "dictionary_sha256": sha256_file(dictionary),
        "phone_map_schema": "mfa_cv19_v3_phone_map_v1",
        "phone_map_sha256": phone_map_sha256(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _timeline_g2p_lexicon(path: Path, timeline: Path) -> Path:
    payload = json.loads(timeline.read_text(encoding="utf-8"))
    path.write_text(
        json.dumps(
            {"entries": {item["text"]: [item["jyutping"]] for item in payload["items"]}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _forced_alignment(path: Path, stem: Path, timeline_path: Path, audio: Path) -> Path:
    segments = []
    for index, (start, phone) in enumerate(((0.0, "e"), (0.5, "u"), (1.0, "i"))):
        segments.append(
            {
                "start_sec": start,
                "end_sec": start + 0.45,
                "normalized_ipa": phone,
                "confidence": 0.95,
                "candidate_phones": [{"symbol": phone, "confidence": 0.95}],
                "metadata": {
                    "alignment_source": "test_forced_aligner",
                    "lyric_item_index": index,
                    "direct_overlap_sec": 0.45,
                    "phone_overlap_ratio": 1.0,
                    "alignment_confidence": 0.95,
                },
            }
        )
    payload = {
        "schema_version": "forced_phone_alignment_v1",
        "vocal_stem_sha256": sha256_file(stem),
        "lyrics_timeline_sha256": lyrics_timeline_sha256(
            load_lyrics_timeline(timeline_path, audio_path=audio)
        ),
        "alignment_source": "test_forced_aligner",
        "model_output": True,
        "model_hash": "1" * 64,
        "segments": segments,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _qualified_model_acceptance(
    path: Path,
    *,
    alignment: Path,
    stem: Path,
    timeline_path: Path,
    audio: Path,
) -> Path:
    model_file = path.with_name("fixture-model.bin")
    dictionary_file = path.with_name("fixture-dictionary.txt")
    tokenizer_file = path.with_name("fixture-tokenizer.json")
    phone_map_file = path.with_name("fixture-phone-map.json")
    license_file = path.with_name("fixture-license.txt")
    training_file = path.with_name("fixture-training-evidence.json")
    model_file.write_bytes(b"fixture qualified model weights")
    dictionary_file.write_text("e e\nu u\n", encoding="utf-8")
    tokenizer_file.write_text('{"tokens":["e","u"]}\n', encoding="utf-8")
    phone_map_file.write_text('{"phones":[] }\n', encoding="utf-8")
    license_file.write_text("Apache License 2.0\n", encoding="utf-8")
    training_file.write_text(
        '{"claim":"trained_or_finetuned_on_cantonese_singing",'
        '"language":"Cantonese","domain":"singing_voice"}\n',
        encoding="utf-8",
    )
    model_hash = sha256_file(model_file)
    license_hash = sha256_file(license_file)
    training_hash = sha256_file(training_file)
    timeline = load_lyrics_timeline(timeline_path, audio_path=audio)
    resolved, _ = resolve_lyrics_timeline(timeline)
    alignment_payload = json.loads(alignment.read_text(encoding="utf-8"))
    alignment_payload["model_hash"] = model_hash
    alignment.write_text(json.dumps(alignment_payload), encoding="utf-8")
    forced = FixtureForcedPhoneAligner(alignment).align(stem, timeline, resolved)
    symbols = sorted({segment.normalized_ipa for segment in forced.segments})
    inventory_hash = canonical_phone_inventory_sha256(symbols)

    def artifact(kind: str, resource: Path, *, inventory: str | None = None) -> dict[str, object]:
        value: dict[str, object] = {
            "path": resource.name,
            "sha256": sha256_file(resource),
            "revision": "fixture-resource-revision",
            "revision_is_immutable": True,
            "license": {
                "identifier": "Apache-2.0",
                "applies_to": kind,
                "noncommercial_restriction": False,
                "commercial_use": "allowed",
                "source": license_file.name,
                "source_path": license_file.name,
                "source_sha256": license_hash,
            },
        }
        if inventory is not None:
            value["phone_inventory_sha256"] = inventory
        return value

    manifest = {
        "schema_version": "forced_phone_model_manifest_v1",
        "required_artifacts": ["model_weights", "dictionary", "tokenizer", "phone_map"],
        "artifacts": {
            "model_weights": artifact("model_weights", model_file),
            "dictionary": artifact("dictionary", dictionary_file),
            "tokenizer": artifact("tokenizer", tokenizer_file),
            "phone_map": artifact("phone_map", phone_map_file, inventory=inventory_hash),
        },
        "phone_inventory": {
            "symbols": symbols,
            "sha256": inventory_hash,
            "source": "fixture-phone-inventory.txt",
        },
        "training_evidence": {
            "path": training_file.name,
            "sha256": training_hash,
            "claim": "trained_or_finetuned_on_cantonese_singing",
            "language": "Cantonese",
            "domain": "singing_voice",
            "datasets": ["fixture-cantonese-singing-corpus"],
            "source_type": "primary",
            "source": training_file.name,
            "source_path": training_file.name,
            "source_revision": "fixture-training-revision",
            "source_sha256": training_hash,
        },
    }
    manifest_path = path.with_name("fixture-resource-manifest.json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    payload = {
        "schema_version": ACCEPTANCE_SCHEMA_VERSION,
        "route": QUALIFIED_MODEL_ROUTE,
        "alignment_artifact_sha256": sha256_file(alignment),
        "vocal_stem_sha256": sha256_file(stem),
        "phone_segments_sha256": canonical_phone_segments_sha256(forced.segments),
        "phone_inventory": {
            "symbols": symbols,
            "sha256": canonical_phone_inventory_sha256(symbols),
            "source": "fixture-phone-inventory@fixture-resource-revision",
        },
        "model": {
            "name": "fixture-cantonese-singing-aligner",
            "revision": "fixture-resource-revision",
            "revision_is_immutable": True,
            "sha256": model_hash,
            "resource_manifest_path": manifest_path.name,
            "resource_manifest_sha256": sha256_file(manifest_path),
            "license": {
                "identifier": "Apache-2.0",
                "applies_to": "model_weights",
                "noncommercial_restriction": False,
                "commercial_use": "allowed",
                "source": license_file.name,
                "source_sha256": license_hash,
            },
            "cantonese_singing_training": {
                "claim": "trained_or_finetuned_on_cantonese_singing",
                "language": "Cantonese",
                "domain": "singing_voice",
                "datasets": ["fixture-cantonese-singing-corpus"],
                "source_type": "primary",
                "source": training_file.name,
                "source_revision": "fixture-training-revision",
                "source_sha256": training_hash,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _pitch(path: Path) -> Path:
    payload = {
        "schema_version": "pitch_note_analysis_v1",
        "audio_path": "input.wav",
        "tempo_bpm": 120.0,
        "model_output": False,
        "analysis_output": True,
        "note_candidates": [
            {"note_id": "p0", "start_sec": 0.0, "duration_sec": 0.5, "midi": 64, "source": "rough_f0_quantized"},
            {"note_id": "p1", "start_sec": 0.5, "duration_sec": 0.5, "midi": 67, "source": "rough_f0_quantized"},
            {"note_id": "p2", "start_sec": 1.0, "duration_sec": 0.5, "midi": 69, "source": "rough_f0_quantized"},
        ],
        "provenance": {"asr_to_vsqx_role": "note_evidence", "full_mix_rough_evidence": True},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _human_review_request(source: dict[str, object]) -> dict[str, object]:
    return {
        "reviewer_id": "runner-integration-reviewer",
        "review_method": "auditory_acoustic_phone_boundary_review",
        "reviewed_at": "2026-08-15T12:00:00+08:00",
        "reviewed_range_sec": source["reviewed_range_sec"],
        "segments": [
            {
                "index": row["index"],
                "symbol": row["symbol"],
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "lyric_item_index": row["lyric_item_index"],
                "reviewed": True,
            }
            for row in source["segments"]  # type: ignore[index]
        ],
    }


def test_audio_runner_uses_user_lyrics_over_asr_and_pitch_evidence(tmp_path: Path) -> None:
    asr = {
        "schema_version": "asr_transcript_v1",
        "provenance": {"truth_layer": "asr_evidence_not_phone_truth"},
        "model_output": True,
        "text": "你好",
        "segments": [],
    }
    asr_path = tmp_path / "asr.json"
    asr_path.write_text(json.dumps(asr), encoding="utf-8")

    rc, summary = run_audio_draft_vsqx(
        audio=_audio(tmp_path / "audio.wav"),
        template=_template(tmp_path / "template.vsqx"),
        output_root=tmp_path / "run",
        lyrics="喜歡你",
        asr_transcript=asr_path,
        pitch_analysis=_pitch(tmp_path / "pitch.json"),
    )

    assert rc == 0
    assert summary["ok"] is True
    assert summary["lyric_source"] == "user_lyrics"
    assert summary["lyrics"] == "喜歡你"
    assert summary["asr_evidence"]["provided"] is True
    assert summary["asr_evidence"]["truth_layer"] == "asr_evidence_not_phone_truth"
    assert summary["pitch_evidence"]["note_candidate_count"] == 3
    assert summary["model_output"] is False
    assert summary["training_data"] is False
    assert summary["user_jyutping_required"] is False

    draft = json.loads(Path(summary["artifacts"]["draft"]).read_text(encoding="utf-8"))
    notes = draft["tracks"][0]["notes"]
    assert [note["lyric"] for note in notes] == ["喜", "歡", "你"]
    assert [note["pitch"] for note in notes] == [64, 67, 69]
    assert Path(summary["artifacts"]["vsqx"]).exists()


def test_template_clone_preserves_part_origin_and_strict_note_order(tmp_path: Path) -> None:
    rc, summary = run_audio_draft_vsqx(
        audio=_audio(tmp_path / "audio.wav"),
        template=_strict_template(tmp_path / "template.vsqx"),
        output_root=tmp_path / "run",
        lyrics="喜歡你",
        pitch_analysis=_pitch(tmp_path / "pitch.json"),
    )

    assert rc == 0
    root = ET.parse(summary["artifacts"]["vsqx"]).getroot()
    part = root.find(".//v:vsPart", NS)
    assert part is not None
    assert part.findtext("v:t", namespaces=NS) == "3840"
    child_names = [child.tag.rsplit("}", 1)[-1] for child in part]
    note_indexes = [index for index, name in enumerate(child_names) if name == "note"]
    assert note_indexes
    assert max(note_indexes) < child_names.index("plane")
    for note in part.findall("v:note", NS):
        assert [child.tag.rsplit("}", 1)[-1] for child in note][:7] == [
            "t",
            "dur",
            "n",
            "v",
            "y",
            "p",
            "nStyle",
        ]
        assert note.find("v:p", NS).get("lock") == "1"
    draft = json.loads(Path(summary["artifacts"]["draft"]).read_text(encoding="utf-8"))
    assert root.findtext("v:masterTrack/v:tempo/v:v", namespaces=NS) == str(
        round(float(draft["tempo"]) * 100.0)
    )


def test_audio_runner_can_route_pitch_extraction_to_existing_vocal_stem(tmp_path: Path) -> None:
    full_mix = _audio(tmp_path / "full_mix.wav")
    stem = _audio(tmp_path / "vocal_stem.wav")

    rc, summary = run_audio_draft_vsqx(
        audio=full_mix,
        vocal_stem=stem,
        template=_template(tmp_path / "template.vsqx"),
        output_root=tmp_path / "run",
        lyrics="喜歡你",
        pitch_analysis=_pitch(tmp_path / "pitch.json"),
    )

    assert rc == 0
    assert summary["pitch_evidence"]["pitch_audio"] == str(stem)
    assert summary["pitch_evidence"]["pitch_audio_role"] == "vocal_stem_or_pitch_dedicated_audio"
    assert summary["acceptance_boundary"]["prefer_vocal_stem_for_pitch"] is True


def test_audio_runner_explicit_auto_pitch_and_phone_evidence_builds_product_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pitch_payload = json.loads(_pitch(tmp_path / "seed-pitch.json").read_text(encoding="utf-8"))
    phones = {
        "segments": [
            {"start_sec": 0.0, "end_sec": 0.45, "confidence": 0.9, "candidate_phones": [{"symbol": "e", "confidence": 0.9}]},
            {"start_sec": 0.5, "end_sec": 0.95, "confidence": 0.9, "candidate_phones": [{"symbol": "u", "confidence": 0.9}]},
            {"start_sec": 1.0, "end_sec": 1.45, "confidence": 0.9, "candidate_phones": [{"symbol": "i", "confidence": 0.9}]},
        ]
    }
    phone_path = tmp_path / "phones.json"
    phone_path.write_text(json.dumps(phones), encoding="utf-8")
    monkeypatch.setattr(audio_runner, "analyze_audio_pitch_notes", lambda path: pitch_payload)

    rc, summary = run_audio_draft_vsqx(
        audio=_audio(tmp_path / "audio.wav"),
        template=_template(tmp_path / "template.vsqx"),
        output_root=tmp_path / "run",
        lyrics="喜歡你",
        auto_pitch=True,
        phone_segments=phone_path,
    )

    assert rc == 0
    assert summary["ok"] is True
    assert summary["draft_vsqx"]["evidence_fusion"]["direct_overlap_ratio"] == 1.0
    assert summary["phone_evidence"]["segment_count"] == 3
    assert Path(summary["artifacts"]["pitch_analysis"]).exists()
    provenance = json.loads(Path(summary["artifacts"]["provenance"]).read_text(encoding="utf-8"))
    assert provenance["manual_acceptance_status"] == "pending_user_validation"
    draft = json.loads(Path(summary["artifacts"]["draft"]).read_text(encoding="utf-8"))
    assert [note["lyric"] for note in draft["tracks"][0]["notes"]] == ["喜", "歡", "你"]
    assert all(note["lyric"] != "la" for note in draft["tracks"][0]["notes"])


def test_audio_runner_product_path_requires_pitch_and_phone_evidence(tmp_path: Path) -> None:
    rc, summary = run_audio_draft_vsqx(
        audio=_audio(tmp_path / "audio.wav"),
        template=_template(tmp_path / "template.vsqx"),
        output_root=tmp_path / "run",
        lyrics="喜歡你",
        phone_segments=tmp_path / "missing-phones.json",
    )

    assert rc == 2
    assert summary["error_kind"] == "invalid_phone_evidence"


def test_audio_runner_can_use_fake_asr_text_when_lyrics_missing(tmp_path: Path) -> None:
    rc, summary = run_audio_draft_vsqx(
        audio=_audio(tmp_path / "audio.wav"),
        template=_template(tmp_path / "template.vsqx"),
        output_root=tmp_path / "run",
        fake_asr=True,
        context="喜歡你",
        timestamps=True,
        pitches=[64, 67, 69],
    )

    assert rc == 0
    assert summary["lyric_source"] == "qwen3_asr_text_evidence"
    assert summary["lyrics"] == "喜歡你"
    assert summary["asr_evidence"]["provided"] is True
    assert summary["asr_evidence"]["model_output"] is False
    assert Path(summary["artifacts"]["asr_transcript"]).exists()
    assert Path(summary["artifacts"]["vsqx"]).exists()


def test_audio_runner_keeps_user_lyrics_when_fake_asr_is_auxiliary(tmp_path: Path) -> None:
    rc, summary = run_audio_draft_vsqx(
        audio=_audio(tmp_path / "audio.wav"),
        template=_template(tmp_path / "template.vsqx"),
        output_root=tmp_path / "run",
        lyrics="\u559c\u6b61\u4f60",
        fake_asr=True,
        context="\u4f60\u597d",
        pitches=[64, 67, 69],
    )

    assert rc == 0
    assert summary["lyric_source"] == "user_lyrics"
    assert summary["lyrics"] == "\u559c\u6b61\u4f60"
    assert summary["asr_evidence"]["provided"] is True


def test_audio_runner_requires_lyrics_or_asr_evidence(tmp_path: Path) -> None:
    rc, summary = run_audio_draft_vsqx(
        audio=_audio(tmp_path / "audio.wav"),
        template=_template(tmp_path / "template.vsqx"),
        output_root=tmp_path / "run",
    )

    assert rc == 2
    assert summary["ok"] is False
    assert summary["error_kind"] == "missing_lyrics_evidence"
    assert any("no lyrics or ASR evidence" in error for error in summary["errors"])
    assert not Path(summary["artifacts"]["vsqx"]).exists()


def test_audio_runner_passes_explicit_qwen_model_and_timestamps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[object] = []

    class FakeAdapter:
        def __init__(self, *, model_name_or_path: str | None = None) -> None:
            calls.append(("load", model_name_or_path))

        def transcribe(self, request: object) -> dict[str, object]:
            calls.append(request)
            return {
                "schema_version": "asr_transcript_v1",
                "text": "喜歡你",
                "model": {"name_or_path": "model-dir"},
                "segments": [],
                "provenance": {
                    "truth_layer": "asr_evidence_not_phone_truth",
                    "timestamps_requested": True,
                },
            }

    monkeypatch.setattr(audio_runner, "Qwen3ASRAdapter", FakeAdapter)
    rc, summary = audio_runner.run_audio_draft_vsqx(
        audio=_audio(tmp_path / "audio.wav"),
        template=_template(tmp_path / "template.vsqx"),
        output_root=tmp_path / "run",
        asr_model="model-dir",
        language="Cantonese",
        context="singing",
        timestamps=True,
        pitches=[64, 67, 69],
    )

    assert rc == 0
    assert calls[0] == ("load", "model-dir")
    request = calls[1]
    assert request.model == "model-dir"
    assert request.language == "Cantonese"
    assert request.context == "singing"
    assert request.timestamps is True
    assert summary["lyric_source"] == "qwen3_asr_text_evidence"
    assert summary["asr_evidence"]["model_output"] is True
    assert summary["asr_evidence"]["timestamps_requested"] is True


def test_audio_runner_reports_qwen_model_failure(tmp_path: Path, monkeypatch) -> None:
    class FailingAdapter:
        def __init__(self, *, model_name_or_path: str | None = None) -> None:
            del model_name_or_path

        def transcribe(self, request: object) -> dict[str, object]:
            del request
            raise audio_runner.Qwen3ASRLoadError("runtime unavailable")

    monkeypatch.setattr(audio_runner, "Qwen3ASRAdapter", FailingAdapter)
    rc, summary = audio_runner.run_audio_draft_vsqx(
        audio=_audio(tmp_path / "audio.wav"),
        template=_template(tmp_path / "template.vsqx"),
        output_root=tmp_path / "run",
        asr_model="model-dir",
    )

    assert rc == 2
    assert summary["ok"] is False
    assert summary["error_kind"] == "asr_model_unavailable"
    assert "runtime unavailable" in summary["errors"][0]


def test_audio_runner_rejects_empty_fake_asr_evidence(tmp_path: Path) -> None:
    rc, summary = run_audio_draft_vsqx(
        audio=_audio(tmp_path / "audio.wav"),
        template=_template(tmp_path / "template.vsqx"),
        output_root=tmp_path / "run",
        fake_asr=True,
        context="  ",
    )

    assert rc == 2
    assert summary["error_kind"] == "asr_transcript_error"


def test_audio_runner_cli_smoke(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [
            sys.executable,
            "tools/build_miku_v4c_audio_draft_vsqx.py",
            "--audio",
            str(_audio(tmp_path / "audio.wav")),
            "--lyrics",
            "喜歡你",
            "--template",
            str(_template(tmp_path / "template.vsqx")),
            "--output-root",
            str(tmp_path / "cli-run"),
            "--pitches",
            "64,67,69",
            "--pretty",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["schema_version"] == "miku_v4c_audio_draft_vsqx_run_v1"
    assert summary["ok"] is True
    assert summary["lyric_source"] == "user_lyrics"
    assert Path(summary["artifacts"]["audio_product_summary"]).exists()
    assert Path(summary["artifacts"]["vsqx"]).exists()


def test_strict_forced_alignment_product_closes_evidence_loop(tmp_path: Path) -> None:
    audio = _real_wav(tmp_path / "audio.wav")
    stem = _real_wav(tmp_path / "stem.wav")
    timeline = _strict_timeline(tmp_path / "timeline.json", audio)
    alignment = _forced_alignment(tmp_path / "forced.json", stem, timeline, audio)
    acceptance = _qualified_model_acceptance(
        tmp_path / "acceptance.json",
        alignment=alignment,
        stem=stem,
        timeline_path=timeline,
        audio=audio,
    )

    rc, summary = run_audio_draft_vsqx(
        audio=audio,
        vocal_stem=stem,
        template=_template(tmp_path / "template.vsqx"),
        output_root=tmp_path / "run",
        lyrics_timeline=timeline,
        forced_phone_alignment=alignment,
        forced_phone_acceptance=acceptance,
        pitch_analysis=_pitch(tmp_path / "pitch.json"),
        strict_product=True,
    )

    assert rc == 0
    assert summary["ok"] is True
    assert summary["product_mode"] == "forced_alignment_product"
    assert summary["lyrics"] == "喜歡你"
    assert summary["draft_vsqx"]["evidence_fusion"]["direct_overlap_ratio"] == 1.0
    assert summary["forced_alignment_evidence"]["provided"] is True
    provenance = json.loads(Path(summary["artifacts"]["provenance"]).read_text(encoding="utf-8"))
    assert provenance["vocal_stem"]["sha256"] == sha256_file(stem)
    assert provenance["manual_acceptance_status"] == "pending_user_validation"
    phone_evidence = provenance["phone_evidence"]
    assert phone_evidence["artifact_sha256"] == sha256_file(Path(phone_evidence["artifact_path"]))
    forced_evidence = provenance["forced_alignment_evidence"]
    assert forced_evidence["artifact_sha256"] == sha256_file(Path(forced_evidence["artifact_path"]))
    assert forced_evidence["acceptance"]["route"] == QUALIFIED_MODEL_ROUTE
    verification = verify_product_provenance(Path(summary["artifacts"]["provenance"]))
    assert verification is not None
    assert verification["ok"] is True, verification["errors"]


def test_strict_product_accepts_review_tool_short_then_full_human_route(
    tmp_path: Path,
) -> None:
    audio = _real_wav(tmp_path / "audio.wav", seconds=12.0)
    stem = _real_wav(tmp_path / "stem.wav", seconds=12.0)
    timeline = _strict_timeline(tmp_path / "timeline.json", audio)
    machine_alignment = _forced_alignment(
        tmp_path / "machine-forced.json", stem, timeline, audio
    )

    short_source = load_review_inputs(
        stem,
        machine_alignment,
        lyrics_timeline_path=timeline,
    )
    short = save_review(
        tmp_path / "short-review",
        short_source,
        _human_review_request(short_source),
    )
    full_source = load_review_inputs(
        stem,
        machine_alignment,
        lyrics_timeline_path=timeline,
        review_scope=FULL_SONG_SCOPE,
        short_clip_acceptance_path=Path(str(short["acceptance_path"])),
    )
    full = save_review(
        tmp_path / "full-review",
        full_source,
        _human_review_request(full_source),
    )

    rc, summary = run_audio_draft_vsqx(
        audio=audio,
        vocal_stem=stem,
        template=_template(tmp_path / "template.vsqx"),
        output_root=tmp_path / "run",
        lyrics_timeline=timeline,
        forced_phone_alignment=Path(str(full["reviewed_alignment_path"])),
        forced_phone_acceptance=Path(str(full["acceptance_path"])),
        pitch_analysis=_pitch(tmp_path / "pitch.json"),
        strict_product=True,
    )

    assert rc == 0, summary
    assert summary["ok"] is True
    acceptance = summary["forced_alignment_evidence"]["acceptance"]
    assert acceptance["route"] == "human_reviewed_phone_alignment"
    assert acceptance["review"]["scope"] == FULL_SONG_SCOPE
    verification = verify_product_provenance(Path(summary["artifacts"]["provenance"]))
    assert verification is not None
    assert verification["ok"] is True, verification["errors"]


def test_strict_product_rejects_self_declared_model_output_without_acceptance(tmp_path: Path) -> None:
    audio = _real_wav(tmp_path / "audio.wav")
    stem = _real_wav(tmp_path / "stem.wav")
    timeline = _strict_timeline(tmp_path / "timeline.json", audio)
    alignment = _forced_alignment(tmp_path / "forced.json", stem, timeline, audio)
    output_root = tmp_path / "run"
    output_root.mkdir()
    stale_vsqx = output_root / "candidate.vsqx"
    stale_draft = output_root / "candidate.json"
    stale_provenance = output_root / "candidate_provenance.json"
    stale_vsqx.write_text("stale", encoding="utf-8")
    stale_draft.write_text("stale", encoding="utf-8")
    stale_provenance.write_text('{"ok": true}', encoding="utf-8")

    rc, summary = run_audio_draft_vsqx(
        audio=audio,
        vocal_stem=stem,
        template=_template(tmp_path / "template.vsqx"),
        output_root=output_root,
        lyrics_timeline=timeline,
        forced_phone_alignment=alignment,
        pitch_analysis=_pitch(tmp_path / "pitch.json"),
        strict_product=True,
        prefix="candidate",
    )

    assert rc == 2
    assert summary["error_kind"] == "unqualified_forced_phone_evidence"
    assert summary["details"] == {
        "qualified_model_route": False,
        "human_review_route": False,
    }
    assert not stale_vsqx.exists()
    assert not stale_draft.exists()
    assert not stale_provenance.exists()


def test_rejected_run_invalidates_all_sidecars_from_prior_success(tmp_path: Path) -> None:
    audio = _real_wav(tmp_path / "audio.wav")
    stem = _real_wav(tmp_path / "stem.wav")
    timeline = _strict_timeline(tmp_path / "timeline.json", audio)
    alignment = _forced_alignment(tmp_path / "forced.json", stem, timeline, audio)
    acceptance = _qualified_model_acceptance(
        tmp_path / "acceptance.json",
        alignment=alignment,
        stem=stem,
        timeline_path=timeline,
        audio=audio,
    )
    output_root = tmp_path / "run"
    first_rc, first = run_audio_draft_vsqx(
        audio=audio,
        vocal_stem=stem,
        template=_template(tmp_path / "template.vsqx"),
        output_root=output_root,
        lyrics_timeline=timeline,
        forced_phone_alignment=alignment,
        forced_phone_acceptance=acceptance,
        pitch_analysis=_pitch(tmp_path / "pitch.json"),
        strict_product=True,
        prefix="candidate",
    )
    assert first_rc == 0, first
    generated_paths = {
        key: Path(value)
        for key, value in first["artifacts"].items()
        if key in {
            "audio_product_summary",
            "draft",
            "vsqx",
            "draft_summary",
            "lyrics_timeline",
            "g2p",
            "forced_phone_alignment",
            "phone_segments",
            "provenance",
        }
    }
    assert all(path.exists() for path in generated_paths.values())

    rejected_rc, rejected = run_audio_draft_vsqx(
        audio=audio,
        vocal_stem=stem,
        template=_template(tmp_path / "template-rejected.vsqx"),
        output_root=output_root,
        lyrics_timeline=timeline,
        forced_phone_alignment=alignment,
        pitch_analysis=_pitch(tmp_path / "pitch-rejected.json"),
        strict_product=True,
        prefix="candidate",
    )

    assert rejected_rc == 2
    assert rejected["error_kind"] == "unqualified_forced_phone_evidence"
    summary_path = Path(rejected["artifacts"]["audio_product_summary"])
    assert summary_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["ok"] is False
    for key, path in generated_paths.items():
        if key != "audio_product_summary":
            assert not path.exists(), key


def test_failed_export_does_not_recreate_provenance_sidecar(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio = _real_wav(tmp_path / "audio.wav")
    stem = _real_wav(tmp_path / "stem.wav")
    timeline = _strict_timeline(tmp_path / "timeline.json", audio)
    alignment = _forced_alignment(tmp_path / "forced.json", stem, timeline, audio)
    acceptance = _qualified_model_acceptance(
        tmp_path / "acceptance.json",
        alignment=alignment,
        stem=stem,
        timeline_path=timeline,
        audio=audio,
    )

    def fail_export(**kwargs: object) -> tuple[int, dict[str, object]]:
        del kwargs
        return (
            1,
            {
                "schema_version": "miku_v4c_draft_vsqx_run_v1",
                "ok": False,
                "error_kind": "vsqx_export_failure",
                "errors": ["simulated exporter rejection"],
                "artifacts": {},
            },
        )

    monkeypatch.setattr(audio_runner, "run_draft_vsqx", fail_export)
    output_root = tmp_path / "run"
    stale_provenance = output_root / "candidate_provenance.json"
    output_root.mkdir()
    stale_provenance.write_text('{"ok": true}', encoding="utf-8")

    rc, summary = run_audio_draft_vsqx(
        audio=audio,
        vocal_stem=stem,
        template=_template(tmp_path / "template.vsqx"),
        output_root=output_root,
        lyrics_timeline=timeline,
        forced_phone_alignment=alignment,
        forced_phone_acceptance=acceptance,
        pitch_analysis=_pitch(tmp_path / "pitch.json"),
        strict_product=True,
        prefix="candidate",
    )

    assert rc == 1
    assert summary["ok"] is False
    assert summary["error_kind"] == "vsqx_export_failure"
    assert not stale_provenance.exists()
    assert not Path(summary["artifacts"]["provenance"]).exists()
    failure_summary = Path(summary["artifacts"]["audio_product_summary"])
    assert failure_summary.exists()
    assert json.loads(failure_summary.read_text(encoding="utf-8"))["ok"] is False


def test_mfa_manifest_cannot_qualify_model_route_without_human_review(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio = _real_wav(tmp_path / "audio.wav")
    stem = _real_wav(tmp_path / "stem.wav")
    timeline = _strict_timeline(tmp_path / "timeline.json", audio)
    fixture_alignment = _forced_alignment(tmp_path / "fixture-forced.json", stem, timeline, audio)
    mfa_alignment = tmp_path / "mfa.json"
    mfa_manifest = tmp_path / "mfa-manifest.json"
    mfa_dictionary = tmp_path / "mfa.dict"
    acceptance = tmp_path / "acceptance.json"
    for path in (mfa_alignment, mfa_manifest, mfa_dictionary, acceptance):
        path.write_text("{}", encoding="utf-8")
    timeline_payload = load_lyrics_timeline(timeline, audio_path=audio)
    resolved_items, _ = resolve_lyrics_timeline(timeline_payload)
    fixture_result = FixtureForcedPhoneAligner(fixture_alignment).align(
        stem,
        timeline_payload,
        resolved_items,
    )

    class FakeMfaAligner:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def align(self, *args: object, **kwargs: object) -> ForcedAlignmentResult:
            del args, kwargs
            return fixture_result

    monkeypatch.setattr(audio_runner, "MfaForcedPhoneAligner", FakeMfaAligner)
    monkeypatch.setattr(
        audio_runner,
        "load_forced_phone_acceptance",
        lambda *args, **kwargs: {"route": QUALIFIED_MODEL_ROUTE},
    )

    rc, summary = run_audio_draft_vsqx(
        audio=audio,
        vocal_stem=stem,
        template=_template(tmp_path / "template.vsqx"),
        output_root=tmp_path / "run",
        lyrics_timeline=timeline,
        mfa_alignment=mfa_alignment,
        mfa_alignment_manifest=mfa_manifest,
        mfa_dictionary=mfa_dictionary,
        forced_model_hash="1" * 64,
        forced_model_revision="fixture-revision",
        forced_phone_acceptance=acceptance,
        pitch_analysis=_pitch(tmp_path / "pitch.json"),
        strict_product=True,
    )

    assert rc == 2
    assert summary["error_kind"] == "unqualified_forced_phone_evidence"
    assert "automatic MFA alignment" in summary["errors"][0]
    assert summary["details"] == {
        "qualified_model_route": False,
        "human_review_route": False,
    }
    assert not Path(summary["artifacts"]["vsqx"]).exists()


def test_strict_product_requires_vocal_stem_and_timing(tmp_path: Path) -> None:
    audio = _real_wav(tmp_path / "audio.wav")
    rc, missing_stem = run_audio_draft_vsqx(
        audio=audio,
        template=_template(tmp_path / "template.vsqx"),
        output_root=tmp_path / "missing-stem",
        lyrics="喜歡你",
        strict_product=True,
    )
    assert rc == 2
    assert missing_stem["error_kind"] == "missing_vocal_stem_evidence"

    stem = _real_wav(tmp_path / "stem.wav")
    rc, missing_timing = run_audio_draft_vsqx(
        audio=audio,
        vocal_stem=stem,
        template=_template(tmp_path / "template-2.vsqx"),
        output_root=tmp_path / "missing-timing",
        lyrics="喜歡你",
        strict_product=True,
    )
    assert rc == 2
    assert missing_timing["error_kind"] == "missing_lyrics_timing_evidence"


def test_mfa_product_requires_dictionary_path(tmp_path: Path) -> None:
    audio = _real_wav(tmp_path / "audio.wav")
    stem = _real_wav(tmp_path / "stem.wav")
    timeline = _strict_timeline(tmp_path / "timeline.json", audio)
    mfa = tmp_path / "mfa.json"
    manifest = tmp_path / "mfa-manifest.json"
    mfa.write_text("{}", encoding="utf-8")
    manifest.write_text("{}", encoding="utf-8")

    rc, summary = run_audio_draft_vsqx(
        audio=audio,
        vocal_stem=stem,
        template=_template(tmp_path / "template.vsqx"),
        output_root=tmp_path / "run",
        lyrics_timeline=timeline,
        mfa_alignment=mfa,
        mfa_alignment_manifest=manifest,
        forced_model_hash="1" * 64,
        forced_model_revision="fixture",
    )

    assert rc == 2
    assert summary["error_kind"] == "invalid_forced_phone_evidence"
    assert "mfa-dictionary" in summary["errors"][0]


def test_mfa_product_binds_dictionary_file_to_manifest_hash(tmp_path: Path) -> None:
    audio, source_alignment, model, dictionary = _mfa_fixture_paths()
    alignment = _mfa_alignment_for_cli(tmp_path / "mfa.json", source_alignment)
    timeline = REPO_ROOT / ".state/runs/forced-aligner-smoke/mfa_clip_timeline.json"
    g2p_lexicon = _timeline_g2p_lexicon(tmp_path / "g2p.json", timeline)
    mismatched_dictionary = tmp_path / "mismatched.dict"
    mismatched_dictionary.write_text(dictionary.read_text(encoding="utf-8") + "\nchanged", encoding="utf-8")
    manifest = _mfa_manifest(
        tmp_path / "mfa-manifest.json",
        alignment=alignment,
        audio=audio,
        timeline=timeline,
        model=model,
        dictionary=dictionary,
    )

    rc, summary = run_audio_draft_vsqx(
        audio=audio,
        vocal_stem=audio,
        template=_template(tmp_path / "template.vsqx"),
        output_root=tmp_path / "run",
        lyrics_timeline=timeline,
        mfa_alignment=alignment,
        mfa_alignment_manifest=manifest,
        mfa_dictionary=mismatched_dictionary,
        g2p_lexicon=g2p_lexicon,
        forced_model_hash=sha256_file(model),
        forced_model_revision="HKCantonese_models@fixture/cv19_v3",
    )

    assert rc == 2
    assert summary["error_kind"] == "invalid_forced_phone_evidence"
    assert "dictionary_sha256" in summary["errors"][0]
    assert not Path(summary["artifacts"]["vsqx"]).exists()


def test_mfa_alignment_is_product_only_and_requires_manifest(tmp_path: Path) -> None:
    audio = _audio(tmp_path / "audio.wav")
    mfa = tmp_path / "mfa.json"
    mfa.write_text("{}", encoding="utf-8")

    rc, summary = run_audio_draft_vsqx(
        audio=audio,
        template=_template(tmp_path / "template.vsqx"),
        output_root=tmp_path / "run",
        lyrics="喜歡你",
        mfa_alignment=mfa,
    )

    assert rc == 2
    assert summary["error_kind"] == "invalid_forced_phone_evidence"
    assert "manifest" in summary["errors"][0]
    assert not Path(summary["artifacts"]["vsqx"]).exists()


def test_mfa_product_gate_requires_stem_and_timeline_before_alignment(tmp_path: Path) -> None:
    audio = _audio(tmp_path / "audio.wav")
    mfa = tmp_path / "mfa.json"
    manifest = tmp_path / "mfa-manifest.json"
    mfa.write_text("{}", encoding="utf-8")
    manifest.write_text("{}", encoding="utf-8")

    rc, missing_stem = run_audio_draft_vsqx(
        audio=audio,
        template=_template(tmp_path / "template-stem.vsqx"),
        output_root=tmp_path / "run-stem",
        lyrics="喜歡你",
        mfa_alignment=mfa,
        mfa_alignment_manifest=manifest,
    )
    assert rc == 2
    assert missing_stem["error_kind"] == "missing_vocal_stem_evidence"
    assert not Path(missing_stem["artifacts"]["vsqx"]).exists()

    stem = _real_wav(tmp_path / "stem.wav")
    rc, missing_timing = run_audio_draft_vsqx(
        audio=audio,
        vocal_stem=stem,
        template=_template(tmp_path / "template-timing.vsqx"),
        output_root=tmp_path / "run-timing",
        lyrics="喜歡你",
        mfa_alignment=mfa,
        mfa_alignment_manifest=manifest,
    )
    assert rc == 2
    assert missing_timing["error_kind"] == "missing_lyrics_timing_evidence"
    assert not Path(missing_timing["artifacts"]["vsqx"]).exists()


def test_audio_runner_rejects_path_traversal_prefix(tmp_path: Path) -> None:
    rc, summary = run_audio_draft_vsqx(
        audio=_audio(tmp_path / "audio.wav"),
        template=_template(tmp_path / "template.vsqx"),
        output_root=tmp_path / "run",
        lyrics="喜歡你",
        pitches=[64, 67, 69],
        prefix="../outside",
    )
    assert rc == 2
    assert summary["error_kind"] == "invalid_arguments"
    assert not (tmp_path / "outside_audio_product_summary.json").exists()
