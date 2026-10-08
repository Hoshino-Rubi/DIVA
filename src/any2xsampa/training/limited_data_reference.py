"""Reference restoration without promoting engine aliases to acoustic truth."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
import unicodedata
import wave

from any2xsampa.training.limited_data_artifacts import (
    finite_number,
    read_json,
    sha256_file,
    sha256_object,
    verify_file,
)
from any2xsampa.vocaloid.vsqx_alignment import parse_vsqx_timeline


SPLITS: dict[str, tuple[str, ...]] = {
    "train": ("freedom-flower", "queen-road-east", "half-moon-night-song", "promise", "like-you"),
    "dev": ("shalong",),
    "test": ("ke-yi-ke-zai", "king-of-karaoke"),
}
SONG_SPLIT = {song: split for split, songs in SPLITS.items() for song in songs}
MANIFEST_SCHEMA = "limited_data_reference_manifest_v1"
DATASET_SCHEMA = "limited_data_phrase_dataset_v1"


def clean_ipa(symbol: str) -> bool:
    return bool(symbol) and not any(
        char in "?\ufffd" or "\u3400" <= char <= "\u9fff" or char.isspace()
        for char in symbol
    )


def load_mapping(path: Path) -> dict[str, Any]:
    mapping = read_json(path)
    if mapping.get("schema_version") != "v4c_reference_ipa_v4":
        raise ValueError("limited-data reference requires the v4 rich-IPA mapping")
    if mapping.get("label_scope") != "derived_engine_reference":
        raise ValueError("restored mappings must remain derived engine references")
    if mapping.get("acoustic_realization_verified") is not False:
        raise ValueError("a restored mapping cannot assert acoustic review")
    evidence = mapping["evidence"]
    source_name = str(evidence["source"])
    if Path(source_name).name != source_name:
        raise ValueError("mapping evidence must name an adjacent source file")
    source_path = path.parent / source_name
    verify_file(source_path, evidence["sha256"])
    source = read_json(source_path)["tokens"]
    tokens = mapping["tokens"]
    if not isinstance(tokens, dict) or not tokens:
        raise ValueError("mapping requires non-empty tokens")
    for token, entry in tokens.items():
        ipa = entry.get("ipa")
        if not isinstance(ipa, list) or not ipa or not all(isinstance(s, str) and clean_ipa(s) for s in ipa):
            raise ValueError(f"damaged or empty IPA: {token}")
        if token not in source or ipa != source[token].get("ipa"):
            raise ValueError(f"IPA has no exact surviving source: {token}")
    if set(tokens) & set(mapping.get("unresolved_tokens", [])):
        raise ValueError("mapped and unresolved tokens must be disjoint")
    expected_aliases = {"ts": "t\u0361s", "ts\u02b0": "t\u0361s\u02b0"}
    if mapping.get("model_token_aliases") != expected_aliases:
        raise ValueError("only the two documented affricate tie-bar aliases are permitted")
    return mapping


def load_vocabulary(path: Path) -> dict[str, int]:
    value = read_json(path)
    ids = list(value.values())
    if not ids or any(isinstance(i, bool) or not isinstance(i, int) for i in ids):
        raise ValueError("vocabulary IDs must be integers")
    if sorted(ids) != list(range(len(ids))):
        raise ValueError("vocabulary IDs must be unique and contiguous")
    if value.get("<blank>") != 0:
        raise ValueError("expected the PhoneticXeus blank ID at zero")
    return value


def project_tokens(
    tokens: Sequence[str], mapping: Mapping[str, Any], vocabulary: Mapping[str, int]
) -> dict[str, Any]:
    entries = mapping["tokens"]
    unresolved = sorted({token for token in tokens if token not in entries})
    raw_ipa = [ipa for token in tokens if token in entries for ipa in entries[token]["ipa"]]
    aliases = mapping["model_token_aliases"]
    normalized = [unicodedata.normalize("NFC", aliases.get(ipa, ipa)) for ipa in raw_ipa]
    unsupported = sorted({ipa for ipa in normalized if ipa not in vocabulary or ipa.startswith("<")})
    valid = bool(tokens) and not unresolved and not unsupported
    return {
        "raw_v4c_tokens": list(tokens),
        "legacy_intended_ipa": raw_ipa,
        "model_target_ipa": normalized if valid else [],
        "model_target_ids": [vocabulary[ipa] for ipa in normalized] if valid else [],
        "unresolved_tokens": unresolved,
        "unsupported_model_ipa": unsupported,
        "mapping_status": "resolved_proxy" if valid else "unresolved",
        "label_scope": "derived_engine_reference",
        "acoustic_review_status": "not_reviewed",
    }


def wav_info(path: Path, expected_sha256: str) -> dict[str, Any]:
    verify_file(path, expected_sha256)
    with wave.open(str(path), "rb") as stream:
        rate, channels, count = stream.getframerate(), stream.getnchannels(), stream.getnframes()
        sample_width = stream.getsampwidth()
    if min(rate, channels, count, sample_width) <= 0:
        raise ValueError(f"empty or invalid WAV: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": expected_sha256,
        "sample_rate": rate,
        "channels": channels,
        "frames": count,
        "sample_width": sample_width,
        "duration_sec": count / rate,
    }


def verified_vsqx_source(record: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    source = Path(record["path"])
    verify_file(source, record["sha256"])
    normalized = Path(record.get("normalized_path", source))
    normalized_hash = record.get("normalized_sha256", record["sha256"])
    applied = record.get("normalization_applied", False)
    if normalized_hash != record["sha256"] and applied is not True:
        raise ValueError("different normalized VSQX bytes require explicit normalization provenance")
    verify_file(normalized, normalized_hash)
    return normalized, {
        "path": str(source.resolve()),
        "sha256": record["sha256"],
        "normalized_path": str(normalized.resolve()),
        "normalized_sha256": normalized_hash,
        "normalization_applied": applied,
        "normalization_executed_this_run": False,
    }


def original_clip_rows(song_id: str, audio: Mapping[str, Any]) -> list[dict[str, Any]]:
    duration = finite_number(audio["duration_sec"], "audio duration", minimum=18.0)
    rows: list[dict[str, Any]] = []
    for index, fraction in enumerate((0.2, 0.5, 0.8), 1):
        start = round((duration - 18.0) * fraction, 6)
        rows.append({
            "example_id": f"{song_id}-original-{index:02d}",
            "song_id": song_id,
            "split": SONG_SPLIT[song_id],
            "audio_role": "original",
            "audio_path": audio["path"],
            "audio_sha256": audio["sha256"],
            "start_sec": start,
            "end_sec": round(start + 18.0, 6),
            "selection": "fixed_original_duration_quantiles_0.2_0.5_0.8_v1",
            "selection_uses_model_outputs": False,
            "selection_uses_score_or_lyrics": False,
            "alignment_status": "unverified",
            "label_scope": "none",
            "acoustic_review_status": "not_reviewed",
            "model_target_ipa": [],
            "model_target_ids": [],
            "training_eligible": False,
            "evaluation_eligible": False,
            "test_outputs_sealed": SONG_SPLIT[song_id] == "test",
        })
    return rows


def _notes(
    song_id: str, timeline: Mapping[str, Any], mapping: Mapping[str, Any], vocabulary: Mapping[str, int]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, note in enumerate(timeline["notes"], 1):
        start = finite_number(note["start_sec"], "note start", minimum=0)
        end = finite_number(note["end_sec"], "note end", minimum=0)
        if end <= start:
            raise ValueError(f"note has non-positive duration: {song_id} {index}")
        raw = note.get("phonemes")
        if not isinstance(raw, list) or not all(isinstance(token, str) for token in raw):
            raise ValueError(f"note has invalid raw tokens: {song_id} {index}")
        result.append({
            "note_id": f"{song_id}-n{index:03d}",
            "note_index": index,
            "song_id": song_id,
            "start_sec": start,
            "end_sec": end,
            "raw_lyric": note.get("lyric"),
            **project_tokens(raw, mapping, vocabulary),
            "raw_label_provenance": "user_authored_vsqx",
            "raw_annotation_modified": False,
        })
    if any(a["start_sec"] > b["start_sec"] for a, b in zip(result, result[1:])):
        raise ValueError(f"notes must be chronological: {song_id}")
    return result


def _windows(notes: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    result: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for note in notes:
        if current:
            gap = note["start_sec"] - current[-1]["end_sec"]
            span = note["end_sec"] - current[0]["start_sec"]
            if gap > 0.5 or (span > 8.0 and gap >= -1e-6):
                result.append(current)
                current = []
        current.append(note)
    if current:
        result.append(current)
    return result


def rendered_examples(song: Mapping[str, Any]) -> list[dict[str, Any]]:
    notes = song["notes"]
    audio = song["rendered_audio"]
    windows = _windows(notes)
    by_index = {row["note_index"]: index for index, row in enumerate(notes)}
    result: list[dict[str, Any]] = []
    for index, window in enumerate(windows, 1):
        first, last = window[0], window[-1]
        lo, hi = by_index[first["note_index"]], by_index[last["note_index"]]
        previous_end = notes[lo - 1]["end_sec"] if lo else 0.0
        next_start = notes[hi + 1]["start_sec"] if hi + 1 < len(notes) else audio["duration_sec"]
        start = max(0.0, first["start_sec"] - min(0.15, max(0.0, (first["start_sec"] - previous_end) / 2)))
        end = min(audio["duration_sec"], last["end_sec"] + min(0.15, max(0.0, (next_start - last["end_sec"]) / 2)))
        reasons: list[str] = []
        if any(note["mapping_status"] != "resolved_proxy" for note in window):
            reasons.append("whole_window_contains_unresolved_reference")
        if end <= start or last["end_sec"] > audio["duration_sec"] + 1e-6:
            reasons.append("reference_outside_rendered_audio")
        if end - start > 20:
            reasons.append("window_exceeds_runtime_limit")
        if first["start_sec"] < previous_end - 1e-6 or last["end_sec"] > next_start + 1e-6:
            reasons.append("window_boundary_crosses_neighbor_note")
        target_ids = [value for note in window for value in note["model_target_ids"]]
        target_ipa = [value for note in window for value in note["model_target_ipa"]]
        result.append({
            "example_id": f"{song['song_id']}-rendered-{index:03d}",
            "song_id": song["song_id"],
            "split": song["split"],
            "audio_role": "rendered",
            "audio_path": audio["path"],
            "audio_sha256": audio["sha256"],
            "start_sec": round(start, 6),
            "end_sec": round(end, 6),
            "source_vsqx_sha256": song["human_vsqx"]["sha256"],
            "note_ids": [note["note_id"] for note in window],
            "raw_v4c_tokens_by_note": [note["raw_v4c_tokens"] for note in window],
            "model_target_ipa": target_ipa if not reasons else [],
            "model_target_ids": target_ids if not reasons else [],
            "label_scope": "derived_engine_reference",
            "alignment_status": "user_declared_render_pair",
            "source_pairing_status": song.get("pairing_status"),
            "acoustic_review_status": "not_reviewed",
            "source_claims": song["source_claims"],
            "eligible": not reasons,
            "exclusion_reasons": reasons,
            "training_eligible": not reasons and song["split"] == "train",
            "test_outputs_sealed": song["split"] == "test",
        })
    return result


def build_reference_manifest(
    triplets_path: Path, mapping_path: Path, vocabulary_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    mapping = load_mapping(mapping_path)
    vocabulary = load_vocabulary(vocabulary_path)
    triplets = read_json(triplets_path)
    items = triplets.get("items", [])
    ids = [item.get("song_id") for item in items]
    if len(ids) != len(set(ids)) or set(ids) != set(SONG_SPLIT):
        raise ValueError("the locked experiment requires exactly the eight declared song groups")
    songs: list[dict[str, Any]] = []
    all_examples: list[dict[str, Any]] = []
    clips: list[dict[str, Any]] = []
    seen_hashes: dict[str, str] = {}
    for item in items:
        song_id = item["song_id"]
        original = wav_info(Path(item["original_wav"]["path"]), item["original_wav"]["sha256"])
        rendered = wav_info(Path(item["rendered_wav"]["path"]), item["rendered_wav"]["sha256"])
        vsqx = item["human_vsqx"]
        parsed_source, vsqx_binding = verified_vsqx_source(vsqx)
        for digest in (original["sha256"], rendered["sha256"], vsqx["sha256"]):
            if digest in seen_hashes and seen_hashes[digest] != song_id:
                raise ValueError(f"source hash crosses song groups: {song_id}")
            seen_hashes[digest] = song_id
        if original["sha256"] == rendered["sha256"]:
            raise ValueError(f"original and rendered source roles are ambiguous: {song_id}")
        timeline = parse_vsqx_timeline(parsed_source)
        notes = _notes(song_id, timeline, mapping, vocabulary)
        if len(notes) != vsqx["note_count"]:
            raise ValueError(f"source note count changed: {song_id}")
        song = {
            "song_id": song_id,
            "display_name": item.get("song"),
            "split": SONG_SPLIT[song_id],
            "original_audio": original,
            "rendered_audio": rendered,
            "human_vsqx": vsqx_binding,
            "source_claims": item.get("human_claims", item.get("annotation_claims", {})),
            "pairing_status": item.get("pairing_status"),
            "notes": notes,
        }
        songs.append(song)
        all_examples.extend(rendered_examples(song))
        clips.extend(original_clip_rows(song_id, original))
    bindings = {
        "triplets": {"path": str(triplets_path.resolve()), "sha256": sha256_file(triplets_path)},
        "mapping": {"path": str(mapping_path.resolve()), "sha256": sha256_file(mapping_path)},
        "vocabulary": {"path": str(vocabulary_path.resolve()), "sha256": sha256_file(vocabulary_path)},
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "bindings": bindings,
        "split": {key: list(value) for key, value in SPLITS.items()},
        "split_sha256": sha256_object(SPLITS),
        "songs": songs,
        "original_clips": clips,
        "policy": {
            "restored_ipa_is_acoustic_truth": False,
            "model_outputs_used_for_clip_selection": False,
            "external_sources_modified_or_copied": False,
            "original_test_metrics_sealed": True,
        },
    }
    manifest["manifest_hash"] = sha256_object(manifest)
    dataset = {
        "schema_version": DATASET_SCHEMA,
        "reference_manifest_hash": manifest["manifest_hash"],
        "bindings": bindings,
        "split_sha256": manifest["split_sha256"],
        "mode": "rendered_proxy_only",
        "examples": all_examples,
        "summary": summarize_reference(songs, all_examples),
        "policy": {
            "whole_unresolved_windows_excluded": True,
            "overlapping_windows": False,
            "original_accuracy_claim_allowed": False,
            "test_song_training_or_selection_allowed": False,
        },
    }
    dataset["dataset_hash"] = sha256_object(dataset)
    return manifest, dataset


def summarize_reference(songs: Sequence[Mapping[str, Any]], examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: list[dict[str, Any]] = []
    for song in songs:
        rows = [row for row in examples if row["song_id"] == song["song_id"]]
        eligible = [row for row in rows if row["eligible"]]
        notes = song["notes"]
        unresolved = Counter(token for row in notes for token in row["unresolved_tokens"])
        summary.append({
            "song_id": song["song_id"],
            "split": song["split"],
            "note_count": len(notes),
            "resolved_proxy_notes": sum(row["mapping_status"] == "resolved_proxy" for row in notes),
            "unresolved_tokens": dict(sorted(unresolved.items())),
            "windows_total": len(rows),
            "windows_eligible": len(eligible),
            "eligible_reference_phones": sum(len(row["model_target_ids"]) for row in eligible),
            "window_coverage": len(eligible) / len(rows) if rows else 0.0,
        })
    return {
        "songs": summary,
        "note_count": sum(row["note_count"] for row in summary),
        "windows_total": len(examples),
        "windows_eligible": sum(row["eligible"] for row in examples),
        "original_acoustically_reviewed_windows": 0,
    }


def validate_dataset(dataset: Mapping[str, Any]) -> None:
    if dataset.get("schema_version") != DATASET_SCHEMA:
        raise ValueError("unsupported limited-data dataset schema")
    unsigned = {key: value for key, value in dataset.items() if key != "dataset_hash"}
    if dataset.get("dataset_hash") != sha256_object(unsigned):
        raise ValueError("dataset hash mismatch")
    if dataset.get("split_sha256") != sha256_object(SPLITS):
        raise ValueError("song split differs from the frozen experiment")
    seen: set[str] = set()
    roles: dict[str, str] = {}
    for row in dataset["examples"]:
        if row["example_id"] in seen:
            raise ValueError("duplicate example ID")
        seen.add(row["example_id"])
        if SONG_SPLIT.get(row["song_id"]) != row["split"]:
            raise ValueError("song-level split leakage")
        if row["audio_role"] != "rendered" or row["label_scope"] != "derived_engine_reference":
            raise ValueError("rendered proxy dataset cannot be relabeled as original acoustic truth")
        prior = roles.setdefault(row["audio_sha256"], row["song_id"])
        if prior != row["song_id"]:
            raise ValueError("audio hash crosses song groups")
        if row["training_eligible"] and (row["split"] != "train" or not row["eligible"]):
            raise ValueError("held-out or invalid example marked for training")
        start = finite_number(row["start_sec"], "window start", minimum=0)
        end = finite_number(row["end_sec"], "window end", minimum=0)
        if row["eligible"] and (not 0 < end - start <= 20 or not row["model_target_ids"]):
            raise ValueError("invalid eligible window")


def training_gate(dataset: Mapping[str, Any]) -> dict[str, Any]:
    validate_dataset(dataset)
    valid = [row for row in dataset["examples"] if row["eligible"]]
    train = [row for row in valid if row["split"] == "train"]
    dev = [row for row in valid if row["split"] == "dev"]
    counts = {
        "train_windows": len(train),
        "train_songs": len({row["song_id"] for row in train}),
        "dev_windows": len(dev),
        "dev_reference_phones": sum(len(row["model_target_ids"]) for row in dev),
    }
    passed = counts["train_windows"] >= 20 and counts["train_songs"] >= 3 and counts["dev_windows"] >= 3 and counts["dev_reference_phones"] >= 100
    return {
        "passed": passed,
        "mode": "rendered_proxy_only",
        "counts": counts,
        "original_truth_gate_passed": False,
        "qualified_recognizer": False,
        "test_selection_allowed": False,
    }


def require_rendered_source_role(row: Mapping[str, Any], triplet: Mapping[str, Any]) -> None:
    expected = triplet["rendered_wav"]
    if row["audio_role"] != "rendered" or row["audio_sha256"] != expected["sha256"]:
        raise ValueError("declared rendered role does not match the triplet's rendered WAV")
    if row["audio_sha256"] == triplet["original_wav"]["sha256"]:
        raise ValueError("original audio cannot be relabeled as rendered proxy data")
    if Path(row["audio_path"]).resolve() != Path(expected["path"]).resolve():
        raise ValueError("rendered source path differs from the frozen triplet")
    if row["source_vsqx_sha256"] != triplet["human_vsqx"]["sha256"]:
        raise ValueError("reference VSQX differs from the frozen triplet")


def validate_dataset_source_roles(dataset: Mapping[str, Any]) -> None:
    binding = dataset["bindings"]["triplets"]
    path = Path(binding["path"])
    verify_file(path, binding["sha256"])
    manifest = read_json(path)
    items = {item["song_id"]: item for item in manifest["items"]}
    if set(items) != set(SONG_SPLIT):
        raise ValueError("source triplets differ from the frozen eight-song experiment")
    for row in dataset["examples"]:
        require_rendered_source_role(row, items[row["song_id"]])
