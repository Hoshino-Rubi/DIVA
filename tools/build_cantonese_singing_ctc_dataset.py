from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import wave


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from any2xsampa.data.label_provenance import (  # noqa: E402
    human_phone_truth,
    optional_sha256,
    stable_identities,
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT.parent / "state" / "data" / "cantonese_singing_ctc"
SCHEMA_VERSION = "cantonese_singing_ctc_dataset_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a leakage-safe note-level CTC dataset index from training-ready human VSQX/WAV rows."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--padding-sec", type=float, default=0.08)
    parser.add_argument("--pretty", action="store_true")
    return parser


def build_dataset(
    *,
    manifest_path: Path,
    output_root: Path,
    padding_sec: float = 0.08,
    pretty: bool = True,
) -> dict[str, Any]:
    if padding_sec < 0.0:
        raise ValueError("padding_sec must be >= 0")
    manifest = _load_json(manifest_path)
    manifest_sha256 = _sha256(manifest_path)
    rows = [row for row in manifest.get("rows", []) if isinstance(row, Mapping)]
    ready_rows = [row for row in rows if row.get("training_ready") is True]
    if not ready_rows:
        raise ValueError("manifest has no training_ready rows")

    audio_probes: dict[str, dict[str, Any]] = {}
    examples: list[dict[str, Any]] = []
    group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    vocab = sorted({str(token) for row in ready_rows for token in row.get("target_phone_tokens", [])})

    for row in ready_rows:
        truth = human_phone_truth(row)
        identities = stable_identities(row)
        segment_id = str(row.get("segment_id") or "")
        group_id = str(row.get("variant_group") or segment_id)
        target = truth["target_phone_tokens"]
        if not target:
            raise ValueError(f"training_ready row has no target phones: {segment_id}#{row.get('note_index')}")
        for audio_index, audio_value in enumerate(row.get("audio_paths", []), start=1):
            audio_path = Path(str(audio_value))
            audio_key = str(audio_path)
            probe = audio_probes.get(audio_key)
            if probe is None:
                probe = _probe_wav(audio_path)
                audio_probes[audio_key] = probe
            start_sec = float(row.get("start_sec") or 0.0)
            end_sec = float(row.get("end_sec") or 0.0)
            crop_start = max(0.0, start_sec - padding_sec)
            crop_end = min(float(probe["duration_sec"]), end_sec + padding_sec)
            if crop_end <= crop_start:
                raise ValueError(f"invalid crop for {segment_id}#{row.get('note_index')}: {crop_start}-{crop_end}")
            approximate_ctc_frames = int((crop_end - crop_start) * 50.0)
            if approximate_ctc_frames < len(target):
                raise ValueError(
                    f"crop too short for CTC target: {segment_id}#{row.get('note_index')} "
                    f"frames={approximate_ctc_frames} target={len(target)}"
                )
            example = {
                "schema_version": "cantonese_singing_ctc_example_v1",
                "example_id": f"{segment_id}-n{int(row.get('note_index') or 0):03d}-a{audio_index:02d}",
                "group_id": group_id,
                "domain": _infer_domain(segment_id),
                "source_manifest": str(manifest_path.resolve(strict=False)),
                "source_manifest_sha256": manifest_sha256,
                "source_corpus": row.get("source_corpus"),
                "source_corpus_sha256": optional_sha256(row, "source_corpus_sha256"),
                "source_vsqx": row.get("source_vsqx"),
                "source_vsqx_sha256": optional_sha256(row, "source_vsqx_sha256"),
                **identities,
                "segment_id": segment_id,
                "variant_group": row.get("variant_group"),
                "note_index": row.get("note_index"),
                "character_index": row.get("character_index"),
                "lyric_character": row.get("lyric_character"),
                "audio_path": str(audio_path),
                "audio_sha256": probe["sha256"],
                "audio_sample_rate": probe["sample_rate"],
                "audio_channels": probe["channels"],
                "note_start_sec": start_sec,
                "note_end_sec": end_sec,
                "crop_start_sec": round(crop_start, 6),
                "crop_end_sec": round(crop_end, 6),
                "crop_duration_sec": round(crop_end - crop_start, 6),
                "approximate_ctc_frames": approximate_ctc_frames,
                "v4c_tokens": [str(token) for token in row.get("v4c_tokens", [])],
                "target_phone_tokens": target,
                "target_phone_sequence": " ".join(target),
                "internal_ipa_tokens": truth["internal_ipa_tokens"],
                "articulatory_features": truth["articulatory_features"],
                "label_provenance": truth["label_provenance"],
                "phoneme_lock": row.get("phoneme_lock") is True,
                "human_authored_annotation_evidence": True,
                "manual_vsqx_rejected": False,
                "training_data": False,
                "training_data_candidate": True,
            }
            examples.append(example)
            group_rows[group_id].append(example)

    folds = _build_group_folds(group_rows)
    if not folds:
        raise ValueError(
            "source and identity grouping cannot produce non-empty train/validation folds"
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(manifest_path.resolve(strict=False)),
        "manifest_sha256": manifest_sha256,
        "padding_sec": padding_sec,
        "row_count": len(ready_rows),
        "example_count": len(examples),
        "audio_file_count": len(audio_probes),
        "group_count": len(group_rows),
        "vocab": vocab,
        "vocab_size": len(vocab),
        "token_counts": dict(Counter(token for example in examples for token in example["target_phone_tokens"])),
        "audio_probes": audio_probes,
        "folds": folds,
        "examples": examples,
        "leakage_policy": {
            "split_unit": "connected_component_identity_and_source_evidence",
            "same_variant_group_may_not_cross_train_validation": True,
            "same_note_wav_variants_may_not_cross_train_validation": True,
        },
        "human_authored_policy": {
            "manual_vsqx_rejected_count": 0,
            "override_similarity_threshold": 0.8,
            "unresolved_rows_preserved_outside_training_ready_subset": len(rows) - len(ready_rows),
        },
    }

    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "ctc_dataset.json"
    jsonl_path = output_root / "ctc_examples.jsonl"
    _write_json(json_path, payload, pretty=pretty)
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {key: value for key, value in payload.items() if key != "examples"}
    summary["artifacts"] = {"dataset": str(json_path), "examples_jsonl": str(jsonl_path)}
    _write_json(output_root / "ctc_dataset_summary.json", summary, pretty=pretty)
    return payload


def _build_group_folds(group_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    parents = {group: group for group in group_rows}

    def find(group: str) -> str:
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    evidence_owner: dict[str, str] = {}
    for group, rows in group_rows.items():
        for row in rows:
            evidence = {
                f"audio_sha256:{str(row.get('audio_sha256') or '').lower()}",
                "source_vsqx_sha256:" + str(row.get("source_vsqx_sha256") or "").lower(),
                "source_corpus_sha256:" + str(row.get("source_corpus_sha256") or "").lower(),
                "source_vsqx:" + str(row.get("source_vsqx") or "").replace("\\", "/").casefold(),
                "source_corpus:" + str(row.get("source_corpus") or "").replace("\\", "/").casefold(),
                "singer_id:" + str(row.get("singer_id") or "").casefold(),
                "song_id:" + str(row.get("song_id") or "").casefold(),
                "phrase_id:" + str(row.get("phrase_id") or "").casefold(),
            }
            for key in evidence:
                if key.endswith(":"):
                    continue
                owner = evidence_owner.setdefault(key, group)
                union(group, owner)

    components: dict[str, list[str]] = defaultdict(list)
    for group in sorted(group_rows):
        components[find(group)].append(group)
    folds = []
    all_groups = sorted(group_rows)
    for fold_index, validation_groups in enumerate(sorted(components.values(), key=lambda values: values[0])):
        validation_rows = [row for group in validation_groups for row in group_rows[group]]
        train_groups = [group for group in all_groups if group not in validation_groups]
        train_tokens = {
            str(token)
            for group in train_groups
            for row in group_rows[group]
            for token in row.get("target_phone_tokens", [])
        }
        validation_tokens = {
            str(token) for row in validation_rows for token in row.get("target_phone_tokens", [])
        }
        if not train_groups:
            continue
        folds.append(
            {
                "fold_index": fold_index,
                "validation_group": validation_groups[0] if len(validation_groups) == 1 else "component:" + hashlib.sha256("\n".join(validation_groups).encode("utf-8")).hexdigest(),
                "validation_groups": validation_groups,
                "validation_domain": validation_rows[0].get("domain") if validation_rows else None,
                "train_groups": train_groups,
                "train_example_count": sum(len(group_rows[group]) for group in train_groups),
                "validation_example_count": len(validation_rows),
                "validation_tokens_missing_from_train": sorted(validation_tokens - train_tokens),
                "leakage_free": True,
                "source_evidence_connected_component": True,
            }
        )
    return folds


def _infer_domain(segment_id: str) -> str:
    if segment_id.startswith("freedom-flower"):
        return "freedom_flower"
    if segment_id.startswith("qre-"):
        return "qre"
    if segment_id.startswith("hktk-"):
        return "human_made_sections"
    return "unknown"


def _probe_wav(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".wav" or not path.is_file():
        raise ValueError(f"CTC dataset accepts existing WAV files only: {path}")
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        frames = handle.getnframes()
        sample_width = handle.getsampwidth()
    if sample_rate <= 0 or frames <= 0:
        raise ValueError(f"invalid WAV metadata: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frames": frames,
        "duration_sec": round(frames / sample_rate, 6),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any], *, pretty: bool) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = build_dataset(
            manifest_path=args.manifest,
            output_root=args.output_root,
            padding_sec=args.padding_sec,
            pretty=args.pretty,
        )
    except (OSError, ValueError, json.JSONDecodeError, wave.Error) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key not in {"examples", "audio_probes"}},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
