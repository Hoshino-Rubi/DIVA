from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from any2xsampa.vocaloid.vsqx_alignment import parse_vsqx_timeline  # noqa: E402


SCHEMA_VERSION = "cantonese_singing_reference_manifest_v1"
DEFAULT_MAPPING = SRC_ROOT / "any2xsampa" / "config" / "v4c_to_recognizer_contract.json"
PHRASE_GAP_SEC = 0.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _features(phone: str, ipa: str, mapping_sha256: str) -> dict[str, str]:
    if phone in {"a", "@", "e", "i", "o", "u"}:
        phone_class = "vowel"
    elif phone in {"m", "n", "N"}:
        phone_class = "nasal"
    elif phone == "l":
        phone_class = "approximant"
    elif phone == "ts":
        phone_class = "affricate"
    elif phone == "s":
        phone_class = "fricative"
    else:
        phone_class = "stop"
    return {
        "class": phone_class,
        "phone": phone,
        "ipa": ipa,
        "source": "v4c_to_recognizer_contract_v1",
        "mapping_sha256": mapping_sha256,
    }


def _map_tokens(
    tokens: list[str],
    mapping: Mapping[str, Any],
    mapping_sha256: str,
) -> tuple[list[str], list[str], list[dict[str, str]], list[str]]:
    entries = mapping.get("tokens")
    if not isinstance(entries, Mapping):
        raise ValueError("V4C mapping requires a tokens object")
    phones: list[str] = []
    ipa: list[str] = []
    losses: list[str] = []
    for token in tokens:
        entry = entries.get(token)
        if not isinstance(entry, Mapping):
            raise ValueError(f"V4C token has no recognizer-contract mapping: {token}")
        token_phones = [str(value).strip() for value in entry.get("phones", [])]
        token_ipa = [str(value).strip() for value in entry.get("ipa", [])]
        if not token_phones or len(token_phones) != len(token_ipa):
            raise ValueError(f"V4C mapping has inconsistent phone/IPA lengths: {token}")
        phones.extend(token_phones)
        ipa.extend(token_ipa)
        if entry.get("loss"):
            losses.append(f"{token}: {entry['loss']}")
    features = [_features(phone, symbol, mapping_sha256) for phone, symbol in zip(phones, ipa)]
    return phones, ipa, features, losses


def build_manifest(
    triplet_manifest_path: Path,
    output_path: Path,
    *,
    mapping_path: Path = DEFAULT_MAPPING,
    allow_unmapped: bool = False,
) -> dict[str, Any]:
    triplets = json.loads(triplet_manifest_path.read_text(encoding="utf-8-sig"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8-sig"))
    if mapping.get("schema_version") not in {
        "v4c_to_recognizer_contract_v1",
        "v4c_to_recognizer_contract_v2",
        "v4c_to_recognizer_contract_v3",
    }:
        raise ValueError("unsupported V4C recognizer-contract mapping schema")
    mapping_sha256 = _sha256(mapping_path)
    rows: list[dict[str, Any]] = []
    songs: list[dict[str, Any]] = []
    for song_index, item in enumerate(triplets.get("items", []), start=1):
        if not isinstance(item, Mapping):
            continue
        song_name = str(item.get("song") or f"song-{song_index}")
        song_id = str(item.get("song_id") or "").strip()
        if not song_id:
            song_id = "freedom-flower" if song_index == 1 else "queen-road-east" if song_index == 2 else f"song-{song_index}"
        rendered = Path(str(item["rendered_wav"]["path"]))
        vsqx = Path(str(item["human_vsqx"]["normalized_path"]))
        expected_audio_hash = str(item["rendered_wav"]["sha256"])
        expected_vsqx_hash = str(item["human_vsqx"]["normalized_sha256"])
        original = Path(str(item["original_wav"]["path"]))
        expected_original_hash = str(item["original_wav"]["sha256"])
        if (
            _sha256(rendered) != expected_audio_hash
            or _sha256(vsqx) != expected_vsqx_hash
            or _sha256(original) != expected_original_hash
        ):
            raise ValueError(f"triplet hash mismatch: {song_name}")
        timeline = parse_vsqx_timeline(vsqx)
        phrase_index = 1
        previous_end: float | None = None
        annotation_sha = str(item["human_vsqx"]["sha256"])
        for note_index, note in enumerate(timeline["notes"], start=1):
            start = float(note["start_sec"])
            end = float(note["end_sec"])
            if previous_end is not None and start - previous_end > PHRASE_GAP_SEC:
                phrase_index += 1
            previous_end = end
            raw_tokens = [str(value) for value in note.get("phonemes", [])]
            mapping_status = "mapped"
            mapping_error: str | None = None
            unmapped_tokens: list[str] = []
            try:
                phones, ipa, features, losses = _map_tokens(raw_tokens, mapping, mapping_sha256)
            except ValueError as exc:
                if not allow_unmapped:
                    raise
                mapping_status = "unresolved"
                mapping_error = str(exc)
                entries = mapping.get("tokens") if isinstance(mapping.get("tokens"), Mapping) else {}
                unmapped_tokens = [token for token in raw_tokens if token not in entries]
                phones, ipa, features, losses = [], [], [], [mapping_error]
            phrase_id = f"{song_id}-phrase-{phrase_index:03d}"
            rows.append(
                {
                    "training_ready": mapping_status == "mapped",
                    "mapping_status": mapping_status,
                    "mapping_error": mapping_error,
                    "unmapped_v4c_tokens": unmapped_tokens,
                    "segment_id": song_id,
                    "variant_group": song_id,
                    "singer_id": "miku-v4c",
                    "song_id": song_id,
                    "phrase_id": phrase_id,
                    "note_index": note_index,
                    "audio_paths": [str(rendered)],
                    "audio_sha256": expected_audio_hash,
                    "original_audio_path": str(original),
                    "original_audio_sha256": expected_original_hash,
                    "original_audio_metadata": item["original_wav"].get("audio", {}),
                    "start_sec": start,
                    "end_sec": end,
                    "pitch": note.get("pitch"),
                    "lyric_character": note.get("lyric"),
                    "v4c_tokens": raw_tokens,
                    "target_phone_tokens": phones,
                    "internal_ipa_tokens": ipa,
                    "articulatory_features": features,
                    "mapping_losses": losses,
                    "phoneme_lock": True,
                    "source_vsqx": str(vsqx),
                    "source_vsqx_sha256": expected_vsqx_hash,
                    "label_provenance": {
                        "schema_version": "human_phone_label_v1",
                        "kind": "human",
                        "source_id": f"user-final-delivery:{song_id}",
                        "source_revision": annotation_sha,
                        "annotation_sha256": annotation_sha,
                        "policy_version": str(mapping.get("schema_version") or "v4c-to-recognizer-contract-v1"),
                        "annotator_id": "workspace-user-final-delivery",
                        "review_status": "approved",
                    },
                }
            )
        songs.append(
            {
                "song": song_name,
                "song_id": song_id,
                "note_count": timeline["note_count"],
                "phrase_count": phrase_index,
                "rendered_wav_sha256": expected_audio_hash,
                "original_wav_sha256": expected_original_hash,
                "normalized_vsqx_sha256": expected_vsqx_hash,
                "unresolved_note_count": sum(
                    1 for row in rows if row["song_id"] == song_id and row["mapping_status"] == "unresolved"
                ),
            }
        )
    if not rows:
        raise ValueError("reference triplet manifest produced no training rows")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_triplet_manifest": str(triplet_manifest_path.resolve()),
        "source_triplet_manifest_sha256": _sha256(triplet_manifest_path),
        "mapping_path": str(mapping_path.resolve()),
        "mapping_sha256": mapping_sha256,
        "mapping_schema_version": str(mapping.get("schema_version") or ""),
        "truth_policy": {
            "human_v4c_tokens_preserved": True,
            "recognizer_contract_projection_is_lossy": True,
            "lyrics_used_as_acoustic_input": False,
            "unmapped_reference_notes_are_preserved": allow_unmapped,
        },
        "songs": songs,
        "row_count": len(rows),
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build hash-bound CTC rows from human VSQX/rendered-WAV triplets.")
    parser.add_argument("triplet_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument(
        "--allow-unmapped",
        action="store_true",
        help="Preserve human notes with unknown V4C tokens as unresolved rows instead of guessing or aborting.",
    )
    args = parser.parse_args(argv)
    try:
        payload = build_manifest(
            args.triplet_manifest,
            args.output,
            mapping_path=args.mapping,
            allow_unmapped=args.allow_unmapped,
        )
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"schema_version": payload["schema_version"], "row_count": payload["row_count"], "songs": payload["songs"]}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
