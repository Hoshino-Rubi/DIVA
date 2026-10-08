"""Real Cantonese singing CTC recognizer over a validated local bundle.

The bundle and checkpoint are explicit resources.  This adapter never consults
lyrics while decoding and never creates a synthetic segment when the model is
missing or emits only CTC blanks.  ``torch``/``transformers`` are imported only
when the default runtime is constructed, so fixture mode remains dependency
free and runnable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from any2xsampa.core.interfaces import UniversalPhoneRecognizer
from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange

from any2xsampa.adapters.checkpoint_loader import (
    CheckpointLoadError,
    load_trained_checkpoint,
    sha256_file,
)


BUNDLE_SCHEMA = "recognizer_bundle_v1"
CHECKPOINT_SCHEMA = "cantonese_singing_ctc_checkpoint_v1"
_FIXED_SAMPLE_RATE = 16_000
_FIXED_CHUNK_SEC = 8.0
_FIXED_OVERLAP_SEC = 1.0
_VERIFIED_TORCH_PREFIX = "2.13.0+cu126"
_VERIFIED_CUDA_VERSION = "12.6"


class CTCAcousticRuntime(Protocol):
    def infer(self, audio: Any, *, sample_rate: int) -> tuple[Any, float]:
        """Return frame logits and frame hop in seconds."""


@dataclass(frozen=True, slots=True)
class RecognizerBundle:
    root: Path
    backbone_path: Path
    checkpoint_path: Path
    vocab_path: Path
    license_manifest_path: Path
    backbone_family: str
    backbone_id: str
    backbone_revision: str
    phones: tuple[str, ...]
    blank_id: int
    sample_rate: int
    chunk_sec: float
    overlap_sec: float
    temperature: float
    manifest: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _DecodedRun:
    start_sec: float
    end_sec: float
    candidates: tuple[tuple[str, float], ...]
    chunk_index: int
    frame_start: int
    frame_end: int
    frame_hop_sec: float

    @property
    def symbol(self) -> str:
        return self.candidates[0][0]

    @property
    def confidence(self) -> float:
        return self.candidates[0][1]


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain an object: {path}")
    return payload


def _safe_bundle_path(root: Path, raw: object, *, label: str) -> Path:
    relative = str(raw or "").strip()
    if not relative:
        raise ValueError(f"Recognizer bundle {label} path is required")
    candidate = (root / relative).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Recognizer bundle {label} path must be a safe relative path") from exc
    return candidate


def _state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    try:
        for name in sorted(state):
            tensor = state[name].detach().cpu().contiguous()
            digest.update(str(name).encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
    except Exception as exc:
        raise ValueError("Recognizer checkpoint state must contain tensor values") from exc
    return digest.hexdigest()


def _manifest_entry(manifest: Mapping[str, Any], key: str, *, root: Path) -> tuple[Path, Mapping[str, Any]]:
    entry = manifest.get(key)
    if not isinstance(entry, Mapping):
        raise ValueError(f"Recognizer bundle is missing {key} metadata")
    path = _safe_bundle_path(root, entry.get("path"), label=f"{key} file")
    if not path.is_file():
        raise ValueError(f"Recognizer bundle {key} file is missing: {path}")
    expected = str(entry.get("sha256", "")).strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError(f"Recognizer bundle {key} SHA-256 is invalid or missing")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Recognizer bundle {key} SHA-256 mismatch: {actual} != {expected}")
    return path, entry


def _load_phones(path: Path, *, blank_id: int) -> tuple[str, ...]:
    payload = _json_object(path, label="Recognizer bundle vocabulary")
    raw: object = payload.get("phones", payload.get("vocabulary", payload.get("tokens")))
    if raw is None and isinstance(payload.get("token_to_id"), Mapping):
        raw = payload["token_to_id"]
    if isinstance(raw, Mapping):
        pairs: list[tuple[int, str]] = []
        for token, value in raw.items():
            if isinstance(value, bool):
                raise ValueError("Recognizer bundle vocabulary IDs must be integers")
            try:
                pairs.append((int(value), str(token).strip()))
            except (TypeError, ValueError) as exc:
                raise ValueError("Recognizer bundle vocabulary IDs must be integers") from exc
        pairs.sort(key=lambda item: item[0])
        if [index for index, _ in pairs] != list(range(len(pairs))):
            raise ValueError("Recognizer bundle vocabulary IDs must be contiguous from zero")
        phones = tuple(token for _, token in pairs)
    elif isinstance(raw, (list, tuple)):
        phones = tuple(str(item).strip() for item in raw)
    else:
        raise ValueError("Recognizer bundle vocabulary requires a phones list or token_to_id map")
    if not phones or any(not phone for phone in phones) or len(set(phones)) != len(phones):
        raise ValueError("Recognizer bundle vocabulary must contain unique non-empty phones")
    if blank_id < 0 or blank_id >= len(phones):
        raise ValueError("Recognizer bundle blank_id is outside the vocabulary")
    return phones


def _load_bundle_manifest(root: Path) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    manifest = _json_object(root / "manifest.json", label="Recognizer bundle manifest")
    if manifest.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError(f"Recognizer bundle schema must be {BUNDLE_SCHEMA!r}")
    backbone = manifest.get("backbone")
    if not isinstance(backbone, Mapping):
        raise ValueError("Recognizer bundle backbone metadata is required")
    backbone_path = _safe_bundle_path(root, backbone.get("path"), label="backbone")
    if not backbone_path.is_dir():
        raise ValueError(f"Recognizer bundle backbone directory is missing: {backbone_path}")
    checkpoint_path, _ = _manifest_entry(manifest, "checkpoint", root=root)
    vocab_path, _ = _manifest_entry(manifest, "vocab", root=root)
    license_path, _ = _manifest_entry(manifest, "license_manifest", root=root)
    license_payload = _json_object(license_path, label="Recognizer bundle license manifest")
    license_schema = str(license_payload.get("schema_version", "")).strip()
    if license_schema not in {"corpus_license_manifest_v1", "model_license_manifest_v1"}:
        raise ValueError("Recognizer bundle license manifest schema is unsupported")
    _validate_license_entries(license_payload, manifest=manifest)
    if manifest.get("manual_vsqx_override_allowed") is True:
        raise ValueError("Recognizer bundle cannot enable manual VSQX overrides")
    if manifest.get("lyrics_used") is True:
        raise ValueError("Recognizer bundle cannot use lyrics as recognizer input")
    return manifest, backbone_path, checkpoint_path, vocab_path, license_path


def _validate_license_entries(
    payload: Mapping[str, Any], *, manifest: Mapping[str, Any]
) -> None:
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("Recognizer bundle license manifest requires non-empty entries")
    covered_hashes: set[str] = set()
    required_fields = {
        "asset_id",
        "asset_sha256",
        "source",
        "revision",
        "license_id",
        "license_source",
        "status",
        "approved_uses",
    }
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Recognizer bundle license entry {index} must be an object")
        missing = sorted(
            field for field in required_fields if field not in raw or raw.get(field) in (None, "")
        )
        if missing:
            raise ValueError(
                f"Recognizer bundle license entry {index} is incomplete: {', '.join(missing)}"
            )
        digest = str(raw.get("asset_sha256") or "").strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Recognizer bundle license entry {index} has invalid asset_sha256")
        uses = raw.get("approved_uses")
        if not isinstance(uses, list) or "inference" not in {
            str(value).strip().casefold() for value in uses
        }:
            raise ValueError(f"Recognizer bundle license entry {index} does not approve inference")
        if str(raw.get("status") or "").strip().casefold() != "approved":
            raise ValueError(f"Recognizer bundle license entry {index} is not approved")
        covered_hashes.add(digest)
    required_hashes = {
        str(manifest[key]["sha256"]).strip().lower()
        for key in ("checkpoint", "vocab")
        if isinstance(manifest.get(key), Mapping)
    }
    backbone = manifest.get("backbone")
    if isinstance(backbone, Mapping) and backbone.get("sha256"):
        required_hashes.add(str(backbone["sha256"]).strip().lower())
    uncovered = sorted(required_hashes - covered_hashes)
    if uncovered:
        raise ValueError(
            "Recognizer bundle license manifest does not cover resource SHA-256: "
            + ", ".join(uncovered)
        )


def _number(value: object, *, label: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Recognizer bundle {label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Recognizer bundle {label} must be numeric") from exc
    if not math.isfinite(number) or (positive and number <= 0.0):
        requirement = "positive and finite" if positive else "finite"
        raise ValueError(f"Recognizer bundle {label} must be {requirement}")
    return number


def load_recognizer_bundle(bundle_dir: Path | str) -> RecognizerBundle:
    """Validate a local bundle and return resolved runtime resources."""

    root = Path(bundle_dir).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise FileNotFoundError(f"Recognizer bundle directory not found: {root}")
    manifest, backbone_path, checkpoint_path, vocab_path, license_path = _load_bundle_manifest(root)
    backbone = manifest["backbone"]
    family = str(backbone.get("family", "")).strip()
    model_id = str(backbone.get("model_id", backbone.get("id", ""))).strip()
    revision = str(backbone.get("revision", "")).strip()
    if not family or not model_id or not revision:
        raise ValueError("Recognizer bundle backbone family, model_id, and revision are required")
    vocab_entry = manifest["vocab"]
    blank_value = vocab_entry.get("blank_id", 0)
    if isinstance(blank_value, bool):
        raise ValueError("Recognizer bundle blank_id must be an integer")
    try:
        blank_id = int(blank_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Recognizer bundle blank_id must be an integer") from exc
    phones = _load_phones(vocab_path, blank_id=blank_id)
    if blank_id != 0 or phones[0] != "<blank>":
        raise ValueError("Recognizer bundle v1 requires '<blank>' at vocabulary index 0")
    audio = manifest.get("audio", {})
    if not isinstance(audio, Mapping):
        raise ValueError("Recognizer bundle audio metadata must be an object")
    sample_rate = int(audio.get("sample_rate", _FIXED_SAMPLE_RATE))
    chunk_sec = _number(audio.get("chunk_sec", _FIXED_CHUNK_SEC), label="audio.chunk_sec", positive=True)
    overlap_sec = _number(audio.get("overlap_sec", _FIXED_OVERLAP_SEC), label="audio.overlap_sec")
    if sample_rate != _FIXED_SAMPLE_RATE or chunk_sec != _FIXED_CHUNK_SEC or overlap_sec != _FIXED_OVERLAP_SEC:
        raise ValueError("Recognizer bundle audio contract must be 16 kHz, 8.0 s chunks, and 1.0 s overlap")
    if overlap_sec >= chunk_sec:
        raise ValueError("Recognizer bundle audio overlap must be shorter than chunk length")
    calibration = manifest.get("confidence_calibration", {})
    if not isinstance(calibration, Mapping):
        raise ValueError("Recognizer bundle confidence calibration must be an object")
    temperature = _number(calibration.get("temperature", 1.0), label="confidence_calibration.temperature", positive=True)
    return RecognizerBundle(
        root=root,
        backbone_path=backbone_path,
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        license_manifest_path=license_path,
        backbone_family=family,
        backbone_id=model_id,
        backbone_revision=revision,
        phones=phones,
        blank_id=blank_id,
        sample_rate=sample_rate,
        chunk_sec=chunk_sec,
        overlap_sec=overlap_sec,
        temperature=temperature,
        manifest=manifest,
    )


def _load_mono_audio(path: Path, *, target_rate: int) -> Any:
    try:
        import soundfile as sf
        import torch
    except ImportError as exc:
        raise RuntimeError("Cantonese singing recognizer requires soundfile and torch") from exc
    try:
        audio, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise RuntimeError(f"Cantonese singing audio could not be decoded: {path}") from exc
    mono = torch.from_numpy(audio).mean(dim=1)
    if int(source_rate) != target_rate:
        mono = _resample_waveform(mono, int(source_rate), target_rate, torch)
    return mono.contiguous()


def _resample_waveform(waveform: Any, source_rate: int, target_rate: int, torch: Any) -> Any:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Audio sample rates must be positive")
    if source_rate == target_rate:
        return waveform
    try:
        import torchaudio
    except (ImportError, OSError):
        target_length = max(1, int(round(int(waveform.numel()) * target_rate / source_rate)))
        return torch.nn.functional.interpolate(
            waveform.reshape(1, 1, -1), size=target_length, mode="linear", align_corners=False
        ).reshape(-1).contiguous()
    return torchaudio.functional.resample(waveform, source_rate, target_rate).contiguous()


def _iter_audio_chunks(audio: Any, *, sample_rate: int, chunk_sec: float, overlap_sec: float) -> list[tuple[int, Any]]:
    chunk_size = max(1, int(round(chunk_sec * sample_rate)))
    step_size = max(1, int(round((chunk_sec - overlap_sec) * sample_rate)))
    sample_count = int(audio.shape[0])
    chunks: list[tuple[int, Any]] = []
    offset = 0
    while offset < sample_count:
        end = min(sample_count, offset + chunk_size)
        chunks.append((offset, audio[offset:end]))
        if end >= sample_count:
            break
        offset += step_size
    return chunks


def _logit_rows(logits: Any) -> list[list[float]]:
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().tolist()
    elif hasattr(logits, "tolist"):
        logits = logits.tolist()
    rows = list(logits)
    if len(rows) == 1 and rows and isinstance(rows[0], (list, tuple)) and rows[0] and isinstance(rows[0][0], (list, tuple)):
        rows = rows[0]
    if not rows or not all(isinstance(row, (list, tuple)) for row in rows):
        raise ValueError("Cantonese singing recognizer returned no logit frames")
    return [[float(value) for value in row] for row in rows]


def _softmax(values: Sequence[float], *, temperature: float) -> list[float]:
    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("Recognizer confidence temperature must be positive")
    scaled = [float(value) / temperature for value in values]
    pivot = max(scaled)
    weights = [math.exp(value - pivot) for value in scaled]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Recognizer logits produced invalid posterior mass")
    return [value / total for value in weights]


def _aggregate_candidates(frames: Sequence[Sequence[float]], *, phones: Sequence[str], blank_id: int) -> tuple[tuple[str, float], ...]:
    means = [sum(frame[index] for frame in frames) / len(frames) for index in range(len(phones))]
    ranked = sorted(
        (index for index in range(len(means)) if index != blank_id),
        key=lambda index: (-means[index], phones[index]),
    )
    return tuple((phones[index], max(0.0, min(1.0, means[index]))) for index in ranked[:3])


def _decode_chunk_logits(
    logits: Any,
    *,
    phones: Sequence[str],
    blank_id: int,
    temperature: float,
    chunk_index: int,
    chunk_offset_sec: float,
    frame_hop_sec: float,
    audio_end_sec: float,
) -> list[_DecodedRun]:
    rows = _logit_rows(logits)
    if frame_hop_sec <= 0.0 or not math.isfinite(frame_hop_sec):
        raise ValueError("Cantonese singing recognizer returned an invalid frame hop")
    if any(len(row) != len(phones) for row in rows):
        raise ValueError("Recognizer logit width does not match bundle vocabulary")
    posteriors = [_softmax(row, temperature=temperature) for row in rows]
    winners = [max(range(len(row)), key=row.__getitem__) for row in posteriors]
    runs: list[_DecodedRun] = []
    start = 0
    while start < len(winners):
        winner = winners[start]
        end = start + 1
        while end < len(winners) and winners[end] == winner:
            end += 1
        if winner != blank_id:
            candidates = _aggregate_candidates(posteriors[start:end], phones=phones, blank_id=blank_id)
            start_sec = chunk_offset_sec + start * frame_hop_sec
            end_sec = min(audio_end_sec, chunk_offset_sec + end * frame_hop_sec)
            if candidates and end_sec > start_sec:
                runs.append(
                    _DecodedRun(
                        start_sec=start_sec,
                        end_sec=end_sec,
                        candidates=candidates,
                        chunk_index=chunk_index,
                        frame_start=start,
                        frame_end=end,
                        frame_hop_sec=frame_hop_sec,
                    )
                )
        start = end
    return runs


def _same_phone_overlap(left: _DecodedRun, right: _DecodedRun) -> float:
    if left.symbol != right.symbol:
        return 0.0
    intersection = max(0.0, min(left.end_sec, right.end_sec) - max(left.start_sec, right.start_sec))
    shorter = min(left.end_sec - left.start_sec, right.end_sec - right.start_sec)
    return intersection / max(shorter, 1e-9)


def _slice_run(run: _DecodedRun, start_sec: float, end_sec: float) -> _DecodedRun:
    return _DecodedRun(
        start_sec,
        end_sec,
        run.candidates,
        run.chunk_index,
        run.frame_start,
        run.frame_end,
        run.frame_hop_sec,
    )


def _merge_overlapping_runs(runs: Sequence[_DecodedRun]) -> list[_DecodedRun]:
    valid = [run for run in runs if run.end_sec > run.start_sec]
    boundaries = sorted({value for run in valid for value in (run.start_sec, run.end_sec)})
    resolved: list[_DecodedRun] = []
    for start_sec, end_sec in zip(boundaries, boundaries[1:]):
        if end_sec <= start_sec:
            continue
        active = [
            run
            for run in valid
            if run.start_sec < end_sec and run.end_sec > start_sec
        ]
        if not active:
            continue
        winner = min(
            active,
            key=lambda run: (
                -run.confidence,
                run.chunk_index,
                run.frame_start,
                run.symbol,
            ),
        )
        piece = _slice_run(winner, start_sec, end_sec)
        if resolved and resolved[-1].symbol == piece.symbol and math.isclose(
            resolved[-1].end_sec, piece.start_sec, abs_tol=1e-9
        ):
            previous = resolved[-1]
            evidence = piece if piece.confidence > previous.confidence else previous
            resolved[-1] = _DecodedRun(
                previous.start_sec,
                piece.end_sec,
                evidence.candidates,
                evidence.chunk_index,
                evidence.frame_start,
                evidence.frame_end,
                evidence.frame_hop_sec,
            )
        else:
            resolved.append(piece)
    return resolved


def _normalized_backbone_family(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "wavlm": "wavlm",
        "wavlm_encoder": "wavlm",
        "wav2vec2": "wav2vec2_xls_r",
        "wav2vec2_xls_r": "wav2vec2_xls_r",
        "xls_r": "wav2vec2_xls_r",
        "mert": "mert",
        "mert_encoder": "mert",
        "whisper": "whisper_encoder",
        "whisper_encoder": "whisper_encoder",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported recognizer backbone family: {value!r}") from exc


def _validate_checkpoint_contract(payload: Mapping[str, Any], bundle: RecognizerBundle) -> tuple[int, Mapping[str, Any], Mapping[str, Any]]:
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError(f"Recognizer checkpoint schema must be {CHECKPOINT_SCHEMA!r}")
    checkpoint_phones = tuple(str(value) for value in payload.get("phones", []))
    if checkpoint_phones != bundle.phones:
        raise ValueError("Recognizer checkpoint vocabulary does not match bundle vocabulary")
    if payload.get("blank_id") != bundle.blank_id:
        raise ValueError("Recognizer checkpoint blank_id does not match bundle vocabulary")
    if bundle.blank_id != 0 or not bundle.phones or bundle.phones[0] != "<blank>":
        raise ValueError("Recognizer checkpoint v1 requires '<blank>' at vocabulary index 0")
    expected_vocab = {phone: index for index, phone in enumerate(bundle.phones) if index != bundle.blank_id}
    checkpoint_vocab = payload.get("vocab")
    if not isinstance(checkpoint_vocab, Mapping) or dict(checkpoint_vocab) != expected_vocab:
        raise ValueError("Recognizer checkpoint non-blank vocabulary IDs do not match bundle order")
    try:
        hidden_size = int(payload.get("hidden_size", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Recognizer checkpoint requires a positive hidden_size") from exc
    if hidden_size <= 0:
        raise ValueError("Recognizer checkpoint requires a positive hidden_size")
    backbone_state = payload.get("backbone_state_dict")
    head_state = payload.get("head_state_dict")
    configuration = payload.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("Recognizer checkpoint requires configuration metadata")
    if not isinstance(backbone_state, Mapping):
        raise ValueError("Recognizer checkpoint backbone_state_dict must be an object")
    if not backbone_state and configuration.get("backbone_frozen") is not True:
        raise ValueError("Recognizer checkpoint requires backbone_state_dict unless the backbone is frozen")
    if not isinstance(head_state, Mapping) or not head_state:
        raise ValueError("Recognizer checkpoint requires head_state_dict")
    expected_model = str(payload.get("model_sha256") or "").strip().lower()
    if len(expected_model) != 64 or any(char not in "0123456789abcdef" for char in expected_model):
        raise ValueError("Recognizer checkpoint requires a model_sha256 digest")
    if _state_sha256(head_state) != expected_model:
        raise ValueError("Recognizer checkpoint model-state SHA-256 mismatch")
    duplicate_state = payload.get("state_dict")
    if duplicate_state is not None and (
        not isinstance(duplicate_state, Mapping)
        or _state_sha256(duplicate_state) != expected_model
    ):
        raise ValueError("Recognizer checkpoint state_dict disagrees with head_state_dict")
    return hidden_size, backbone_state, head_state


def _validate_frozen_backbone_identity(payload: Mapping[str, Any], bundle: RecognizerBundle) -> None:
    configuration = payload.get("configuration")
    if not isinstance(configuration, Mapping) or configuration.get("backbone_frozen") is not True:
        return
    expected = str(configuration.get("backbone_sha256") or "").strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("Frozen-backbone checkpoint requires configuration.backbone_sha256")
    backbone = bundle.manifest.get("backbone")
    if not isinstance(backbone, Mapping):
        raise ValueError("Recognizer bundle backbone metadata is required")
    raw_weights_path = backbone.get("weights_path")
    declared = str(backbone.get("sha256") or "").strip().lower()
    if not raw_weights_path or len(declared) != 64 or any(char not in "0123456789abcdef" for char in declared):
        raise ValueError("Frozen-backbone bundle requires weights_path and sha256")
    weights_path = _safe_bundle_path(bundle.root, raw_weights_path, label="backbone weights")
    try:
        weights_path.relative_to(bundle.backbone_path)
    except ValueError as exc:
        raise ValueError("Recognizer bundle backbone weights must stay inside the backbone directory") from exc
    if not weights_path.is_file():
        raise ValueError(f"Recognizer bundle backbone weights are missing: {weights_path}")
    actual = sha256_file(weights_path)
    if actual != declared:
        raise ValueError(f"Recognizer bundle backbone SHA-256 mismatch: {actual} != {declared}")
    if actual != expected:
        raise ValueError("Recognizer checkpoint backbone SHA-256 does not match the bundle")


def _validate_backbone_hidden_size(backbone: Any, expected: int) -> None:
    config = getattr(backbone, "config", None)
    raw_actual = getattr(config, "hidden_size", getattr(config, "d_model", None))
    try:
        actual = int(raw_actual)
    except (TypeError, ValueError) as exc:
        raise ValueError("Recognizer backbone config exposes no hidden size") from exc
    if actual != expected:
        raise ValueError(f"Recognizer checkpoint hidden_size does not match backbone: {expected} != {actual}")


def _load_state_dict_strict(module: Any, state: Mapping[str, Any], *, label: str) -> None:
    try:
        module.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ValueError(f"Recognizer checkpoint {label} state is incompatible") from exc


def _build_torch_runtime(bundle: RecognizerBundle, device: str) -> CTCAcousticRuntime:
    return _TorchCTCRuntime(bundle, device=device)


class _TorchCTCRuntime:
    def __init__(self, bundle: RecognizerBundle, *, device: str) -> None:
        try:
            import torch
            from torch import nn
            from transformers import AutoFeatureExtractor, AutoModel, Wav2Vec2Model, WavLMModel
        except ImportError as exc:
            raise RuntimeError("Real Cantonese singing inference requires torch and transformers") from exc
        self.torch = torch
        self.device = self._resolve_device(device)
        try:
            checkpoint = load_trained_checkpoint(
                bundle.checkpoint_path,
                expected_schema=CHECKPOINT_SCHEMA,
                expected_vocabulary=bundle.phones,
                require_provenance=True,
            ).payload
        except (CheckpointLoadError, FileNotFoundError) as exc:
            raise RuntimeError(f"Cantonese singing checkpoint could not be loaded: {exc}") from exc
        _validate_frozen_backbone_identity(checkpoint, bundle)
        hidden_size, backbone_state, head_state = _validate_checkpoint_contract(checkpoint, bundle)
        family = _normalized_backbone_family(bundle.backbone_family)
        self.family = family
        kwargs = {"local_files_only": True, "trust_remote_code": False}
        if family == "wavlm":
            self.extractor = AutoFeatureExtractor.from_pretrained(str(bundle.backbone_path), **kwargs)
            self.backbone = WavLMModel.from_pretrained(str(bundle.backbone_path), **kwargs)
        elif family == "wav2vec2_xls_r":
            self.extractor = AutoFeatureExtractor.from_pretrained(str(bundle.backbone_path), **kwargs)
            self.backbone = Wav2Vec2Model.from_pretrained(str(bundle.backbone_path), **kwargs)
        elif family == "mert":
            self.extractor = AutoFeatureExtractor.from_pretrained(str(bundle.backbone_path), **kwargs)
            try:
                self.backbone = AutoModel.from_pretrained(
                    str(bundle.backbone_path),
                    local_files_only=True,
                    trust_remote_code=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    "MERT bundles requiring model-supplied Python remain quarantined; "
                    "use a reviewed native Transformers model class"
                ) from exc
        else:
            try:
                from transformers import WhisperFeatureExtractor, WhisperModel
            except ImportError as exc:
                raise RuntimeError("Whisper checkpoint requires transformers Whisper support") from exc
            self.extractor = WhisperFeatureExtractor.from_pretrained(str(bundle.backbone_path), local_files_only=True)
            wrapper = WhisperModel.from_pretrained(str(bundle.backbone_path), local_files_only=True)
            self.backbone = wrapper.encoder
        _validate_backbone_hidden_size(self.backbone, hidden_size)
        if backbone_state:
            _load_state_dict_strict(self.backbone, backbone_state, label="backbone")
        self.head = nn.Linear(hidden_size, len(bundle.phones))
        _load_state_dict_strict(self.head, head_state, label="CTC head")
        self.backbone.eval().to(self.device)
        self.head.eval().to(self.device)

    def _resolve_device(self, requested: str) -> Any:
        normalized = str(requested or "auto").strip().lower()
        if normalized not in {"auto", "cpu", "cuda"}:
            raise ValueError("Recognizer device must be one of: auto, cuda, cpu")
        if normalized == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        use_cuda = normalized == "cuda" or (normalized == "auto" and self.torch.cuda.is_available())
        if use_cuda and (
            not str(self.torch.__version__).startswith(_VERIFIED_TORCH_PREFIX)
            or str(self.torch.version.cuda) != _VERIFIED_CUDA_VERSION
        ):
            raise RuntimeError("CUDA inference requires the verified torch 2.13.0+cu126 / CUDA 12.6 stack")
        return self.torch.device("cuda" if use_cuda else "cpu")

    def infer(self, audio: Any, *, sample_rate: int) -> tuple[Any, float]:
        with self.torch.inference_mode():
            processor_rate = 24_000 if self.family == "mert" else sample_rate
            model_audio = (
                _resample_waveform(audio, sample_rate, processor_rate, self.torch)
                if processor_rate != sample_rate
                else audio
            )
            array = model_audio.detach().cpu().numpy() if hasattr(model_audio, "detach") else model_audio
            inputs = self.extractor(
                array,
                sampling_rate=processor_rate,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if self.family == "whisper_encoder":
                input_values = inputs["input_features"] if isinstance(inputs, Mapping) else inputs.input_features
                kwargs: dict[str, Any] = {"input_features": input_values.to(self.device)}
            else:
                input_values = inputs["input_values"] if isinstance(inputs, Mapping) else inputs.input_values
                kwargs = {"input_values": input_values.to(self.device)}
            mask = inputs.get("attention_mask") if isinstance(inputs, Mapping) else getattr(inputs, "attention_mask", None)
            if mask is not None and self.family != "whisper_encoder":
                kwargs["attention_mask"] = mask.to(self.device)
            hidden = self.backbone(**kwargs).last_hidden_state
            if self.family == "whisper_encoder" and mask is not None:
                mask_frames = int(mask.shape[-1])
                valid_input_frames = int(mask.sum().item())
                valid_hidden_frames = max(
                    1,
                    math.ceil(valid_input_frames * int(hidden.shape[1]) / max(mask_frames, 1)),
                )
                hidden = hidden[:, :valid_hidden_frames]
            logits = self.head(hidden)[0]
        frames = int(logits.shape[0])
        if frames <= 0:
            raise ValueError("Recognizer backbone returned no frames")
        duration = int(audio.shape[0]) / sample_rate
        return logits.cpu(), duration / frames


class CantoneseSingingCTCRecognizer(UniversalPhoneRecognizer):
    """Decode CTC phone runs from a local, hash-validated recognizer bundle."""

    def __init__(
        self,
        bundle_path: Path | str,
        *,
        device: str = "auto",
        runtime_factory: Callable[[RecognizerBundle, str], CTCAcousticRuntime] | None = None,
    ) -> None:
        self.bundle = load_recognizer_bundle(bundle_path)
        self.device = str(device or "auto").strip().lower()
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Recognizer device must be one of: auto, cuda, cpu")
        self.runtime_factory = runtime_factory or _build_torch_runtime
        self._runtime: CTCAcousticRuntime | None = None

    def recognize(self, vocal_audio_path: Path, lyrics: str | None = None) -> list[PhoneSegment]:
        audio_path = Path(vocal_audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Vocal audio file not found: {audio_path}")
        if self._runtime is None:
            self._runtime = self.runtime_factory(self.bundle, self.device)
        audio = _load_mono_audio(audio_path, target_rate=self.bundle.sample_rate)
        audio_end_sec = int(audio.shape[0]) / self.bundle.sample_rate
        runs: list[_DecodedRun] = []
        for chunk_index, (offset, chunk) in enumerate(
            _iter_audio_chunks(
                audio,
                sample_rate=self.bundle.sample_rate,
                chunk_sec=self.bundle.chunk_sec,
                overlap_sec=self.bundle.overlap_sec,
            )
        ):
            logits, frame_hop_sec = self._runtime.infer(chunk, sample_rate=self.bundle.sample_rate)
            runs.extend(
                _decode_chunk_logits(
                    logits,
                    phones=self.bundle.phones,
                    blank_id=self.bundle.blank_id,
                    temperature=self.bundle.temperature,
                    chunk_index=chunk_index,
                    chunk_offset_sec=offset / self.bundle.sample_rate,
                    frame_hop_sec=float(frame_hop_sec),
                    audio_end_sec=audio_end_sec,
                )
            )
        return self._segments(_merge_overlapping_runs(runs), lyrics=lyrics)

    def _segments(self, runs: Sequence[_DecodedRun], *, lyrics: str | None) -> list[PhoneSegment]:
        checkpoint_hash = sha256_file(self.bundle.checkpoint_path)
        result: list[PhoneSegment] = []
        for run in runs:
            candidates = [PhoneCandidate(symbol=symbol, confidence=confidence) for symbol, confidence in run.candidates]
            result.append(
                PhoneSegment(
                    time_range=TimeRange(start_sec=run.start_sec, end_sec=run.end_sec),
                    candidate_phones=candidates,
                    confidence=candidates[0].confidence,
                    metadata={
                        "source_mode": "real",
                        "backend_kind": "cantonese-singing-ctc",
                        "adapter_kind": "cantonese-singing-ctc",
                        "real_backend": True,
                        "real_model_output": True,
                        "approximation_only": False,
                        "candidate_aggregation": "mean_ctc_frame_posterior",
                        "raw_model_output_kind": "cantonese_singing_ctc_model",
                        "checkpoint_path": str(self.bundle.checkpoint_path),
                        "checkpoint_sha256": checkpoint_hash,
                        "bundle_path": str(self.bundle.root),
                        "bundle_schema_version": BUNDLE_SCHEMA,
                        "backbone_family": self.bundle.backbone_family,
                        "backbone_id": self.bundle.backbone_id,
                        "backbone_revision": self.bundle.backbone_revision,
                        "chunk_index": run.chunk_index,
                        "frame_start": run.frame_start,
                        "frame_end": run.frame_end,
                        "frame_hop_sec": run.frame_hop_sec,
                        "lyrics_used_as_recognizer_input": False,
                        "lyrics_used": False,
                        "lyrics_supplied": bool(lyrics),
                        "truth_layer": "phones_ipa_articulatory",
                        "export_layer": "xsampa_or_engine_only",
                    },
                )
            )
        return result


RealCantoneseSingingCTCRecognizer = CantoneseSingingCTCRecognizer


__all__ = [
    "BUNDLE_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "CantoneseSingingCTCRecognizer",
    "CTCAcousticRuntime",
    "RealCantoneseSingingCTCRecognizer",
    "RecognizerBundle",
    "_decode_chunk_logits",
    "_load_mono_audio",
    "_normalized_backbone_family",
    "load_recognizer_bundle",
]
