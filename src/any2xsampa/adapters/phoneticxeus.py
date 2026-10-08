"""Lightweight PhoneticXeus evidence codec; the model lives in another process."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from any2xsampa.adapters.audio_provenance import validate_audio_binding
from any2xsampa.core.interfaces import UniversalPhoneRecognizer
from any2xsampa.core.models import PhoneSegment
from any2xsampa.training.limited_data_artifacts import finite_number, read_json, sha256_file, verify_file


MODEL_ID = "changelinglab/PhoneticXeus"
REVISION = "8d83dee94817a07dc150f87d08f7e0ee01bdb66d"
WEIGHT_SHA256 = "ad58bf20a60e9d0380327bd8b2d0e8e90a9b8de2adccbfb479f9b21ea85eda18"


def verify_snapshot(root: Path) -> dict[str, Any]:
    lock = read_json(root / "snapshot-lock.json")
    if (
        lock.get("schema_version") != "phoneticxeus_snapshot_lock_v1"
        or lock.get("model_id") != MODEL_ID
        or lock.get("revision") != REVISION
        or lock.get("license") != "apache-2.0"
    ):
        raise ValueError("snapshot does not match the authorized model/revision/license")
    members = lock["files"]
    names: set[str] = set()
    for row in members:
        relative = Path(row["file"])
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()) or row["file"] in names:
            raise ValueError("invalid or duplicate snapshot member")
        names.add(row["file"])
        verify_file(target, row["sha256"])
        if target.stat().st_size != row["bytes"]:
            raise ValueError(f"snapshot member size mismatch: {relative}")
    required = {
        "model.safetensors", "config.json", "modeling_phoneticxeus.py",
        "configuration_phoneticxeus.py", "src/model/xeusphoneme/resources/ipa_vocab.json",
        "src/model/xeusphoneme/resources/xeus_config.yaml",
    }
    if not required <= names:
        raise ValueError("snapshot is incomplete")
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*")
        if path.is_file() and path.name != "snapshot-lock.json"
    }
    if actual != names:
        raise ValueError("snapshot contains undeclared files")
    weight = next(row for row in members if row["file"] == "model.safetensors")
    if weight["sha256"] != WEIGHT_SHA256:
        raise ValueError("unauthorized model weights")
    return {
        "model_id": MODEL_ID, "revision": REVISION, "license": "apache-2.0",
        "snapshot_lock_sha256": sha256_file(root / "snapshot-lock.json"),
        "weights_sha256": WEIGHT_SHA256,
        "vocabulary_sha256": sha256_file(root / "src/model/xeusphoneme/resources/ipa_vocab.json"),
        "files_verified": len(members),
    }


def ctc_collapse(ids: Sequence[int], blank_id: int = 0) -> list[int]:
    output: list[int] = []
    previous: int | None = None
    for value in ids:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("CTC IDs must be non-negative integers")
        if value != previous and value != blank_id:
            output.append(value)
        previous = value
    return output


def convolution_geometry(shapes: Sequence[tuple[int, int]], samples: int) -> dict[str, int]:
    count, receptive, stride = samples, 1, 1
    for kernel, step in shapes:
        if kernel < 1 or step < 1:
            raise ValueError("invalid convolution geometry")
        count = (count - kernel) // step + 1
        receptive += (kernel - 1) * stride
        stride *= step
    if count < 1:
        raise ValueError("waveform too short for the acoustic frontend")
    return {"frame_count": count, "stride_samples": stride, "receptive_field_samples": receptive}


def emission_payload(
    logits: Any,
    vocabulary: Mapping[str, int],
    *,
    input_audio: Mapping[str, Any],
    crop: Mapping[str, Any],
    model_metadata: Mapping[str, Any],
    geometry: Mapping[str, int],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[1] != len(vocabulary) or values.shape[0] < 1:
        raise ValueError("CTC logits must be non-empty [frames, vocabulary]")
    if not np.isfinite(values).all():
        raise ValueError("CTC logits contain non-finite values")
    if values.shape[0] != geometry["frame_count"]:
        raise ValueError("encoder frame count differs from verified convolution geometry")
    if sorted(vocabulary.values()) != list(range(len(vocabulary))) or vocabulary.get("<blank>") != 0:
        raise ValueError("invalid CTC vocabulary")
    labels = [""] * len(vocabulary)
    for label, index in vocabulary.items():
        labels[index] = label
    offset = finite_number(crop["effective_start_sec"], "crop offset", minimum=0)
    end_limit = finite_number(crop["effective_end_sec"], "crop end", minimum=0)
    if end_limit <= offset:
        raise ValueError("invalid crop interval")
    stride = geometry["stride_samples"] / 16000
    center_offset = (geometry["receptive_field_samples"] - 1) / 32000
    probabilities = np.exp(values.astype(np.float64) - values.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    raw_ids = values.argmax(axis=1).tolist()
    collapsed = ctc_collapse(raw_ids)
    phone_ids = [index for index, label in enumerate(labels) if not label.startswith("<")]
    segments: list[dict[str, Any]] = []
    first = 0
    for stop in range(1, len(raw_ids) + 1):
        if stop < len(raw_ids) and raw_ids[stop] == raw_ids[first]:
            continue
        label_id = raw_ids[first]
        if label_id in phone_ids:
            means = probabilities[first:stop].mean(axis=0)
            candidates = sorted(phone_ids, key=lambda index: (-means[index], index))[:8]
            start = max(offset, offset + center_offset + (first - 0.5) * stride)
            end = min(end_limit, offset + center_offset + (stop - 0.5) * stride)
            if end <= start:
                raise ValueError("CTC emission time outside source crop")
            segments.append({
                "start_sec": start,
                "end_sec": end,
                "confidence": float(means[label_id]),
                "candidate_phones": [{"symbol": labels[index], "confidence": float(means[index])} for index in candidates],
                "posterior_summary": {
                    "source": "phoneticxeus:full_vocabulary_softmax",
                    "blank_mass": float(means[0]),
                    "retained_candidate_mass": float(sum(means[index] for index in candidates)),
                    "renormalized_after_truncation": False,
                },
                "raw_label_trace": {
                    "source": "phoneticxeus:raw_ctc_logits",
                    "start_frame": first, "end_frame_exclusive": stop,
                    "label_id": label_id, "label": labels[label_id],
                },
                "boundary_uncertainty": {
                    "timestamp_kind": "ctc_emission_frame_cell",
                    "frame_quantization_sec": stride,
                    "phone_duration_verified": False,
                },
                "notes": ["Sparse CTC emission evidence, not a continuous phone boundary."],
            })
        first = stop
    return {
        "schema_version": "external_posterior_json_v1",
        "producer": {
            "name": "phoneticxeus-audio-only", "backend_kind": "phoneticxeus",
            "real_backend": True, "real_model_output": True, "approximation_only": False,
            "diagnostic_baseline": True, "qualified_recognizer": False,
            "raw_model_output": "full_ctc_logits", "phone_representation": "IPA",
            "model_metadata": dict(model_metadata), "runtime": dict(runtime),
            "input_audio": dict(input_audio), "crop": dict(crop),
            "lyrics_used": False, "score_used": False, "rendered_reference_used": False,
        },
        "segments": segments,
        "decoding": {
            "algorithm": "greedy_ctc_v1", "blank_id": 0,
            "status": "phones" if segments else "no_phone_emissions",
            "collapsed_raw_ids": collapsed,
            "collapsed_raw_labels": [labels[index] for index in collapsed],
            "special_emissions": [labels[index] for index in collapsed if labels[index].startswith("<")],
            "frame_count": len(raw_ids),
            "blank_frame_count": raw_ids.count(0),
            "geometry": dict(geometry),
            "special_tokens_retained_in_raw_evidence": True,
        },
    }


class PhoneticXeusPosteriorRecognizer(UniversalPhoneRecognizer):
    """Replay verified model evidence without loading the model or its runtime."""

    def __init__(self, posterior_path: Path, *, fixture: bool = False, vocabulary_path: Path | None = None) -> None:
        self.posterior_path = posterior_path
        self.fixture = fixture
        self.vocabulary_path = vocabulary_path

    def recognize(self, vocal_audio_path: Path, lyrics: str | None = None) -> list[PhoneSegment]:
        if lyrics:
            raise ValueError("PhoneticXeus is audio-only; lyric input is not permitted")
        payload = read_json(self.posterior_path)
        if payload.get("test_outputs_sealed"):
            raise ValueError("sealed final-test outputs cannot be opened by the development adapter")
        producer = payload.get("producer", {})
        if producer.get("backend_kind") != "phoneticxeus" or payload.get("schema_version") != "external_posterior_json_v1":
            raise ValueError("expected PhoneticXeus posterior evidence")
        binding = validate_audio_binding(producer.get("input_audio"), vocal_audio_path)
        if not binding["verified"]:
            raise ValueError("PhoneticXeus replay requires a bound source audio hash")
        raw = payload.get("raw_evidence")
        if not isinstance(raw, dict):
            raise ValueError("PhoneticXeus replay requires raw evidence")
        raw_path = (self.posterior_path.parent / raw["file"]).resolve()
        if not raw_path.is_relative_to(self.posterior_path.parent.resolve()):
            raise ValueError("raw evidence must remain adjacent to its posterior")
        verify_file(raw_path, raw["sha256"])
        vocabulary_path = self.vocabulary_path
        if vocabulary_path is None:
            name = raw.get("vocabulary_file")
            if not isinstance(name, str):
                raise ValueError("replay requires a hash-bound vocabulary file")
            vocabulary_path = (self.posterior_path.parent / name).resolve()
            if not vocabulary_path.is_relative_to(self.posterior_path.parent.resolve()):
                raise ValueError("adjacent vocabulary path escapes the artifact directory")
        vocabulary_hash = producer.get("model_metadata", {}).get("vocabulary_sha256")
        verify_file(vocabulary_path, vocabulary_hash)
        self._verify_raw_projection(payload, raw_path, read_json(vocabulary_path))
        if not payload.get("segments"):
            if payload.get("decoding", {}).get("status") != "no_phone_emissions" or payload["decoding"]["frame_count"] < 1:
                raise ValueError("unexplained empty recognizer output")
            return []
        from any2xsampa.adapters.recognizer_real import _build_segments

        segments = _build_segments(
            payload, source_mode="fixture" if self.fixture else "real",
            backend_name="phoneticxeus", source_label="verified PhoneticXeus posterior",
            lyrics=None,
        )
        for segment in segments:
            segment.metadata["audio_binding"] = binding
            segment.metadata["phone_representation"] = "IPA"
            segment.metadata["raw_evidence"] = raw
            segment.metadata["qualified_recognizer"] = False
        return segments

    @staticmethod
    def _verify_raw_projection(payload: Mapping[str, Any], raw_path: Path, vocabulary: Mapping[str, int]) -> None:
        try:
            import numpy as np
            from safetensors.numpy import load_file
        except ImportError as exc:
            raise RuntimeError("PhoneticXeus tensor replay requires optional numpy and safetensors; the ordinary JSON fixture backend remains available") from exc

        if raw_path.stat().st_size > 64 * 1024 ** 2:
            raise ValueError("raw evidence exceeds the bounded clip artifact limit")
        tensors = load_file(str(raw_path))
        if "frame_logits" not in tensors or tensors["frame_logits"].dtype != np.float32:
            raise ValueError("replay requires float32 raw frame logits")
        producer = payload["producer"]
        expected = emission_payload(
            tensors["frame_logits"], vocabulary,
            input_audio=producer["input_audio"], crop=producer["crop"],
            model_metadata=producer["model_metadata"],
            geometry=payload["decoding"]["geometry"], runtime=producer["runtime"],
        )
        if payload["decoding"] != expected["decoding"] or len(payload["segments"]) != len(expected["segments"]):
            raise ValueError("posterior decoding differs from its raw logits")
        for actual, reference in zip(payload["segments"], expected["segments"]):
            for field in ("start_sec", "end_sec", "confidence"):
                actual_value = finite_number(actual[field], f"posterior {field}", minimum=0)
                if not np.isclose(actual_value, reference[field], rtol=1e-7, atol=1e-8):
                    raise ValueError(f"posterior {field} differs from raw emission evidence")
            if actual["raw_label_trace"] != reference["raw_label_trace"]:
                raise ValueError("posterior raw-label lineage differs from its logits")
            pairs = zip(actual["candidate_phones"], reference["candidate_phones"])
            if len(actual["candidate_phones"]) != len(reference["candidate_phones"]) or any(
                a["symbol"] != b["symbol"] or not np.isclose(a["confidence"], b["confidence"], rtol=1e-6, atol=1e-8)
                for a, b in pairs
            ):
                raise ValueError("posterior candidates differ from the full-vocabulary softmax")
