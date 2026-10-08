"""POWSM-CTC evidence codec; model-specific dependencies stay in its worker."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import math
import unicodedata

from any2xsampa.adapters.audio_provenance import validate_audio_binding
from any2xsampa.adapters.phoneticxeus import ctc_collapse
from any2xsampa.core.interfaces import UniversalPhoneRecognizer
from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange
from any2xsampa.training.limited_data_artifacts import read_json, verify_file, finite_number

MODEL_ID = "espnet/powsm_ctc"
REVISION = "f9d7f5c0e21f4e821b743eeab07344cb2d517d35"
WEIGHT_SHA256 = "d9d56a166e5dfe3088ca2080c8c3e2eb8124827166a68cd41b673963f99c9ef8"


def token_phone(token: str) -> str | None:
    """Only remove upstream phone wrappers; never guess an acoustic alias."""
    if token.startswith("/") and token.endswith("/") and len(token) > 2:
        return unicodedata.normalize("NFC", token[1:-1])
    return None


def decoded_tokens(ids: Sequence[int], labels: Sequence[str], *, legacy: bool = False) -> dict[str, Any]:
    collapsed = ctc_collapse(ids)
    if any(i >= len(labels) for i in collapsed):
        raise ValueError("CTC ID outside vocabulary")
    tokens = [labels[i] for i in collapsed]
    specials = [t for t in tokens if t.startswith("<") and t.endswith(">")]
    remaining = [t for t in tokens if not (t.startswith("<") and t.endswith(">"))]
    separators = [] if legacy else [t for t in remaining if t == "▁"]
    if not legacy:
        remaining = [t for t in remaining if t != "▁"]
    result = {"collapsed_raw_ids": collapsed, "collapsed_raw_labels": tokens,
            "special_emissions": specials,
            "prediction_phones": [token_phone(t) or "<nonphone:" + t + ">" for t in remaining],
            "nonphone_emissions": [t for t in remaining if token_phone(t) is None]}
    if not legacy:
        result["separator_emissions"] = separators
    return result


def emission_payload(logits: Any, labels: Sequence[str], *, audio: Mapping[str, Any],
                     crop: Mapping[str, Any], identity: Mapping[str, Any],
                     runtime: Mapping[str, Any], legacy: bool = False) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] != len(labels):
        raise ValueError("POWSM logits must be non-empty [frames, vocabulary]")
    if not np.isfinite(values).all() or len(set(labels)) != len(labels) or labels[0] != "<blank>":
        raise ValueError("invalid logits or vocabulary")
    offset = finite_number(crop["effective_start_sec"], "crop start", minimum=0)
    limit = finite_number(crop["effective_end_sec"], "crop end", minimum=0)
    if not 0 < limit - offset <= 20.000001 or limit > audio["duration_sec"] + 1e-6:
        raise ValueError("invalid source crop")
    raw_ids = values.argmax(-1).tolist()
    decoding = decoded_tokens(raw_ids, labels, legacy=legacy)
    probabilities = np.exp(values.astype(np.float64) - values.max(-1, keepdims=True))
    probabilities /= probabilities.sum(-1, keepdims=True)
    phone_ids = [i for i, t in enumerate(labels) if token_phone(t) is not None]
    segments, untimed = [], []
    first = 0
    for stop in range(1, len(raw_ids) + 1):
        if stop < len(raw_ids) and raw_ids[stop] == raw_ids[first]:
            continue
        label_id = raw_ids[first]
        phone = token_phone(labels[label_id])
        if phone is not None:
            # Two non-audio prefix positions precede centered-STFT/conv2d frames.
            # Conv kernels 3/2 then 3/2: center 3 hops, stride 4 hops; hop=10ms.
            start = offset + 0.03 + (first - 2 - 0.5) * 0.04
            end = offset + 0.03 + (stop - 2 - 0.5) * 0.04
            trace = {"start_frame": first, "end_frame_exclusive": stop,
                     "label_id": label_id, "label": labels[label_id]}
            if end <= offset or start >= limit:
                untimed.append({**trace, "phone": phone,
                                "reason": "prefix_position" if stop <= 2 else "zero_padding_position"})
            else:
                means = probabilities[first:stop].mean(0)
                candidates = sorted(phone_ids, key=lambda i: (-means[i], i))[:8]
                segments.append({"start_sec": max(offset, start), "end_sec": min(limit, end),
                    "confidence": float(means[label_id]),
                    "candidate_phones": [{"symbol": token_phone(labels[i]), "confidence": float(means[i])} for i in candidates],
                    "raw_label_trace": trace,
                    "boundary_uncertainty": {"timestamp_kind": "ctc_emission_frame_cell",
                        "phone_duration_verified": False, "frame_quantization_sec": 0.04},
                    "posterior_summary": {"blank_mass": float(means[0]),
                        "retained_candidate_mass": float(sum(means[i] for i in candidates)),
                        "normalization_inventory": len(labels), "renormalized_after_truncation": False}})
        first = stop
    decoding.update(algorithm="powsm_full_inventory_greedy_v1" if legacy else "powsm_full_inventory_greedy_v2", blank_id=0,
        frame_count=len(raw_ids), blank_frame_count=raw_ids.count(0),
        untimed_phone_emissions=untimed, padding_emissions_retained_in_scoring=True,
        status="phones" if decoding["prediction_phones"] else "no_phone_emissions",
        special_removal="upstream angle-bracket metadata only; raw tokens retained",
        normalization="phone wrappers removed; Unicode NFC only" if legacy else
            "phone wrappers removed; Unicode NFC; standalone SentencePiece space delimiter excluded")
    return {"schema_version": "external_posterior_json_v1",
        "producer": {"name": "powsm-ctc-audio-only", "backend_kind": "powsm_ctc",
            "real_backend": True, "real_model_output": True, "qualified_recognizer": False,
            "phone_representation": "IPA", "model_metadata": dict(identity),
            "input_audio": dict(audio), "crop": dict(crop), "runtime": dict(runtime),
            "lyrics_used": False, "score_used": False, "language_hint_used": False},
        "segments": segments, "decoding": decoding}


def adjacent_file(posterior: Path, name: str) -> Path:
    path = (posterior.parent / name).resolve()
    if not path.is_relative_to(posterior.parent.resolve()):
        raise ValueError("evidence path escapes its artifact directory")
    return path


def _same_projection(actual: Any, expected: Any) -> bool:
    """Allow only float64 rounding differences across independent NumPy runtimes."""
    if isinstance(expected, float):
        return (isinstance(actual, (float, int)) and not isinstance(actual, bool)
                and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12))
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(_same_projection(actual[k], v) for k, v in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_same_projection(a, b) for a, b in zip(actual, expected)))
    return type(actual) is type(expected) and actual == expected


class PowsmPosteriorRecognizer(UniversalPhoneRecognizer):
    """Replay hash-bound raw logits into the unchanged PhoneSegment contract."""
    def __init__(self, posterior_path: Path, *, fixture: bool = False) -> None:
        self.path = posterior_path
        self.fixture = fixture

    def recognize(self, vocal_audio_path: Path, lyrics: str | None = None) -> list[PhoneSegment]:
        if lyrics is not None:
            raise ValueError("POWSM inference accepts audio only")
        from safetensors.numpy import load_file
        import numpy as np

        payload = read_json(self.path)
        producer = payload.get("producer", {})
        identity = producer.get("model_metadata", {})
        if (payload.get("schema_version") != "external_posterior_json_v1"
                or producer.get("backend_kind") != "powsm_ctc"
                or identity.get("model_id") != MODEL_ID or identity.get("revision") != REVISION
                or identity.get("weight_sha256") != WEIGHT_SHA256):
            raise ValueError("unexpected POWSM model or evidence schema")
        if payload.get("test_outputs_sealed"):
            raise ValueError("sealed test evidence is not a development input")
        binding = validate_audio_binding(producer["input_audio"], vocal_audio_path)
        raw = payload["raw_evidence"]
        tensor_path = adjacent_file(self.path, raw["file"])
        vocab_path = adjacent_file(self.path, raw["vocabulary_file"])
        verify_file(tensor_path, raw["sha256"])
        verify_file(vocab_path, identity["vocabulary_sha256"])
        if tensor_path.stat().st_size > 96 * 1024 ** 2:
            raise ValueError("clip evidence exceeds POWSM size bound")
        tensors = load_file(str(tensor_path))
        if set(tensors) != {"frame_logits"} or tensors["frame_logits"].dtype != np.float32:
            raise ValueError("fixture requires full FP32 frame logits")
        labels = json.loads(vocab_path.read_text(encoding="utf-8"))
        if not isinstance(labels, list) or not all(isinstance(label, str) and label for label in labels):
            raise ValueError("POWSM vocabulary must be a non-empty string list")
        expected = emission_payload(tensors["frame_logits"], labels,
            audio=producer["input_audio"], crop=producer["crop"], identity=identity,
            runtime=producer["runtime"], legacy=payload.get("decoding", {}).get("algorithm") == "powsm_full_inventory_greedy_v1")
        if expected["decoding"] != payload["decoding"] or not _same_projection(payload["segments"], expected["segments"]):
            raise ValueError("posterior differs from its raw logits")
        if payload["decoding"]["untimed_phone_emissions"]:
            raise ValueError("phone emissions in non-audio positions cannot become timed PhoneSegments; retained in raw diagnostic")
        return [PhoneSegment(TimeRange(row["start_sec"], row["end_sec"]),
            [PhoneCandidate(**p) for p in row["candidate_phones"]], row["confidence"],
            normalized_ipa=row["candidate_phones"][0]["symbol"],
            metadata={"source_mode": "fixture" if self.fixture else "real", "backend": "powsm_ctc",
                "audio_binding": binding, "raw_label_trace": row["raw_label_trace"],
                "boundary_uncertainty": row["boundary_uncertainty"], "qualified_recognizer": False})
            for row in payload["segments"]]
