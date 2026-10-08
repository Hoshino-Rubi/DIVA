"""Replaceable acoustic feature extraction for CTC training caches."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar, Mapping, Protocol, Sequence


class AcousticFeatureExtractor(Protocol):
    def extract(self, examples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]: ...
    def metadata(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class LibrosaLogMelFeatureExtractor:
    sample_rate: int = 16_000
    n_fft: int = 400
    hop_length: int = 320
    n_mels: int = 80
    fmin: float = 20.0
    fmax: float = 7_600.0

    def _configuration(self) -> dict[str, Any]:
        return {
            "adapter": "librosa_log_mel_v1",
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "fmin": self.fmin,
            "fmax": self.fmax,
        }

    def metadata(self) -> Mapping[str, Any]:
        configuration = self._configuration()
        identity = hashlib.sha256(
            json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {**configuration, "model_sha256": identity, "pretrained_model": False}

    def extract(self, examples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        import librosa
        import numpy as np
        import torch

        result: dict[str, Any] = {}
        for row in examples:
            example_id = str(row.get("example_id") or "").strip()
            audio_path = Path(str(row.get("audio_path") or ""))
            start = float(row.get("crop_start_sec") or 0.0)
            end = float(row.get("crop_end_sec") or 0.0)
            if not example_id or not audio_path.is_file() or end <= start:
                raise ValueError(f"invalid feature example: {example_id or '<missing>'}")
            values, _ = librosa.load(
                audio_path,
                sr=self.sample_rate,
                mono=True,
                offset=start,
                duration=end - start,
            )
            if values.size == 0:
                raise ValueError(f"feature crop is empty: {example_id}")
            power = librosa.feature.melspectrogram(
                y=values,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
                power=2.0,
            )
            feature = np.log10(np.maximum(power, 1e-10)).T.astype("float32", copy=False)
            result[example_id] = torch.from_numpy(feature.copy())
        return result


@dataclass(slots=True)
class OnnxEspeakLogitFeatureExtractor:
    device: str = "cuda"
    cpu_thread_limit: int | None = 4
    _recognizer: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from any2xsampa.adapters.onnx_espeak import OnnxEspeakCTCUniversalPhoneRecognizer

        self._recognizer = OnnxEspeakCTCUniversalPhoneRecognizer(
            device=self.device,
            cpu_thread_limit=self.cpu_thread_limit,
        )

    def metadata(self) -> Mapping[str, Any]:
        self._recognizer._load_runtime()
        model = self._recognizer.model_metadata()
        return {
            "adapter": "onnx_espeak_raw_ctc_logits_v1",
            "device": self.device,
            "cpu_thread_limit": self.cpu_thread_limit,
            "model_sha256": model["model_sha256"],
            "vocab_sha256": model["vocab_sha256"],
            "providers": model["providers"],
            "pretrained_model": True,
        }

    def extract(self, examples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        import torch

        arrays: dict[Path, Any] = {}
        result: dict[str, Any] = {}
        np, session = self._recognizer._load_runtime()
        inputs = list(session.get_inputs())
        if not inputs:
            raise ValueError("ONNX/eSpeak model declares no inputs")
        for row in examples:
            example_id = str(row.get("example_id") or "").strip()
            audio_path = Path(str(row.get("audio_path") or ""))
            start = float(row.get("crop_start_sec") or 0.0)
            end = float(row.get("crop_end_sec") or 0.0)
            if not example_id or not audio_path.is_file() or end <= start:
                raise ValueError(f"invalid feature example: {example_id or '<missing>'}")
            if audio_path not in arrays:
                sample_rate, values = self._recognizer._read_pcm16_mono(audio_path)
                if sample_rate != 16_000:
                    raise ValueError("ONNX/eSpeak feature audio must resolve to 16 kHz")
                arrays[audio_path] = values
            values = arrays[audio_path]
            left = max(0, int(round(start * 16_000)))
            right = min(len(values), int(round(end * 16_000)))
            crop = values[left:right]
            if crop.size == 0:
                raise ValueError(f"feature crop is empty: {example_id}")
            feed: dict[str, Any] = {inputs[0].name: crop.reshape(1, -1)}
            for model_input in inputs[1:]:
                if model_input.name != "attention_mask":
                    raise ValueError(f"unsupported ONNX feature input: {model_input.name}")
                feed[model_input.name] = np.ones((1, crop.size), dtype=np.int64)
            outputs = session.run(None, feed)
            logits = np.asarray(outputs[0], dtype=np.float32)
            if logits.ndim != 3 or logits.shape[0] != 1 or not bool(np.isfinite(logits).all()):
                raise ValueError(f"invalid ONNX logits for feature example: {example_id}")
            result[example_id] = torch.from_numpy(logits[0].copy())
        return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_digest(value: object, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return digest


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object: {path}")
    return payload


def _verified_speech_snapshot(
    manifest_path: Path,
    *,
    expected_family: str,
    expected_model_type: str,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(f"pretrained speech snapshot manifest not found: {manifest_file}")
    manifest = _json_object(manifest_file, label="pretrained speech snapshot manifest")
    if manifest.get("schema_version") != "pretrained_acoustic_backbone_manifest_v1":
        raise ValueError("unsupported pretrained acoustic backbone manifest schema")
    if manifest.get("family") != expected_family:
        raise ValueError(f"feature extraction requires a {expected_family} manifest family")
    repository = str(manifest.get("repository") or "").strip()
    revision = str(manifest.get("revision") or "").strip().lower()
    if not repository or len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("pretrained speech snapshot requires a repository and immutable commit revision")

    retrieval = manifest.get("retrieval")
    if not isinstance(retrieval, Mapping) or (
        retrieval.get("direct_network_fallback") is not False
        or retrieval.get("local_files_only_runtime") is not True
    ):
        raise ValueError("pretrained speech snapshot must require local-only runtime with no direct fallback")
    license_metadata = manifest.get("license")
    if not isinstance(license_metadata, Mapping):
        raise ValueError("pretrained speech snapshot requires license evidence")
    if license_metadata.get("non_commercial_restriction") is not False:
        raise ValueError("pretrained speech snapshot license must explicitly have no NonCommercial restriction")
    license_id = str(license_metadata.get("identifier") or "").strip()
    if not license_id:
        raise ValueError("pretrained speech snapshot license identifier is missing")

    root = manifest_file.parent.resolve()
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("pretrained speech snapshot manifest requires files")
    files_by_role: dict[str, Path] = {}
    for raw in raw_files:
        if not isinstance(raw, Mapping):
            raise ValueError("pretrained speech snapshot file entries must be objects")
        relative = Path(str(raw.get("path") or ""))
        role = str(raw.get("role") or "").strip()
        if not role or relative.is_absolute() or not relative.parts:
            raise ValueError("pretrained speech snapshot file entry has an invalid path or role")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"pretrained speech snapshot file escapes its root: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"pretrained speech snapshot file is missing: {path}")
        try:
            expected_bytes = int(raw.get("bytes"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"pretrained speech snapshot file size is invalid: {relative}") from exc
        if expected_bytes < 0 or path.stat().st_size != expected_bytes:
            raise ValueError(f"pretrained speech snapshot file size mismatch: {relative}")
        expected_hash = _sha256_digest(raw.get("sha256"), label=f"{relative} sha256")
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"pretrained speech snapshot SHA-256 mismatch: {relative}")
        if role in files_by_role:
            raise ValueError(f"pretrained speech snapshot role is duplicated: {role}")
        files_by_role[role] = path

    required_roles = {
        "model_card",
        "upstream_model_metadata",
        "model_config",
        "preprocessor_config",
        "model_weights",
        "license",
    }
    missing_roles = sorted(required_roles - set(files_by_role))
    if missing_roles:
        raise ValueError("pretrained speech snapshot roles are missing: " + ", ".join(missing_roles))
    if Path(str(license_metadata.get("evidence_path") or "")) != files_by_role["model_card"].relative_to(root):
        raise ValueError("pretrained speech model-card license evidence path is inconsistent")
    if Path(str(license_metadata.get("license_path") or "")) != files_by_role["license"].relative_to(root):
        raise ValueError("pretrained speech license path is inconsistent")

    upstream = _json_object(files_by_role["upstream_model_metadata"], label="upstream model metadata")
    if (
        upstream.get("id") != repository
        or str(upstream.get("sha") or "").lower() != revision
        or upstream.get("private") is not False
        or upstream.get("gated") is not False
    ):
        raise ValueError("pretrained speech upstream metadata does not bind the declared public revision")
    config = _json_object(files_by_role["model_config"], label="pretrained speech model config")
    feature_contract = manifest.get("feature_contract")
    if not isinstance(feature_contract, Mapping):
        raise ValueError("pretrained speech feature contract is missing")
    if (
        config.get("model_type") != expected_model_type
        or int(config.get("hidden_size") or 0) != int(feature_contract.get("hidden_size") or 0)
    ):
        raise ValueError("pretrained speech model config disagrees with the feature contract")
    preprocessor = _json_object(files_by_role["preprocessor_config"], label="pretrained speech preprocessor config")
    if int(preprocessor.get("sampling_rate") or 0) != int(feature_contract.get("sample_rate") or 0):
        raise ValueError("pretrained speech preprocessor sample rate disagrees with the feature contract")
    model_card = files_by_role["model_card"].read_text(encoding="utf-8-sig")
    license_url = str(license_metadata.get("source_url") or "").strip()
    evidence_kind = str(license_metadata.get("evidence_kind") or "").strip()
    if evidence_kind == "model_card_explicit_link":
        if "official license" not in model_card.casefold() or not license_url or license_url not in model_card:
            raise ValueError("model card does not contain the declared official-license evidence")
    elif evidence_kind == "structured_upstream_metadata":
        card_data = upstream.get("cardData")
        upstream_license = card_data.get("license") if isinstance(card_data, Mapping) else None
        if str(upstream_license or "").strip().casefold() != str(
            license_metadata.get("upstream_identifier") or ""
        ).strip().casefold():
            raise ValueError("upstream model metadata does not contain the declared license identifier")
    else:
        raise ValueError("pretrained speech snapshot has an unsupported license evidence kind")
    license_text = files_by_role["license"].read_text(encoding="utf-8-sig")
    license_marker = str(license_metadata.get("text_marker") or "").strip()
    if not license_marker or license_marker not in license_text:
        raise ValueError("pretrained speech license text does not match its declared marker")
    return {
        "manifest": manifest,
        "manifest_path": manifest_file,
        "manifest_sha256": _sha256_file(manifest_file),
        "root": root,
        "repository": repository,
        "revision": revision,
        "license_id": license_id,
        "family": expected_family,
        "weights_path": files_by_role["model_weights"],
        "weights_sha256": _sha256_file(files_by_role["model_weights"]),
        "sample_rate": int(feature_contract["sample_rate"]),
        "hidden_size": int(feature_contract["hidden_size"]),
        "num_hidden_layers": int(config.get("num_hidden_layers") or 0),
    }


@dataclass(slots=True)
class FrozenWavLMFeatureExtractor:
    _family: ClassVar[str] = "wavlm"
    _model_type: ClassVar[str] = "wavlm"
    _adapter_name: ClassVar[str] = "frozen_wavlm_hidden_state_v1"
    manifest_path: Path | str
    device: str = "cuda"
    hidden_layer: int = -1
    cpu_thread_limit: int = 4
    _snapshot: dict[str, Any] = field(init=False, repr=False)
    _runtime: tuple[Any, Any, Any, Any] | None = field(init=False, default=None, repr=False)
    _runtime_metadata: dict[str, Any] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.manifest_path = Path(self.manifest_path)
        if isinstance(self.hidden_layer, bool) or not isinstance(self.hidden_layer, int):
            raise ValueError("WavLM hidden_layer must be an integer")
        if isinstance(self.cpu_thread_limit, bool) or self.cpu_thread_limit <= 0:
            raise ValueError("WavLM cpu_thread_limit must be a positive integer")
        normalized_device = str(self.device or "").strip().lower()
        if normalized_device != "cuda" and not normalized_device.startswith("cuda:"):
            raise ValueError("WavLM feature extraction requires an explicit CUDA device")
        self.device = normalized_device
        self._snapshot = _verified_speech_snapshot(
            self.manifest_path,
            expected_family=self._family,
            expected_model_type=self._model_type,
        )
        state_count = self._snapshot["num_hidden_layers"] + 1
        if not -state_count <= self.hidden_layer < state_count:
            raise ValueError(f"WavLM hidden_layer is outside the {state_count} available states")

    def _load_runtime(self) -> tuple[Any, Any, Any, Any]:
        if self._runtime is not None:
            return self._runtime
        try:
            import torch
            import transformers
            from transformers import AutoFeatureExtractor, Wav2Vec2Model, WavLMModel
        except ImportError as exc:
            raise RuntimeError("WavLM feature extraction requires torch and transformers") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("WavLM CUDA feature extraction was requested but CUDA is unavailable")
        if not str(torch.__version__).startswith("2.13.0+cu126") or str(torch.version.cuda) != "12.6":
            raise RuntimeError(
                "WavLM CUDA extraction requires torch 2.13.0+cu126 / CUDA 12.6; "
                f"found torch {torch.__version__}, CUDA {torch.version.cuda}"
            )
        device = torch.device(self.device)
        torch.empty(1, device=device)
        torch.set_num_threads(self.cpu_thread_limit)
        kwargs = {"local_files_only": True, "trust_remote_code": False}
        processor = AutoFeatureExtractor.from_pretrained(str(self._snapshot["root"]), **kwargs)
        model_class = WavLMModel if self._family == "wavlm" else Wav2Vec2Model
        model = model_class.from_pretrained(
            str(self._snapshot["root"]),
            use_safetensors=False,
            **kwargs,
        )
        if int(model.config.hidden_size) != self._snapshot["hidden_size"]:
            raise ValueError("loaded WavLM hidden size disagrees with the verified snapshot")
        model.requires_grad_(False)
        model.eval().to(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        self._runtime_metadata = {
            "runtime_verified": True,
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "transformers_version": str(transformers.__version__),
            "cuda_device_name": torch.cuda.get_device_name(device),
        }
        self._runtime = torch, processor, model, device
        return self._runtime

    def metadata(self) -> Mapping[str, Any]:
        return {
            "adapter": self._adapter_name,
            "family": self._family,
            "repository": self._snapshot["repository"],
            "revision": self._snapshot["revision"],
            "manifest_path": str(self._snapshot["manifest_path"]),
            "manifest_sha256": self._snapshot["manifest_sha256"],
            "model_sha256": self._snapshot["weights_sha256"],
            "license": self._snapshot["license_id"],
            "sample_rate": self._snapshot["sample_rate"],
            "hidden_size": self._snapshot["hidden_size"],
            "hidden_layer": self.hidden_layer,
            "device": self.device,
            "cpu_thread_limit": self.cpu_thread_limit,
            "pretrained_model": True,
            "backbone_frozen": True,
            "local_files_only": True,
            "lyrics_used_as_acoustic_input": False,
            "vsqx_used_as_acoustic_input": False,
            **self._runtime_metadata,
        }

    def extract(self, examples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        import librosa

        torch, processor, model, device = self._load_runtime()
        arrays: dict[Path, Any] = {}
        result: dict[str, Any] = {}
        resolved_layer: int | None = None
        total_frames = 0
        max_crop_samples = 0
        for row in examples:
            example_id = str(row.get("example_id") or "").strip()
            audio_path = Path(str(row.get("audio_path") or ""))
            start = float(row.get("crop_start_sec") or 0.0)
            end = float(row.get("crop_end_sec") or 0.0)
            if not example_id or not audio_path.is_file() or end <= start:
                raise ValueError(f"invalid feature example: {example_id or '<missing>'}")
            if example_id in result:
                raise ValueError(f"duplicate feature example: {example_id}")
            if audio_path not in arrays:
                values, _ = librosa.load(
                    audio_path,
                    sr=self._snapshot["sample_rate"],
                    mono=True,
                )
                if values.size == 0:
                    raise ValueError(f"feature audio is empty: {audio_path}")
                arrays[audio_path] = values
            values = arrays[audio_path]
            left = max(0, int(round(start * self._snapshot["sample_rate"])))
            right = min(len(values), int(round(end * self._snapshot["sample_rate"])))
            crop = values[left:right]
            if crop.size == 0:
                raise ValueError(f"feature crop is empty: {example_id}")
            max_crop_samples = max(max_crop_samples, int(crop.size))
            inputs = processor(
                crop,
                sampling_rate=self._snapshot["sample_rate"],
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_values = inputs["input_values"].to(device)
            attention_mask = inputs.get("attention_mask")
            kwargs: dict[str, Any] = {
                "input_values": input_values,
                "output_hidden_states": True,
            }
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask.to(device)
            with torch.inference_mode():
                output = model(**kwargs)
                hidden_states = output.hidden_states
                if hidden_states is None:
                    raise RuntimeError("WavLM did not return hidden states")
                current_layer = self.hidden_layer % len(hidden_states)
                if resolved_layer is None:
                    resolved_layer = current_layer
                elif resolved_layer != current_layer:
                    raise RuntimeError("WavLM hidden-state count changed between examples")
                feature = hidden_states[self.hidden_layer][0].detach().float().cpu().contiguous()
            if feature.ndim != 2 or int(feature.shape[1]) != self._snapshot["hidden_size"]:
                raise ValueError(f"invalid WavLM feature shape: {example_id}")
            result[example_id] = feature
            total_frames += int(feature.shape[0])
            del output, hidden_states, input_values, kwargs
        torch.cuda.synchronize(device)
        self._runtime_metadata.update(
            {
                "resolved_hidden_layer": resolved_layer,
                "example_count": len(result),
                "total_feature_frames": total_frames,
                "max_crop_samples": max_crop_samples,
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
        )
        return result


class FrozenXLSRFeatureExtractor(FrozenWavLMFeatureExtractor):
    _family = "wav2vec2_xls_r"
    _model_type = "wav2vec2"
    _adapter_name = "frozen_xls_r_hidden_state_v1"


__all__ = [
    "AcousticFeatureExtractor",
    "FrozenWavLMFeatureExtractor",
    "FrozenXLSRFeatureExtractor",
    "LibrosaLogMelFeatureExtractor",
    "OnnxEspeakLogitFeatureExtractor",
]
