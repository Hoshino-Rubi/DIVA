from __future__ import annotations

"""Isolated ONNX/eSpeak CTC recognizer adapter.

This is a functional reconstruction. It keeps raw acoustic labels in IPA-like
form until an explicit mapping projects them into the project's phone inventory.
X-SAMPA conversion remains downstream of this adapter.
"""

from collections.abc import Callable, Mapping, Sequence
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
from typing import Any
import wave

from any2xsampa.core.interfaces import UniversalPhoneRecognizer
from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange


class OnnxEspeakResourceError(RuntimeError):
    """A required local model/runtime resource is unavailable or invalid."""


class OnnxEspeakInferenceError(RuntimeError):
    """The configured model could not produce valid phone evidence."""


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MODEL_BUNDLE = _PROJECT_ROOT / "models" / "onnx-espeak" / "model"
_DEFAULT_MODEL_FILE = _DEFAULT_MODEL_BUNDLE / "onnx" / "model_int8.onnx"
_DEFAULT_VOCAB_FILE = _PROJECT_ROOT / "models" / "onnx-espeak" / "tokenizer" / "vocab.json"
_MODEL_MANIFEST = _PROJECT_ROOT / "models" / "MANIFEST.json"
_MODEL_MANIFEST_IDS = {
    "model": "wav2vec2-lv-60-espeak-cv-ft-onnx-int8",
    "vocab": "wav2vec2-lv-60-espeak-cv-ft-tokenizer",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise OnnxEspeakResourceError(f"ONNX/eSpeak {label} file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OnnxEspeakResourceError(f"ONNX/eSpeak {label} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise OnnxEspeakResourceError(f"ONNX/eSpeak {label} must contain a JSON object: {path}")
    return payload


def _resolve_model_file(model_path: Path | None) -> Path:
    if model_path is None:
        path = _DEFAULT_MODEL_BUNDLE.resolve(strict=False)
    else:
        path = model_path.resolve(strict=False)
    if path.is_file():
        if path.suffix.lower() != ".onnx":
            raise OnnxEspeakResourceError(f"ONNX/eSpeak model must be an .onnx file: {path}")
        return path
    if not path.is_dir():
        raise OnnxEspeakResourceError(f"ONNX/eSpeak model resource not found: {path}")
    candidates = (
        path / "onnx" / "model_int8.onnx",
        path / "model_int8.onnx",
        path / "onnx" / "model.onnx",
        path / "model.onnx",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise OnnxEspeakResourceError(
        f"ONNX/eSpeak bundle has no supported model file under {path}; "
        "expected onnx/model_int8.onnx, model_int8.onnx, onnx/model.onnx, or model.onnx."
    )


def _resolve_vocab_file(model_file: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        path = explicit_path.resolve(strict=False)
        if path.is_file():
            return path
        raise OnnxEspeakResourceError(f"ONNX/eSpeak vocabulary file not found: {path}")
    candidates = (
        model_file.parent / "vocab.json",
        model_file.parent.parent / "vocab.json",
    )
    if model_file.resolve() == _DEFAULT_MODEL_FILE.resolve():
        candidates = (*candidates, _DEFAULT_VOCAB_FILE)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise OnnxEspeakResourceError(
        "ONNX/eSpeak vocabulary file not found beside the model. "
        "Provide vocab_path explicitly or place vocab.json in the model bundle root."
    )


def _manifest_hashes(*, manifest_path: Path = _MODEL_MANIFEST) -> dict[str, str]:
    """Return recorded hashes for the bundled model resources."""

    payload = _read_json_object(manifest_path, label="model manifest")
    models = payload.get("models")
    if not isinstance(models, list):
        raise OnnxEspeakResourceError("ONNX/eSpeak model manifest models must be a list.")
    hashes: dict[str, str] = {}
    for item in models:
        if not isinstance(item, Mapping):
            continue
        model_id = str(item.get("id", ""))
        for resource_kind, resource_id in _MODEL_MANIFEST_IDS.items():
            if model_id != resource_id:
                continue
            raw_hash = item.get("primary_sha256")
            if not isinstance(raw_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", raw_hash):
                raise OnnxEspeakResourceError(
                    f"ONNX/eSpeak manifest hash for {resource_id!r} is invalid."
                )
            hashes[resource_kind] = raw_hash.lower()
    missing = sorted(set(_MODEL_MANIFEST_IDS) - set(hashes))
    if missing:
        raise OnnxEspeakResourceError(
            "ONNX/eSpeak model manifest is missing resource hash(es): " + ", ".join(missing)
        )
    return hashes


def _load_vocabulary(path: Path) -> tuple[str, ...]:
    payload = _read_json_object(path, label="vocabulary")
    by_id: dict[int, str] = {}
    for raw_label, raw_id in payload.items():
        if isinstance(raw_id, bool) or not isinstance(raw_id, int) or raw_id < 0:
            raise OnnxEspeakResourceError(f"ONNX/eSpeak vocabulary id for {raw_label!r} is invalid.")
        if raw_id in by_id:
            raise OnnxEspeakResourceError(f"ONNX/eSpeak vocabulary contains duplicate id {raw_id}.")
        by_id[raw_id] = str(raw_label)
    if not by_id or set(by_id) != set(range(max(by_id) + 1)):
        raise OnnxEspeakResourceError("ONNX/eSpeak vocabulary ids must be contiguous from zero.")
    return tuple(by_id[index] for index in range(len(by_id)))


class OnnxEspeakCTCUniversalPhoneRecognizer(UniversalPhoneRecognizer):
    """Run a local wav2vec2 eSpeak-phone CTC model through ONNX Runtime."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        vocab_path: Path | str | None = None,
        phone_mapping_path: Path | str | None = None,
        device: str = "auto",
        expected_model_sha256: str | None = None,
        expected_vocab_sha256: str | None = None,
        preserve_unmapped_ipa: bool = True,
        top_k: int = 5,
        session_factory: Callable[[str, list[str]], Any] | None = None,
        enable_profiling: bool = False,
        profile_output_dir: Path | str | None = None,
        cpu_thread_limit: int | None = None,
        chunk_duration_sec: float | None = None,
        chunk_overlap_sec: float = 0.5,
        **_: object,
    ) -> None:
        config_root = Path(__file__).resolve().parent.parent / "config"
        self.model_path = Path(model_path) if model_path is not None else None
        self.vocab_path = Path(vocab_path) if vocab_path is not None else None
        self.phone_mapping_path = Path(phone_mapping_path) if phone_mapping_path else (
            config_root / "recognizer_onnx_espeak_ipa_map.json"
        )
        self.device = str(device or "auto").strip().lower()
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("ONNX/eSpeak device must be auto, cpu, or cuda.")
        if top_k < 1:
            raise ValueError("ONNX/eSpeak top_k must be at least one.")
        self.expected_model_sha256 = (
            str(expected_model_sha256).strip().lower() if expected_model_sha256 else None
        )
        self.expected_vocab_sha256 = (
            str(expected_vocab_sha256).strip().lower() if expected_vocab_sha256 else None
        )
        self.preserve_unmapped_ipa = bool(preserve_unmapped_ipa)
        self.top_k = int(top_k)
        self._session_factory = session_factory
        self.enable_profiling = bool(enable_profiling)
        if cpu_thread_limit is not None and int(cpu_thread_limit) < 1:
            raise ValueError("ONNX/eSpeak cpu_thread_limit must be at least one.")
        self.cpu_thread_limit = int(cpu_thread_limit) if cpu_thread_limit is not None else None
        if chunk_duration_sec is not None and float(chunk_duration_sec) <= 0.0:
            raise ValueError("ONNX/eSpeak chunk_duration_sec must be positive.")
        if float(chunk_overlap_sec) < 0.0:
            raise ValueError("ONNX/eSpeak chunk_overlap_sec cannot be negative.")
        if chunk_duration_sec is not None and float(chunk_overlap_sec) >= float(chunk_duration_sec):
            raise ValueError("ONNX/eSpeak chunk_overlap_sec must be shorter than chunk_duration_sec.")
        self.chunk_duration_sec = float(chunk_duration_sec) if chunk_duration_sec is not None else None
        self.chunk_overlap_sec = float(chunk_overlap_sec)
        self.profile_output_dir = (
            Path(profile_output_dir)
            if profile_output_dir is not None
            else _PROJECT_ROOT / ".state" / "onnx-profiles"
        )
        self._session: Any | None = None
        self._numpy: Any | None = None
        self._model_file: Path | None = None
        self._vocab_file: Path | None = None
        self._model_sha256: str | None = None
        self._vocab_sha256: str | None = None
        self._labels: tuple[str, ...] = ()
        self._providers: tuple[str, ...] = ()
        self._available_providers: tuple[str, ...] = ()
        self._requested_providers: tuple[str, ...] = ()
        self._provider_fallback_reason: str | None = None
        self._dll_preload: dict[str, object] = {
            "attempted": False,
            "success": False,
            "method": None,
            "error": None,
        }
        self._manifest_verified = False
        self._manifest_path: Path | None = None
        self._profile_path: str | None = None
        self._profile_finalized = False
        self._load_mapping()

    def _load_mapping(self) -> None:
        payload = _read_json_object(self.phone_mapping_path, label="phone mapping")
        self._mapping_profile = str(payload.get("profile", "unknown"))
        self._mapping_schema = str(payload.get("schema_version", "unknown"))
        blank_labels = payload.get("blank_labels", [])
        if not isinstance(blank_labels, list):
            raise OnnxEspeakResourceError("ONNX/eSpeak phone mapping blank_labels must be a list.")
        self._blank_labels = {str(value) for value in blank_labels}
        normalization = payload.get("normalization", {})
        if not isinstance(normalization, dict):
            raise OnnxEspeakResourceError("ONNX/eSpeak phone mapping normalization must be an object.")
        self._strip_trailing_tone_digits = bool(
            normalization.get("strip_trailing_tone_digits", False)
        )
        raw_aliases = normalization.get("aliases", {})
        if not isinstance(raw_aliases, dict):
            raise OnnxEspeakResourceError("ONNX/eSpeak phone mapping normalization aliases must be an object.")
        self._raw_label_aliases = {
            str(source): str(target) for source, target in raw_aliases.items()
        }
        mappings = payload.get("mappings")
        if not isinstance(mappings, dict):
            raise OnnxEspeakResourceError("ONNX/eSpeak phone mapping mappings must be an object.")
        self._raw_to_internal: dict[str, str] = {}
        for internal, raw_labels in mappings.items():
            if not isinstance(raw_labels, list):
                raise OnnxEspeakResourceError(f"Phone mapping for {internal!r} must be a list.")
            for raw_label in raw_labels:
                raw = str(raw_label)
                previous = self._raw_to_internal.get(raw)
                if previous is not None and previous != str(internal):
                    raise OnnxEspeakResourceError(
                        f"Raw label {raw!r} is mapped to both {previous!r} and {internal!r}."
                    )
                self._raw_to_internal[raw] = str(internal)

    def _map_raw_label(self, label: str) -> str | None:
        text = str(label)
        if text in self._blank_labels:
            return None
        mapped = self._raw_to_internal.get(text)
        if mapped is not None:
            return mapped
        normalized = re.sub(r"[1-6]$", "", text) if self._strip_trailing_tone_digits else text
        normalized = self._raw_label_aliases.get(normalized, normalized)
        return self._raw_to_internal.get(normalized)

    def _project_raw_label(self, label: str) -> str | None:
        mapped = self._map_raw_label(label)
        if mapped is not None:
            return mapped
        if label in self._blank_labels:
            return None
        return label if self.preserve_unmapped_ipa else None

    def _load_runtime(self) -> tuple[Any, Any]:
        if self._session is not None and self._numpy is not None:
            return self._numpy, self._session
        model_file = _resolve_model_file(self.model_path)
        vocab_file = _resolve_vocab_file(model_file, self.vocab_path)
        labels = _load_vocabulary(vocab_file)
        model_hash = _sha256(model_file)
        vocab_hash = _sha256(vocab_file)
        bundled_model = model_file.resolve() == _DEFAULT_MODEL_FILE.resolve()
        bundled_vocab = vocab_file.resolve() == _DEFAULT_VOCAB_FILE.resolve()
        manifest_hashes: dict[str, str] = {}
        if bundled_model or bundled_vocab:
            manifest_hashes = _manifest_hashes()
        manifest_model_hash = manifest_hashes.get("model") if bundled_model else None
        manifest_vocab_hash = manifest_hashes.get("vocab") if bundled_vocab else None
        if manifest_model_hash and model_hash != manifest_model_hash:
            raise OnnxEspeakResourceError(
                "ONNX/eSpeak model SHA-256 mismatch: "
                f"expected {manifest_model_hash}, got {model_hash}."
            )
        if manifest_vocab_hash and vocab_hash != manifest_vocab_hash:
            raise OnnxEspeakResourceError(
                "ONNX/eSpeak vocabulary SHA-256 mismatch: "
                f"expected {manifest_vocab_hash}, got {vocab_hash}."
            )
        if self.expected_model_sha256 and model_hash != self.expected_model_sha256:
            raise OnnxEspeakResourceError(
                "ONNX/eSpeak model SHA-256 mismatch: "
                f"expected {self.expected_model_sha256}, got {model_hash}."
            )
        if self.expected_vocab_sha256 and vocab_hash != self.expected_vocab_sha256:
            raise OnnxEspeakResourceError(
                "ONNX/eSpeak vocabulary SHA-256 mismatch: "
                f"expected {self.expected_vocab_sha256}, got {vocab_hash}."
            )
        try:
            np = importlib.import_module("numpy")
            ort = importlib.import_module("onnxruntime")
        except ImportError as exc:
            raise OnnxEspeakResourceError(
                "ONNX/eSpeak real mode requires the optional numpy and onnxruntime packages."
            ) from exc
        self._preload_cuda_dlls(ort)
        available = tuple(str(value) for value in ort.get_available_providers())
        self._available_providers = available
        providers = self._select_providers(available)
        try:
            if self._session_factory is not None:
                session = self._session_factory(str(model_file), list(providers))
            else:
                session = self._create_session(ort, str(model_file), providers)
        except Exception as exc:
            if self.device == "auto" and providers != ("CPUExecutionProvider",):
                self._provider_fallback_reason = f"CUDA session initialization failed; fell back to CPU: {exc}"
                try:
                    session = (
                        self._session_factory(str(model_file), ["CPUExecutionProvider"])
                        if self._session_factory is not None
                        else self._create_session(ort, str(model_file), ("CPUExecutionProvider",))
                    )
                    self._requested_providers = ("CPUExecutionProvider",)
                except Exception as cpu_exc:
                    raise OnnxEspeakResourceError(
                        f"ONNX/eSpeak model could not be loaded by ONNX Runtime on CUDA or CPU: {model_file}: {cpu_exc}"
                    ) from cpu_exc
            else:
                raise OnnxEspeakResourceError(
                    f"ONNX/eSpeak model could not be loaded by ONNX Runtime: {model_file}: {exc}"
                ) from exc
        self._numpy = np
        try:
            active_providers = tuple(str(value) for value in session.get_providers())
        except Exception as exc:
            raise OnnxEspeakResourceError(
                "ONNX/eSpeak session did not expose its active providers."
            ) from exc
        active_cuda = "CUDAExecutionProvider" in active_providers
        if self.device == "cuda" and not active_cuda:
            raise OnnxEspeakResourceError(
                "ONNX/eSpeak CUDA was requested but the created session did not activate "
                "CUDAExecutionProvider."
            )
        if self.device == "auto" and "CUDAExecutionProvider" in self._requested_providers:
            if not active_cuda:
                self._provider_fallback_reason = (
                    "CUDAExecutionProvider was available/requested but the session activated "
                    "CPUExecutionProvider instead."
                )
        self._session = session
        self._model_file = model_file
        self._vocab_file = vocab_file
        self._model_sha256 = model_hash
        self._vocab_sha256 = vocab_hash
        self._labels = labels
        self._providers = active_providers
        self._manifest_verified = bool(manifest_hashes) and (
            (not bundled_model or manifest_model_hash == model_hash)
            and (not bundled_vocab or manifest_vocab_hash == vocab_hash)
        )
        self._manifest_path = _MODEL_MANIFEST if manifest_hashes else None
        return np, session

    def _preload_cuda_dlls(self, ort: Any) -> None:
        """Best-effort CUDA DLL preload before ORT provider discovery."""

        if self.device == "cpu":
            return
        self._dll_preload["attempted"] = True
        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            try:
                preload()
                self._dll_preload.update({"success": True, "method": "onnxruntime.preload_dlls"})
                return
            except Exception as exc:
                self._dll_preload["error"] = str(exc)
        try:
            importlib.import_module("torch")
            self._dll_preload.update({"success": True, "method": "torch_import"})
        except Exception as exc:
            self._dll_preload["error"] = str(exc)

    def _create_session(self, ort: Any, model_path: str, providers: Sequence[str]) -> Any:
        session_options_factory = getattr(ort, "SessionOptions", None)
        needs_options = self.enable_profiling or self.cpu_thread_limit is not None
        if not needs_options or not callable(session_options_factory):
            return ort.InferenceSession(model_path, providers=list(providers))
        options = session_options_factory()
        if self.cpu_thread_limit is not None:
            options.intra_op_num_threads = self.cpu_thread_limit
            options.inter_op_num_threads = 1
        options.enable_profiling = self.enable_profiling
        if self.enable_profiling:
            self.profile_output_dir.mkdir(parents=True, exist_ok=True)
            options.profile_file_prefix = str(
                (self.profile_output_dir / "onnxruntime_profile").resolve(strict=False)
            )
            self._profile_finalized = False
        return ort.InferenceSession(model_path, sess_options=options, providers=list(providers))

    def _select_providers(self, available: Sequence[str]) -> tuple[str, ...]:
        has_cuda = "CUDAExecutionProvider" in available
        has_cpu = "CPUExecutionProvider" in available
        self._provider_fallback_reason = None
        if self.device == "cuda" and not has_cuda:
            raise OnnxEspeakResourceError(
                "ONNX/eSpeak CUDA was requested but CUDAExecutionProvider is unavailable."
            )
        if self.device == "cuda":
            selected = ("CUDAExecutionProvider", "CPUExecutionProvider") if has_cpu else (
                "CUDAExecutionProvider",
            )
            self._requested_providers = selected
            return selected
        if self.device == "auto" and has_cuda:
            selected = ("CUDAExecutionProvider", "CPUExecutionProvider") if has_cpu else (
                "CUDAExecutionProvider",
            )
            self._requested_providers = selected
            return selected
        if not has_cpu:
            raise OnnxEspeakResourceError("ONNX/eSpeak CPUExecutionProvider is unavailable.")
        self._provider_fallback_reason = (
            "CUDAExecutionProvider is unavailable; auto device selection fell back to CPU."
            if self.device == "auto"
            else None
        )
        selected = ("CPUExecutionProvider",)
        self._requested_providers = selected
        return selected

    def finalize_profile(self) -> str | None:
        """Finalize an enabled ORT profile and return its generated path."""

        if self._session is None or not self.enable_profiling or self._profile_finalized:
            return self._profile_path
        end_profiling = getattr(self._session, "end_profiling", None)
        if not callable(end_profiling):
            self._profile_finalized = True
            return None
        try:
            profile_path = end_profiling()
        except Exception as exc:
            raise OnnxEspeakInferenceError(f"ONNX/eSpeak profiling could not be finalized: {exc}") from exc
        self._profile_path = str(profile_path) if profile_path else None
        self._profile_finalized = True
        return self._profile_path

    def model_metadata(self) -> dict[str, object]:
        return {
            "adapter": self.__class__.__name__,
            "model_backend": "onnxruntime",
            "raw_model_output_kind": "trained_onnx_ctc_model",
            "phone_mapping_profile": self._mapping_profile,
            "phone_mapping_schema_version": self._mapping_schema,
            "preserve_unmapped_ipa": self.preserve_unmapped_ipa,
            "device": self.device,
            "providers": list(self._providers),
            "session_providers": list(self._providers),
            "available_providers": list(self._available_providers),
            "requested_providers": list(self._requested_providers),
            "cuda_active": "CUDAExecutionProvider" in self._providers,
            "provider_fallback_reason": self._provider_fallback_reason,
            "cuda_dll_preload": dict(self._dll_preload),
            "model_path": str(self._model_file or self.model_path or _DEFAULT_MODEL_BUNDLE),
            "model_sha256": self._model_sha256,
            "vocab_path": str(self._vocab_file or self.vocab_path or _DEFAULT_VOCAB_FILE),
            "vocab_sha256": self._vocab_sha256,
            "manifest_path": str(self._manifest_path) if self._manifest_path else None,
            "manifest_verified": self._manifest_verified,
            "profiling_requested": self.enable_profiling,
            "profile_output_dir": str(self.profile_output_dir),
            "profile_path": self._profile_path,
            "cpu_thread_limit": self.cpu_thread_limit,
            "chunk_duration_sec": self.chunk_duration_sec,
            "chunk_overlap_sec": self.chunk_overlap_sec,
        }

    def _read_pcm16_mono(self, audio_path: Path | str) -> tuple[int, Any]:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Vocal audio file not found: {path}")
        np, _ = self._load_runtime()
        try:
            with wave.open(str(path), "rb") as stream:
                channels = stream.getnchannels()
                sample_width = stream.getsampwidth()
                sample_rate = stream.getframerate()
                frames = stream.readframes(stream.getnframes())
        except (OSError, wave.Error) as exc:
            raise OnnxEspeakInferenceError(f"ONNX/eSpeak requires a readable PCM WAV file: {path}") from exc
        if channels < 1 or sample_rate < 1:
            raise OnnxEspeakInferenceError(f"ONNX/eSpeak WAV metadata is invalid: {path}")
        if sample_width == 1:
            samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sample_width == 2:
            samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        elif sample_width == 4:
            samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise OnnxEspeakInferenceError(
                f"ONNX/eSpeak supports 8-, 16-, or 32-bit integer PCM WAV, got {sample_width * 8}-bit."
            )
        if samples.size == 0 or samples.size % channels:
            raise OnnxEspeakInferenceError(f"ONNX/eSpeak WAV contains no complete audio frames: {path}")
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        samples = self._resample(samples, sample_rate, 16000, np)
        mean = float(samples.mean())
        variance = float(samples.var())
        if variance > 1e-14:
            samples = (samples - mean) / math.sqrt(variance + 1e-7)
        else:
            samples = samples - mean
        return 16000, np.asarray(samples, dtype=np.float32)

    @staticmethod
    def _resample(samples: Any, source_rate: int, target_rate: int, np: Any) -> Any:
        if source_rate == target_rate:
            return samples
        target_count = max(1, int(round(len(samples) * target_rate / source_rate)))
        source_positions = np.arange(len(samples), dtype=np.float64)
        target_positions = np.arange(target_count, dtype=np.float64) * source_rate / target_rate
        return np.interp(target_positions, source_positions, samples).astype(np.float32)

    def _predict_frames(
        self,
        samples: Any,
        sample_rate: int,
        frame_size: int | None = None,
        hop_size: int | None = None,
    ) -> list[dict[str, Any]]:
        del frame_size, hop_size
        if sample_rate != 16000:
            raise OnnxEspeakInferenceError("ONNX/eSpeak model input must be resampled to 16000 Hz.")
        np, session = self._load_runtime()
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if values.size == 0:
            raise OnnxEspeakInferenceError("ONNX/eSpeak received empty audio.")
        inputs = list(session.get_inputs())
        if not inputs:
            raise OnnxEspeakInferenceError("ONNX/eSpeak model declares no inputs.")
        feed: dict[str, Any] = {inputs[0].name: values.reshape(1, -1)}
        for model_input in inputs[1:]:
            if model_input.name == "attention_mask":
                feed[model_input.name] = np.ones((1, values.size), dtype=np.int64)
            else:
                raise OnnxEspeakInferenceError(
                    f"ONNX/eSpeak model has unsupported required input: {model_input.name}"
                )
        try:
            outputs = session.run(None, feed)
        except Exception as exc:
            raise OnnxEspeakInferenceError(f"ONNX/eSpeak inference failed: {exc}") from exc
        if not outputs:
            raise OnnxEspeakInferenceError("ONNX/eSpeak model returned no outputs.")
        logits = np.asarray(outputs[0])
        if logits.ndim != 3 or logits.shape[0] != 1:
            raise OnnxEspeakInferenceError(
                f"ONNX/eSpeak logits must have shape [1, frames, labels], got {tuple(logits.shape)}."
            )
        if logits.shape[2] != len(self._labels):
            raise OnnxEspeakInferenceError(
                f"ONNX/eSpeak logits expose {logits.shape[2]} labels but vocabulary has {len(self._labels)}."
            )
        if not bool(np.isfinite(logits).all()):
            raise OnnxEspeakInferenceError("ONNX/eSpeak logits contain non-finite values.")
        frame_count = int(logits.shape[1])
        predictions: list[dict[str, Any]] = []
        for index in range(frame_count):
            start = int(math.floor(index * values.size / frame_count))
            end = max(start + 1, int(math.ceil((index + 1) * values.size / frame_count)))
            predictions.append(self._frame_prediction(logits[0, index], start, min(end, values.size), np))
        return predictions

    def _predict_chunked_frames(self, samples: Any, sample_rate: int) -> list[dict[str, Any]]:
        """Run bounded waveform chunks and project retained frames to global time."""

        if self.chunk_duration_sec is None:
            return self._predict_frames(samples, sample_rate)
        chunk_size = max(1, int(round(self.chunk_duration_sec * sample_rate)))
        overlap = int(round(self.chunk_overlap_sec * sample_rate))
        values = samples
        if len(values) <= chunk_size:
            return self._predict_frames(values, sample_rate)
        step = max(1, chunk_size - overlap)
        predictions: list[dict[str, Any]] = []
        chunk_start = 0
        chunk_index = 0
        total = len(values)
        while chunk_start < total:
            chunk_end = min(total, chunk_start + chunk_size)
            local = self._predict_frames(values[chunk_start:chunk_end], sample_rate)
            local_length = chunk_end - chunk_start
            keep_start = 0 if chunk_start == 0 else overlap // 2
            keep_end = local_length if chunk_end >= total else max(keep_start + 1, local_length - overlap // 2)
            for prediction in local:
                local_start = int(prediction["start_index"])
                local_end = int(prediction["end_index"])
                center = (local_start + local_end) / 2.0
                if not (keep_start <= center < keep_end):
                    continue
                projected = dict(prediction)
                projected["start_index"] = local_start + chunk_start
                projected["end_index"] = local_end + chunk_start
                projected["_chunk_index"] = chunk_index
                predictions.append(projected)
            if chunk_end >= total:
                break
            chunk_start += step
            chunk_index += 1
        return sorted(predictions, key=lambda item: (int(item["start_index"]), int(item["end_index"])))

    def _frame_prediction(self, logits: Any, start: int, end: int, np: Any) -> dict[str, Any]:
        shifted = logits - float(logits.max())
        weights = np.exp(shifted)
        probabilities = weights / float(weights.sum())
        raw_order = np.argsort(probabilities)[::-1][: self.top_k]
        mapped_probabilities: dict[str, float] = {}
        raw_top_labels: list[dict[str, Any]] = []
        for raw_id, label in enumerate(self._labels):
            symbol = self._project_raw_label(label)
            if symbol is not None:
                mapped_probabilities[symbol] = mapped_probabilities.get(symbol, 0.0) + float(
                    probabilities[raw_id]
                )
        for raw_id in raw_order:
            label = self._labels[int(raw_id)]
            raw_top_labels.append(
                {
                    "label": label,
                    "posterior": float(probabilities[int(raw_id)]),
                    "logit": float(logits[int(raw_id)]),
                    "mapped_symbol": self._project_raw_label(label),
                }
            )
        mapped_order = sorted(mapped_probabilities.items(), key=lambda item: (-item[1], item[0]))
        candidates = [
            PhoneCandidate(symbol=symbol, confidence=confidence)
            for symbol, confidence in mapped_order[: self.top_k]
        ]
        mapped_frame_candidates = [
            {
                "symbol": symbol,
                "posterior": confidence,
                "logit": math.log(max(confidence, 1e-30)),
            }
            for symbol, confidence in mapped_order[: self.top_k]
        ]
        raw_top_id = int(raw_order[0])
        raw_top_label = self._labels[raw_top_id]
        return {
            "start_index": start,
            "end_index": end,
            "raw_top_label": raw_top_label,
            "symbol": self._project_raw_label(raw_top_label),
            "candidates": candidates,
            "raw_top_labels": raw_top_labels,
            "mapped_frame_candidates": mapped_frame_candidates,
        }

    def _build_real_segment(
        self,
        symbol: str,
        windows: list[dict[str, Any]],
        start_index: int,
        end_index: int,
        sample_rate: int,
        frame_size: int | None,
        hop_size: int | None,
        lyrics: str | None,
    ) -> PhoneSegment:
        del frame_size, hop_size
        scores: dict[str, list[float]] = {}
        for window in windows:
            for candidate in window.get("candidates", []):
                if isinstance(candidate, PhoneCandidate):
                    scores.setdefault(candidate.symbol, []).append(candidate.confidence)
        means = {key: sum(values) / len(windows) for key, values in scores.items()}
        ordered = sorted(means.items(), key=lambda item: (-item[1], item[0]))
        candidates = [PhoneCandidate(name, value) for name, value in ordered[: self.top_k]]
        if not candidates or candidates[0].symbol != symbol:
            symbol_score = means.get(symbol, 0.0)
            candidates = [PhoneCandidate(symbol, symbol_score)] + [
                candidate for candidate in candidates if candidate.symbol != symbol
            ]
        posterior_frames = [
            {
                "candidates": [
                    {"symbol": candidate.symbol, "confidence": candidate.confidence}
                    for candidate in window.get("candidates", [])
                    if isinstance(candidate, PhoneCandidate)
                ]
            }
            for window in windows
        ]
        logit_frames = [
            {
                "candidates": [
                    {
                        "symbol": item["symbol"],
                        "logit": item["logit"],
                    }
                    for item in window.get("mapped_frame_candidates", [])
                    if isinstance(item, Mapping) and item.get("symbol") is not None
                ]
            }
            for window in windows
        ]
        frame_hop = (end_index - start_index) / max(1, len(windows)) / sample_rate
        metadata: dict[str, Any] = {
            "source_mode": "real",
            "backend_kind": "onnx-espeak",
            "real_backend": True,
            "real_model_output": True,
            "model_output": True,
            "raw_model_output_kind": "trained_onnx_ctc_model",
            "raw_candidates": [candidate.symbol for candidate in candidates],
            "lyrics_guided": False,
            "lyrics_input_ignored": bool(lyrics),
            "candidate_aggregation": "mean_mapped_frame_posterior",
            "posterior_trace": {
                "source": "producer:onnx_mapped_frame_posteriors",
                "frame_hop_sec": frame_hop,
                "frames": posterior_frames,
            },
            "logit_trace": {
                "source": "producer:onnx_mapped_frame_logits",
                "frame_hop_sec": frame_hop,
                "frames": logit_frames,
            },
            "raw_label_trace": {
                "source": "onnxruntime:raw_ctc_logits",
                "frames": [
                    {"candidates": list(window.get("raw_top_labels", []))} for window in windows
                ],
            },
            "mapped_frame_trace": {
                "source": "producer:onnx_mapped_frame_posteriors",
                "frames": [
                    {"candidates": list(window.get("mapped_frame_candidates", []))}
                    for window in windows
                ],
            },
            "adapter_diagnostics": {
                "backend_kind": "onnx-espeak",
                "adapter_kind": "onnx_espeak_ctc_runtime",
                "candidate_aggregation": "mean_mapped_frame_posterior",
                "posterior_like_available": True,
                "logit_like_available": True,
                "approximation_only": False,
                "real_backend": True,
                "real_model_output": True,
                "frame_count": len(windows),
                **self.model_metadata(),
            },
            "chunked_inference": self.chunk_duration_sec is not None,
            "chunk_indices": sorted(
                {
                    int(window["_chunk_index"])
                    for window in windows
                    if "_chunk_index" in window
                }
            ),
        }
        return PhoneSegment(
            time_range=TimeRange(start_index / sample_rate, end_index / sample_rate),
            candidate_phones=candidates,
            confidence=candidates[0].confidence,
            metadata=metadata,
        )

    def recognize(self, vocal_audio_path: Path, lyrics: str | None = None) -> list[PhoneSegment]:
        sample_rate, samples = self._read_pcm16_mono(vocal_audio_path)
        predictions = (
            self._predict_chunked_frames(samples, sample_rate)
            if self.chunk_duration_sec is not None
            else self._predict_frames(samples, sample_rate)
        )
        self.finalize_profile()
        segments: list[PhoneSegment] = []
        current_symbol: str | None = None
        current_windows: list[dict[str, Any]] = []
        start_index = 0
        for prediction in predictions:
            symbol = prediction.get("symbol")
            if not isinstance(symbol, str) or not symbol:
                if current_symbol is not None:
                    segments.append(
                        self._build_real_segment(
                            current_symbol,
                            current_windows,
                            start_index,
                            int(current_windows[-1]["end_index"]),
                            sample_rate,
                            None,
                            None,
                            lyrics,
                        )
                    )
                current_symbol = None
                current_windows = []
                continue
            if current_symbol is not None and symbol != current_symbol:
                segments.append(
                    self._build_real_segment(
                        current_symbol,
                        current_windows,
                        start_index,
                        int(current_windows[-1]["end_index"]),
                        sample_rate,
                        None,
                        None,
                        lyrics,
                    )
                )
                current_windows = []
            if not current_windows:
                start_index = int(prediction["start_index"])
            current_symbol = symbol
            current_windows.append(prediction)
        if current_symbol is not None and current_windows:
            segments.append(
                self._build_real_segment(
                    current_symbol,
                    current_windows,
                    start_index,
                    int(current_windows[-1]["end_index"]),
                    sample_rate,
                    None,
                    None,
                    lyrics,
                )
            )
        if not segments:
            raise OnnxEspeakInferenceError(
                "ONNX/eSpeak model produced no non-blank phone segments for this audio."
            )
        return segments


__all__ = [
    "OnnxEspeakCTCUniversalPhoneRecognizer",
    "OnnxEspeakInferenceError",
    "OnnxEspeakResourceError",
]
