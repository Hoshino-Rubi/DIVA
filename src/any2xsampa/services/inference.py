from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from any2xsampa.adapters.recognizer import RecognizerConfig
from any2xsampa.dialects.registry import DialectRegistry
from any2xsampa.pipeline.note_context import ScoreNoteContextLoader
from any2xsampa.pipeline.runner import SingingToPhonesPipeline
from any2xsampa.presentation import with_status_capabilities


SUPPORTED_EXPORT_FORMATS = ["xsampa", "vocaloid_like", "openutau_like"]
SUPPORTED_NOTE_CONTEXT_FORMATS = ["auto", "json", "ustx", "ust"]
SUPPORTED_RECOGNIZER_MODES = ["fixture", "real"]


class RecognizerContractError(RuntimeError):
    """Stable transport-facing error for failures in the real recognizer adapter."""

    error_code_suffix: str
    error_kind: str

    def __init__(
        self,
        message: str,
        error_code_suffix: str = "recognizer_error",
        error_kind: str = "recognizer_error",
    ) -> None:
        super().__init__(message)
        self.error_code_suffix = str(error_code_suffix)
        self.error_kind = str(error_kind)


@dataclass(slots=True)
class InferenceRequest:
    mode: str
    audio_path: str
    dialect: str
    lyrics: str | None = None
    export_format: str = "xsampa"
    engine_profile: str | None = None
    engine_inventory: str | None = None
    engine_subbank: str | None = None
    engine_color: str | None = None
    engine_style: str | None = None
    engine_tags: list[str] = field(default_factory=list)
    note_context: str | None = None
    note_context_format: str = "auto"
    recognizer_mode: str = "fixture"
    recognizer_fixture: str | None = None
    recognizer_real_backend: str | None = None
    recognizer_real_command: str | None = None
    recognizer_real_output: str = "real_recognizer_output.json"
    recognizer_model: str | None = None
    recognizer_device: str = "auto"
    disable_recognizer_fallback: bool = False
    debug_allow_recognizer_stub_fallback: bool = False

    def __post_init__(self) -> None:
        self.mode = str(self.mode).strip().lower()
        self.audio_path = str(self.audio_path).strip()
        self.dialect = str(self.dialect).strip().lower()
        self.export_format = str(self.export_format).strip().lower()
        self.note_context_format = str(self.note_context_format or "auto").strip().lower()
        self.recognizer_mode = str(self.recognizer_mode or "fixture").strip().lower()
        self.recognizer_real_output = str(self.recognizer_real_output or "real_recognizer_output.json").strip()
        self.recognizer_model = _optional_text(self.recognizer_model)
        self.recognizer_device = str(self.recognizer_device or "auto").strip().lower()
        self.engine_tags = [str(tag).strip() for tag in self.engine_tags if str(tag).strip()]
        if self.mode not in {"infer-open", "infer-with-lyrics"}:
            raise ValueError("mode must be 'infer-open' or 'infer-with-lyrics'")
        if not self.audio_path:
            raise ValueError("audio_path is required")
        if not self.dialect:
            raise ValueError("dialect is required")
        if self.mode == "infer-with-lyrics" and not (self.lyrics and str(self.lyrics).strip()):
            raise ValueError("lyrics is required for infer-with-lyrics")
        if self.export_format not in SUPPORTED_EXPORT_FORMATS:
            raise ValueError(
                f"Unsupported export_format '{self.export_format}'. "
                f"Available formats: {', '.join(SUPPORTED_EXPORT_FORMATS)}"
            )
        if self.note_context_format not in SUPPORTED_NOTE_CONTEXT_FORMATS:
            raise ValueError(
                f"Unsupported note_context_format '{self.note_context_format}'. "
                f"Available formats: {', '.join(SUPPORTED_NOTE_CONTEXT_FORMATS)}"
            )
        if self.recognizer_mode not in SUPPORTED_RECOGNIZER_MODES:
            raise ValueError(
                f"Unsupported recognizer_mode '{self.recognizer_mode}'. "
                f"Available modes: {', '.join(SUPPORTED_RECOGNIZER_MODES)}"
            )
        if self.recognizer_device not in {"auto", "cuda", "cpu"}:
            raise ValueError(
                f"Unsupported recognizer_device '{self.recognizer_device}'. "
                "Available devices: auto, cuda, cpu"
            )
        if self.recognizer_real_backend and self.recognizer_real_command:
            raise ValueError(
                "recognizer_real_backend and recognizer_real_command are mutually exclusive"
            )

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], mode: str) -> InferenceRequest:
        return cls(
            mode=mode,
            audio_path=str(payload.get("audio_path", "")).strip(),
            dialect=str(payload.get("dialect", "")).strip(),
            lyrics=_optional_text(payload.get("lyrics")),
            export_format=str(payload.get("export_format", "xsampa")).strip(),
            engine_profile=_optional_text(payload.get("engine_profile")),
            engine_inventory=_optional_text(payload.get("engine_inventory")),
            engine_subbank=_optional_text(payload.get("engine_subbank")),
            engine_color=_optional_text(payload.get("engine_color")),
            engine_style=_optional_text(payload.get("engine_style")),
            engine_tags=_tag_list(payload.get("engine_tags")),
            note_context=_optional_text(payload.get("note_context")),
            note_context_format=str(payload.get("note_context_format", "auto")).strip(),
            recognizer_mode=str(payload.get("recognizer_mode", "fixture")).strip(),
            recognizer_fixture=_optional_text(payload.get("recognizer_fixture")),
            recognizer_real_backend=_optional_text(payload.get("recognizer_real_backend")),
            recognizer_real_command=_optional_text(payload.get("recognizer_real_command")),
            recognizer_real_output=str(payload.get("recognizer_real_output", "real_recognizer_output.json")).strip(),
            recognizer_model=_optional_text(payload.get("recognizer_model")),
            recognizer_device=str(payload.get("recognizer_device", "auto")).strip(),
            disable_recognizer_fallback=_strict_bool(
                payload, "disable_recognizer_fallback", default=False
            ),
            debug_allow_recognizer_stub_fallback=_strict_bool(
                payload, "debug_allow_recognizer_stub_fallback", default=False
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "audio_path": self.audio_path,
            "dialect": self.dialect,
            "export_format": self.export_format,
            "engine_profile": self.engine_profile,
            "engine_inventory": self.engine_inventory,
            "engine_subbank": self.engine_subbank,
            "engine_color": self.engine_color,
            "engine_style": self.engine_style,
            "engine_tags": list(self.engine_tags),
            "note_context": self.note_context,
            "note_context_format": self.note_context_format,
            "recognizer_mode": self.recognizer_mode,
            "recognizer_fixture": self.recognizer_fixture,
            "recognizer_real_backend": self.recognizer_real_backend,
            "recognizer_real_command": self.recognizer_real_command,
            "recognizer_real_output": self.recognizer_real_output,
            "disable_recognizer_fallback": self.disable_recognizer_fallback,
        }
        if self.lyrics:
            payload["lyrics"] = self.lyrics
        return payload

    def to_runtime_mapping(self) -> dict[str, Any]:
        """Serialize the complete request for trusted in-process transports."""

        payload = self.to_mapping()
        payload.update(
            {
                "recognizer_model": self.recognizer_model,
                "recognizer_device": self.recognizer_device,
                "debug_allow_recognizer_stub_fallback": (
                    self.debug_allow_recognizer_stub_fallback
                ),
            }
        )
        return payload

    @classmethod
    def from_args_namespace(
        cls,
        args: Any,
        *,
        mode: str,
    ) -> InferenceRequest:
        return cls.from_mapping(_request_mapping_from_namespace(args), mode=mode)

    def validate_paths(self) -> None:
        if not Path(self.audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {self.audio_path}")
        for label, candidate in (
            ("Recognizer fixture", self.recognizer_fixture),
            ("Recognizer model", self.recognizer_model),
            ("Engine inventory", self.engine_inventory),
            ("Note context file", self.note_context),
        ):
            if candidate is not None and not Path(candidate).exists():
                raise FileNotFoundError(f"{label} not found: {candidate}")

    def engine_metadata_preferences(self) -> dict[str, list[str]] | None:
        preferences: dict[str, list[str]] = {}
        if self.engine_subbank:
            preferences["subbanks"] = [self.engine_subbank]
        if self.engine_color:
            preferences["colors"] = [self.engine_color]
        if self.engine_style:
            preferences["styles"] = [self.engine_style]
        if self.engine_tags:
            preferences["tags"] = list(self.engine_tags)
        return preferences or None


class InferenceService:
    def __init__(self) -> None:
        self.package_root = Path(__file__).resolve().parent.parent
        self.fixture_root = self.package_root / "recognizer_fixtures"
        self.default_fixture = self.fixture_root / "default_fixture.json"
        self.dialect_registry = DialectRegistry()
        self.note_context_loader = ScoreNoteContextLoader()

    def health(self) -> dict[str, object]:
        return {"status": "ok", "service": "any2xsampa"}

    def status(self) -> dict[str, object]:
        return with_status_capabilities(
            {
            "status": "ok",
            "service": "any2xsampa",
                "supported_dialects": self.dialect_registry.list_names(),
                "supported_export_formats": SUPPORTED_EXPORT_FORMATS,
                "supported_note_context_formats": self.list_supported_note_context_formats()["note_context_formats"],
                "supported_recognizer_modes": SUPPORTED_RECOGNIZER_MODES,
            }
        )

    def list_supported_export_formats(self) -> dict[str, object]:
        return {"export_formats": list(SUPPORTED_EXPORT_FORMATS)}

    def list_supported_note_context_formats(self) -> dict[str, object]:
        adapter_names = ["auto", *[adapter.name for adapter in self.note_context_loader.adapters]]
        deduped: list[str] = []
        for adapter_name in adapter_names:
            if adapter_name not in deduped:
                deduped.append(adapter_name)
        return {"note_context_formats": deduped}

    def run(self, request: InferenceRequest) -> dict[str, Any]:
        request.validate_paths()
        if request.dialect not in self.dialect_registry.list_names():
            available = ", ".join(self.dialect_registry.list_names())
            raise ValueError(f"Unknown dialect '{request.dialect}'. Available dialects: {available}")
        recognizer_config = RecognizerConfig(
            mode=request.recognizer_mode,
            fixture_path=Path(request.recognizer_fixture) if request.recognizer_fixture else self.default_fixture,
            real_backend=request.recognizer_real_backend,
            real_command=request.recognizer_real_command,
            real_output_path=Path(request.recognizer_real_output),
            model_path=Path(request.recognizer_model) if request.recognizer_model else None,
            device=request.recognizer_device,
            allow_stub_fallback=(
                request.debug_allow_recognizer_stub_fallback
                and not request.disable_recognizer_fallback
            ),
        )
        try:
            pipeline = SingingToPhonesPipeline(recognizer_config=recognizer_config)
            result = pipeline.run(
                audio_path=request.audio_path,
                dialect=request.dialect,
                mode=request.mode,
                lyrics=request.lyrics,
                export_format=request.export_format,
                engine_profile=request.engine_profile,
                engine_inventory=request.engine_inventory,
                engine_metadata_preferences=request.engine_metadata_preferences(),
                note_context=request.note_context,
                note_context_format=request.note_context_format,
            )
        except RecognizerContractError:
            raise
        except Exception as exc:
            if request.recognizer_mode == "real":
                raise RecognizerContractError(
                    str(exc),
                    error_code_suffix="real_backend_unavailable",
                    error_kind="real_backend_unavailable",
                ) from exc
            raise
        return result.to_dict()


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _strict_bool(payload: dict[str, Any], field_name: str, *, default: bool) -> bool:
    if field_name not in payload:
        return default
    value = payload[field_name]
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a JSON boolean")
    return value


def _tag_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(tag).strip() for tag in value if str(tag).strip()]
    text = str(value).strip()
    return [text] if text else []


def _request_mapping_from_namespace(args: Any) -> dict[str, Any]:
    return {
        "audio_path": getattr(args, "audio", ""),
        "dialect": getattr(args, "dialect", ""),
        "lyrics": getattr(args, "lyrics", None),
        "export_format": getattr(args, "export_format", "xsampa"),
        "engine_profile": getattr(args, "engine_profile", None),
        "engine_inventory": getattr(args, "engine_inventory", None),
        "engine_subbank": getattr(args, "engine_subbank", None),
        "engine_color": getattr(args, "engine_color", None),
        "engine_style": getattr(args, "engine_style", None),
        "engine_tags": list(getattr(args, "engine_tag", []) or []),
        "note_context": getattr(args, "note_context", None),
        "note_context_format": getattr(args, "note_context_format", "auto"),
        "recognizer_mode": getattr(args, "recognizer_mode", "fixture"),
        "recognizer_fixture": getattr(args, "recognizer_fixture", None),
        "recognizer_real_backend": getattr(args, "recognizer_real_backend", None),
        "recognizer_real_command": getattr(args, "recognizer_real_command", None),
        "recognizer_real_output": getattr(args, "recognizer_real_output", "real_recognizer_output.json"),
        "recognizer_model": getattr(args, "recognizer_model", None),
        "recognizer_device": getattr(args, "recognizer_device", "auto"),
        "disable_recognizer_fallback": bool(getattr(args, "disable_recognizer_fallback", False)),
        "debug_allow_recognizer_stub_fallback": bool(
            getattr(args, "debug_allow_recognizer_stub_fallback", False)
        ),
    }
