from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import string
from typing import Protocol

from any2xsampa.core.models import PhoneSegment

_ALLOWED_REAL_COMMAND_FIELDS = {"audio", "output"}


class RecognizerConfigLike(Protocol):
    mode: object
    fixture_path: object
    real_backend: object
    real_command: object
    real_output_path: object
    model_path: object
    device: object
    allow_stub_fallback: object


@dataclass(slots=True, frozen=True)
class RecognizerDispatchPlan:
    mode: str
    primary_kind: str
    fixture_path: Path | None = None
    real_backend: str | None = None
    real_command: str | None = None
    real_output_path: Path | None = None
    model_path: Path | None = None
    device: str = "auto"


def normalize_recognizer_mode(value: object) -> str:
    return str(value or "").strip().lower()


def build_recognizer_dispatch_plan(config: RecognizerConfigLike) -> RecognizerDispatchPlan:
    mode = normalize_recognizer_mode(config.mode)
    fixture_path = _optional_path(config.fixture_path)
    real_backend = _optional_text(config.real_backend)
    real_command = _optional_text(config.real_command)
    real_output_path = _optional_path(config.real_output_path)
    model_path = _optional_path(getattr(config, "model_path", None))
    device = str(getattr(config, "device", "auto") or "auto").strip().lower()
    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError("Recognizer device must be one of: auto, cuda, cpu.")

    if mode == "stub":
        raise ValueError("Recognizer stub mode has been removed from production dispatch; use fixture replay or an explicit real backend.")
    if mode == "fixture":
        if fixture_path is None:
            raise ValueError("Fixture mode requires a fixture_path.")
        return RecognizerDispatchPlan(
            mode=mode,
            primary_kind="fixture",
            fixture_path=fixture_path,
        )
    if mode == "real":
        if real_backend:
            return RecognizerDispatchPlan(
                mode=mode,
                primary_kind="real_backend",
                real_backend=real_backend,
                real_output_path=real_output_path,
                model_path=model_path,
                device=device,
            )
        if real_command is None or real_output_path is None:
            raise ValueError("Real mode requires real_backend or real_command with real_output_path.")
        return RecognizerDispatchPlan(
            mode=mode,
            primary_kind="real_command",
            real_command=real_command,
            real_output_path=real_output_path,
        )
    raise ValueError(f"Unknown recognizer mode: {config.mode}")


def recognizer_fallback_mode(config: RecognizerConfigLike) -> str:
    return normalize_recognizer_mode(config.mode)


def recognizer_debug_fallback_requested(config: RecognizerConfigLike) -> bool:
    return bool(getattr(config, "allow_stub_fallback", False))


def annotate_debug_fallback_rejection(
    segments: list[PhoneSegment],
    *,
    fallback_mode: str,
    plan: RecognizerDispatchPlan,
    error: Exception,
) -> list[PhoneSegment]:
    """Attach explicit fallback rejection metadata to already-real replay segments."""
    fallback_metadata = {
        "requested_mode": fallback_mode,
        "primary_kind": plan.primary_kind,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "fallback_status": "rejected_no_synthetic_debug_recognizer",
    }
    if plan.fixture_path is not None:
        fallback_metadata["fixture_path"] = str(plan.fixture_path)
    if plan.real_backend is not None:
        fallback_metadata["real_backend"] = plan.real_backend
    if plan.real_output_path is not None:
        fallback_metadata["real_output_path"] = str(plan.real_output_path)
    for segment in segments:
        metadata = segment.metadata
        metadata["fallback_from_mode"] = fallback_mode
        metadata["recognizer_fallback"] = dict(fallback_metadata)
    return segments


def validate_real_command_template(command: str) -> None:
    formatter = string.Formatter()
    unknown_fields: list[str] = []
    for _, field_name, _, _ in formatter.parse(command):
        if field_name is None:
            continue
        normalized = field_name.strip()
        if not normalized:
            continue
        if normalized not in _ALLOWED_REAL_COMMAND_FIELDS and normalized not in unknown_fields:
            unknown_fields.append(normalized)
    if unknown_fields:
        allowed = ", ".join(f"{{{name}}}" for name in sorted(_ALLOWED_REAL_COMMAND_FIELDS))
        unknown = ", ".join(f"{{{name}}}" for name in unknown_fields)
        raise ValueError(
            "Real recognizer command template contains unknown placeholder(s): "
            f"{unknown}. Allowed placeholders: {allowed}. "
            "If you need literal braces in the command, escape them as '{{' and '}}'."
        )


def render_real_command(
    command_template: str,
    *,
    audio_path: Path,
    output_path: Path,
    lyrics: str | None = None,
) -> str:
    if not command_template.strip():
        raise ValueError("Real recognizer command is not configured.")
    validate_real_command_template(command_template)
    try:
        return command_template.format(
            audio=str(audio_path),
            output=str(output_path),
            lyrics=lyrics or "",
        )
    except KeyError as exc:
        missing = str(exc).strip("\"") or "<unknown>"
        raise ValueError(
            "Real recognizer command template references an unknown placeholder "
            f"'{missing}'. Allowed placeholders: {{audio}}, {{output}}. "
            "If you need literal braces in the command, escape them as '{{' and '}}'."
        ) from exc


def render_real_command_args(
    command_template: str,
    *,
    audio_path: Path,
    output_path: Path,
) -> list[str]:
    """Render an argv vector without invoking a command shell."""

    if not command_template.strip():
        raise ValueError("Real recognizer command is not configured.")
    validate_real_command_template(command_template)
    tokens = shlex.split(command_template, posix=os.name != "nt")
    if os.name == "nt":
        tokens = [
            token[1:-1]
            if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}
            else token
            for token in tokens
        ]
    if not tokens:
        raise ValueError("Real recognizer command is not configured.")
    return [
        token.format(audio=str(audio_path), output=str(output_path))
        for token in tokens
    ]


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    text = str(value).strip()
    return Path(text) if text else None
