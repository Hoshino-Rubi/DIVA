from __future__ import annotations

"""Python-side bridge for the WinUI/gRPC Miku V4C draft VSQX contract.

The actual network gRPC runtime is optional.  This module keeps a small typed
service surface that can be tested without grpcio while the WinUI 3 client and
real grpc server wiring are being added.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "miku_v4c_grpc_product_bridge_v1"


@dataclass(frozen=True, slots=True)
class InferDraftVsqxRequest:
    audio_path: Path
    template_path: Path
    output_root: Path
    lyrics: str | None = None
    jyutping_override: str | None = None
    asr_transcript_path: Path | None = None
    pitch_analysis_path: Path | None = None
    qwen3_asr_model: str | None = None
    use_fake_asr: bool = False
    timestamps: bool = False
    language: str | None = "Chinese"
    context: str = ""
    pitches: list[int] | None = None
    tempo: float = 128.0
    prefix: str = "miku_v4c_grpc_draft"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "InferDraftVsqxRequest":
        return cls(
            audio_path=Path(_required_text(payload, "audio_path")),
            template_path=Path(_required_text(payload, "template_path")),
            output_root=Path(_required_text(payload, "output_root")),
            lyrics=_optional_text(payload.get("lyrics")),
            jyutping_override=_optional_text(payload.get("jyutping_override")),
            asr_transcript_path=_optional_path(payload.get("asr_transcript_path")),
            pitch_analysis_path=_optional_path(payload.get("pitch_analysis_path")),
            qwen3_asr_model=_optional_text(payload.get("qwen3_asr_model")),
            use_fake_asr=bool(payload.get("use_fake_asr", False)),
            timestamps=bool(payload.get("timestamps", False)),
            language=_optional_text(payload.get("language")) or "Chinese",
            context=_optional_text(payload.get("context")) or "",
            pitches=_parse_pitches(payload.get("pitches") or payload.get("pitches_csv")),
            tempo=float(payload.get("tempo", 128.0)),
            prefix=_optional_text(payload.get("prefix")) or "miku_v4c_grpc_draft",
        )


@dataclass(frozen=True, slots=True)
class InferDraftVsqxResponse:
    ok: bool
    schema_version: str
    error_kind: str | None = None
    errors: list[str] = field(default_factory=list)
    product_summary: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    lyric_source: str | None = None
    manual_vocaloid_import_required: bool = True
    user_jyutping_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ok": self.ok,
            "error_kind": self.error_kind,
            "errors": list(self.errors),
            "product_summary": dict(self.product_summary),
            "artifacts": dict(self.artifacts),
            "lyric_source": self.lyric_source,
            "manual_vocaloid_import_required": self.manual_vocaloid_import_required,
            "user_jyutping_required": self.user_jyutping_required,
        }


class MikuV4CProductDraftBridge:
    """Core service used by the future grpc servicer and tests."""

    def infer_draft_vsqx(self, request: InferDraftVsqxRequest | Mapping[str, Any]) -> InferDraftVsqxResponse:
        if not isinstance(request, InferDraftVsqxRequest):
            request = InferDraftVsqxRequest.from_mapping(request)
        from tools.build_miku_v4c_audio_draft_vsqx import run_audio_draft_vsqx

        rc, summary = run_audio_draft_vsqx(
            audio=request.audio_path,
            template=request.template_path,
            output_root=request.output_root,
            lyrics=request.lyrics,
            jyutping_override=request.jyutping_override,
            asr_transcript=request.asr_transcript_path,
            fake_asr=request.use_fake_asr,
            asr_model=request.qwen3_asr_model,
            language=request.language,
            context=request.context,
            timestamps=request.timestamps,
            pitch_analysis=request.pitch_analysis_path,
            pitches=request.pitches,
            tempo=request.tempo,
            prefix=request.prefix,
            pretty=True,
        )
        artifacts = summary.get("artifacts", {}) if isinstance(summary.get("artifacts"), dict) else {}
        return InferDraftVsqxResponse(
            ok=rc == 0 and bool(summary.get("ok")),
            schema_version=SCHEMA_VERSION,
            error_kind=summary.get("error_kind"),
            errors=[str(error) for error in summary.get("errors", [])],
            product_summary=dict(summary),
            artifacts={str(key): str(value) for key, value in artifacts.items()},
            lyric_source=summary.get("lyric_source"),
            manual_vocaloid_import_required=bool(summary.get("manual_vocaloid_import_required", True)),
            user_jyutping_required=bool(summary.get("user_jyutping_required", False)),
        )


def grpc_runtime_available() -> bool:
    try:
        import grpc  # noqa: F401
    except Exception:
        return False
    return True


def require_grpc_runtime() -> None:
    if not grpc_runtime_available():
        raise RuntimeError(
            "grpcio is not installed. Install the optional desktop bridge dependencies, "
            "for example `pip install -e .[grpc]`, before starting the network gRPC server."
        )


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    text = _optional_text(payload.get(key))
    if not text:
        raise ValueError(f"{key} is required")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_path(value: object) -> Path | None:
    text = _optional_text(value)
    return Path(text) if text else None


def _parse_pitches(value: object) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        result = [int(item) for item in value]
    else:
        result = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    return result or None
