from __future__ import annotations

import json
from typing import Any

from any2xsampa.grpc.generated import miku_v4c_product_pb2 as pb2
from any2xsampa.grpc.generated import miku_v4c_product_pb2_grpc as pb2_grpc
from any2xsampa.grpc.product_bridge import MikuV4CProductDraftBridge


class MikuV4CProductDraftServicer(pb2_grpc.MikuV4CProductDraftServiceServicer):
    """Translate the protobuf contract to the isolated product bridge."""

    def __init__(self, bridge: MikuV4CProductDraftBridge | Any) -> None:
        self.bridge = bridge

    def InferDraftVsqx(self, request: Any, context: Any) -> pb2.InferDraftVsqxResponse:
        payload = {
            "audio_path": request.audio_path,
            "template_path": request.template_path,
            "output_root": request.output_root,
            "lyrics": request.lyrics,
            "jyutping_override": request.jyutping_override,
            "asr_transcript_path": request.asr_transcript_path,
            "pitch_analysis_path": request.pitch_analysis_path,
            "qwen3_asr_model": request.qwen3_asr_model,
            "use_fake_asr": request.use_fake_asr,
            "timestamps": request.timestamps,
            "language": request.language,
            "context": request.context,
            "pitches_csv": request.pitches_csv,
            "tempo": request.tempo,
            "prefix": request.prefix,
        }
        try:
            response = self.bridge.infer_draft_vsqx(payload)
        except ValueError as exc:
            return pb2.InferDraftVsqxResponse(
                ok=False,
                error_kind="invalid_arguments",
                errors=[str(exc)],
                manual_vocaloid_import_required=True,
            )
        except Exception:
            return pb2.InferDraftVsqxResponse(
                ok=False,
                error_kind="internal_error",
                errors=["draft generation failed"],
                manual_vocaloid_import_required=True,
            )
        artifacts = response.artifacts
        return pb2.InferDraftVsqxResponse(
            ok=response.ok,
            schema_version=response.schema_version,
            error_kind=response.error_kind or "",
            errors=response.errors,
            product_summary_json=json.dumps(
                response.product_summary,
                ensure_ascii=False,
                sort_keys=True,
            ),
            vsqx_path=artifacts.get("vsqx", ""),
            draft_path=artifacts.get("draft", ""),
            phoneme_lattice_path=artifacts.get("phoneme_lattice", artifacts.get("lattice", "")),
            export_summary_path=artifacts.get("export_summary", artifacts.get("summary", "")),
            audio_product_summary_path=artifacts.get("audio_product_summary", ""),
            lyric_source=response.lyric_source or "",
            manual_vocaloid_import_required=response.manual_vocaloid_import_required,
            user_jyutping_required=response.user_jyutping_required,
        )
