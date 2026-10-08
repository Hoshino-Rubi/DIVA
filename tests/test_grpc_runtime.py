from __future__ import annotations

from concurrent import futures
import json
from typing import Any

import grpc

from any2xsampa.grpc.generated import miku_v4c_product_pb2 as pb2
from any2xsampa.grpc.generated import miku_v4c_product_pb2_grpc as pb2_grpc
from any2xsampa.grpc.product_bridge import InferDraftVsqxResponse
from any2xsampa.grpc.servicer import MikuV4CProductDraftServicer


class _Bridge:
    def __init__(self) -> None:
        self.request: dict[str, Any] | None = None

    def infer_draft_vsqx(self, request: dict[str, Any]) -> InferDraftVsqxResponse:
        self.request = request
        return InferDraftVsqxResponse(
            ok=True,
            schema_version="miku_v4c_grpc_product_bridge_v1",
            product_summary={"ok": True, "truth_layer": "phones_ipa"},
            artifacts={
                "vsqx": "out/song.vsqx",
                "draft": "out/song.json",
                "phoneme_lattice": "out/lattice.json",
                "export_summary": "out/export.json",
                "audio_product_summary": "out/audio.json",
            },
            lyric_source="user_lyrics",
            manual_vocaloid_import_required=True,
        )


class _FailingBridge:
    def infer_draft_vsqx(self, request: dict[str, Any]) -> InferDraftVsqxResponse:
        raise RuntimeError("local secret path must not be exposed")


def test_generated_grpc_contract_round_trip() -> None:
    bridge = _Bridge()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    pb2_grpc.add_MikuV4CProductDraftServiceServicer_to_server(
        MikuV4CProductDraftServicer(bridge),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = pb2_grpc.MikuV4CProductDraftServiceStub(channel)
            response = stub.InferDraftVsqx(
                pb2.InferDraftVsqxRequest(
                    audio_path="in.wav",
                    template_path="template.vsqx",
                    output_root="out",
                    lyrics="laa",
                    jyutping_override="laa1",
                    asr_transcript_path="asr.json",
                    pitch_analysis_path="pitch.json",
                    qwen3_asr_model="Qwen/Qwen3-ASR-0.6B",
                    use_fake_asr=False,
                    timestamps=True,
                    language="Chinese",
                    context="Cantonese singing",
                    pitches_csv="60,62",
                    tempo=128.0,
                    prefix="song",
                ),
                timeout=5,
            )
    finally:
        server.stop(grace=None).wait()

    assert bridge.request is not None
    assert bridge.request["qwen3_asr_model"] == "Qwen/Qwen3-ASR-0.6B"
    assert bridge.request["pitches_csv"] == "60,62"
    assert response.ok is True
    assert json.loads(response.product_summary_json)["truth_layer"] == "phones_ipa"
    assert response.vsqx_path == "out/song.vsqx"
    assert response.phoneme_lattice_path == "out/lattice.json"
    assert response.manual_vocaloid_import_required is True


def test_servicer_converts_unexpected_bridge_failure_to_stable_response() -> None:
    response = MikuV4CProductDraftServicer(_FailingBridge()).InferDraftVsqx(
        pb2.InferDraftVsqxRequest(audio_path="in.wav"),
        context=None,
    )
    assert response.ok is False
    assert response.error_kind == "internal_error"
    assert list(response.errors) == ["draft generation failed"]
    assert response.manual_vocaloid_import_required is True
