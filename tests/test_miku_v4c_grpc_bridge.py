from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from any2xsampa.grpc import (
    SCHEMA_VERSION,
    InferDraftVsqxRequest,
    MikuV4CProductDraftBridge,
    grpc_runtime_available,
    require_grpc_runtime,
)
from any2xsampa.vocaloid.vsqx import VSQ4_NS

REPO_ROOT = Path(__file__).resolve().parents[1]


def _template(path: Path) -> Path:
    root = ET.Element(f"{{{VSQ4_NS}}}vsq4")
    ET.SubElement(root, f"{{{VSQ4_NS}}}vender").text = "Yamaha corporation"
    ET.SubElement(root, f"{{{VSQ4_NS}}}version").text = "4.0.0.3"
    table = ET.SubElement(root, f"{{{VSQ4_NS}}}vVoiceTable")
    voice = ET.SubElement(table, f"{{{VSQ4_NS}}}vVoice")
    ET.SubElement(voice, f"{{{VSQ4_NS}}}bs").text = "1"
    ET.SubElement(voice, f"{{{VSQ4_NS}}}pc").text = "11"
    ET.SubElement(voice, f"{{{VSQ4_NS}}}id").text = "template"
    ET.SubElement(voice, f"{{{VSQ4_NS}}}name").text = "MIKU_V4X_Original_EVEC"
    ET.SubElement(root, f"{{{VSQ4_NS}}}mixer")
    master = ET.SubElement(root, f"{{{VSQ4_NS}}}masterTrack")
    ET.SubElement(master, f"{{{VSQ4_NS}}}seqName").text = "template"
    ET.SubElement(master, f"{{{VSQ4_NS}}}comment").text = "template"
    ET.SubElement(master, f"{{{VSQ4_NS}}}resolution").text = "480"
    ET.SubElement(master, f"{{{VSQ4_NS}}}preMeasure").text = "1"
    ET.SubElement(master, f"{{{VSQ4_NS}}}timeSig")
    ET.SubElement(master, f"{{{VSQ4_NS}}}tempo")
    track = ET.SubElement(root, f"{{{VSQ4_NS}}}vsTrack")
    ET.SubElement(track, f"{{{VSQ4_NS}}}tNo").text = "0"
    ET.SubElement(track, f"{{{VSQ4_NS}}}name").text = "Track"
    ET.SubElement(track, f"{{{VSQ4_NS}}}comment").text = "Track"
    part = ET.SubElement(track, f"{{{VSQ4_NS}}}vsPart")
    ET.SubElement(part, f"{{{VSQ4_NS}}}t").text = "3840"
    ET.SubElement(part, f"{{{VSQ4_NS}}}playTime").text = "1920"
    ET.SubElement(part, f"{{{VSQ4_NS}}}name").text = "Part"
    ET.SubElement(part, f"{{{VSQ4_NS}}}comment").text = "Part"
    ET.SubElement(part, f"{{{VSQ4_NS}}}sPlug")
    ET.SubElement(part, f"{{{VSQ4_NS}}}singer")
    ET.SubElement(part, f"{{{VSQ4_NS}}}plane").text = "0"
    ET.SubElement(root, f"{{{VSQ4_NS}}}monoTrack")
    ET.SubElement(root, f"{{{VSQ4_NS}}}stTrack")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def test_product_bridge_core_generates_vsqx_without_grpcio(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFFfake")
    request = InferDraftVsqxRequest(
        audio_path=audio,
        template_path=_template(tmp_path / "template.vsqx"),
        output_root=tmp_path / "out",
        lyrics="喜歡你",
        pitches=[64, 67, 69],
        prefix="bridge_smoke",
    )

    response = MikuV4CProductDraftBridge().infer_draft_vsqx(request)

    assert response.ok is True
    assert response.schema_version == SCHEMA_VERSION
    assert response.lyric_source == "user_lyrics"
    assert response.manual_vocaloid_import_required is True
    assert response.user_jyutping_required is False
    assert Path(response.artifacts["vsqx"]).exists()
    assert response.product_summary["model_output"] is False
    assert response.product_summary["draft_vsqx"]["export_summary"]["export_mode"] == "template_clone_import_safe"


def test_product_bridge_accepts_mapping_request(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFFfake")

    response = MikuV4CProductDraftBridge().infer_draft_vsqx(
        {
            "audio_path": str(audio),
            "template_path": str(_template(tmp_path / "template.vsqx")),
            "output_root": str(tmp_path / "out"),
            "lyrics": "喜歡你",
            "pitches_csv": "64,67,69",
        }
    )

    assert response.ok is True
    assert Path(response.to_dict()["artifacts"]["audio_product_summary"]).exists()


def test_grpc_runtime_guard_reports_missing_optional_dependency() -> None:
    if grpc_runtime_available():
        pytest.skip("grpcio is installed in this environment")
    with pytest.raises(RuntimeError, match="grpcio is not installed"):
        require_grpc_runtime()


def test_grpc_server_check_runtime_cli_fails_clearly_without_grpcio() -> None:
    if grpc_runtime_available():
        pytest.skip("grpcio is installed in this environment")
    result = subprocess.run(
        [sys.executable, "-m", "any2xsampa.grpc.server", "--check-runtime"],
        cwd=REPO_ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    payload = json.loads(result.stderr)
    assert payload["ok"] is False
    assert payload["error_kind"] == "grpc_runtime_missing"
    assert "pip install -e .[grpc]" in payload["error"]
