from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import wave
from pathlib import Path
from xml.etree import ElementTree as ET

from any2xsampa.adapters.forced_alignment_acceptance import (
    canonical_phone_inventory_sha256,
)
from any2xsampa.vocaloid.vsqx import VSQ4_NS
from tools.verify_miku_v4c_vsqx_smoke import (
    _verify_forced_acceptance,
    _verify_human_review_acceptance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NS = {"v": VSQ4_NS}


def _review_wav(path: Path, duration_sec: float = 12.0) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * int(16000 * duration_sec))
    return path


def _acceptance_artifact(
    path: Path,
    *,
    stem_hash: str,
    scope: str,
    reviewed_range: list[float],
    predecessor: dict[str, str] | None = None,
) -> dict[str, object]:
    segment = {
        "candidate_phones": [{"symbol": "e", "confidence": 1.0}],
        "confidence": 1.0,
        "normalized_ipa": "e",
        "xsampa": "",
        "engine_export": None,
        "dialect_result": None,
        "syllable_group": None,
        "metadata": {},
        "start_sec": reviewed_range[0],
        "end_sec": reviewed_range[1],
        "reviewed": True,
    }
    canonical_segment = {key: value for key, value in segment.items() if key != "reviewed"}
    segment_bytes = json.dumps(
        [canonical_segment], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    alignment_path = path.with_name(f"{path.stem}-alignment.json")
    alignment_path.write_text(
        json.dumps(
            {
                "schema_version": "forced_phone_alignment_v1",
                "vocal_stem_sha256": stem_hash,
                "segments": [segment],
            }
        ),
        encoding="utf-8",
    )
    review: dict[str, object] = {
        "review_status": "approved",
        "reviewer_kind": "human",
        "reviewer_id": "reviewer-1",
        "reviewed_at": "2026-08-15T12:00:00+08:00",
        "review_method": "auditory_acoustic_phone_boundary_review",
        "phone_boundaries_reviewed": True,
        "scope": scope,
        "reviewed_range_sec": reviewed_range,
    }
    if predecessor is not None:
        review["short_clip_predecessor"] = predecessor
    payload: dict[str, object] = {
        "schema_version": "forced_phone_acceptance_v1",
        "route": "human_reviewed_phone_alignment",
        "alignment_artifact_path": alignment_path.name,
        "alignment_artifact_sha256": hashlib.sha256(alignment_path.read_bytes()).hexdigest(),
        "vocal_stem_sha256": stem_hash,
        "phone_segments_sha256": hashlib.sha256(segment_bytes).hexdigest(),
        "phone_inventory": {
            "symbols": ["e"],
            "sha256": canonical_phone_inventory_sha256(["e"]),
            "source": "fixture",
        },
        "review": review,
    }
    encoded = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    path.write_bytes(encoded)
    return {
        **payload,
        "alignment_artifact_path": str(alignment_path.resolve()),
        "acceptance_artifact_path": str(path),
        "acceptance_artifact_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _write_smoke_vsqx(path: Path, phonemes: list[str]) -> None:
    root = ET.Element(f"{{{VSQ4_NS}}}vsq4")
    part = ET.SubElement(ET.SubElement(root, f"{{{VSQ4_NS}}}vsTrack"), f"{{{VSQ4_NS}}}vsPart")
    for index, (lyric, phoneme) in enumerate(zip(["\u559c", "\u6b61", "\u4f60"], phonemes)):
        note = ET.SubElement(part, f"{{{VSQ4_NS}}}note")
        ET.SubElement(note, f"{{{VSQ4_NS}}}t").text = str(index * 480)
        ET.SubElement(note, f"{{{VSQ4_NS}}}dur").text = "480"
        ET.SubElement(note, f"{{{VSQ4_NS}}}n").text = "64"
        ET.SubElement(note, f"{{{VSQ4_NS}}}y").text = lyric
        ET.SubElement(note, f"{{{VSQ4_NS}}}p").text = phoneme
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def test_vsqx_smoke_verifier_accepts_expected_phrase(tmp_path: Path) -> None:
    vsqx = tmp_path / "ok.vsqx"
    _write_smoke_vsqx(vsqx, ["x ei", "f u n", "n ei"])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    result = subprocess.run(
        [sys.executable, "tools/verify_miku_v4c_vsqx_smoke.py", str(vsqx), "--expected-smoke"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["vsqx"]["manual_playback_required"] is True


def test_vsqx_smoke_verifier_rejects_known_silent_sequence(tmp_path: Path) -> None:
    vsqx = tmp_path / "bad.vsqx"
    _write_smoke_vsqx(vsqx, ["h e i", "f UN", "n e i"])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    result = subprocess.run(
        [sys.executable, "tools/verify_miku_v4c_vsqx_smoke.py", str(vsqx), "--expected-smoke"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert any("forbidden" in error for error in payload["errors"])


def test_vsqx_smoke_verifier_checks_product_evidence_gate(tmp_path: Path) -> None:
    vsqx = tmp_path / "ok.vsqx"
    provenance = tmp_path / "provenance.json"
    _write_smoke_vsqx(vsqx, ["x ei", "f u n", "n ei"])
    provenance.write_text(
        json.dumps(
            {
                "lyric_source": "user_lyrics",
                "pitch_evidence": {"provided": True},
                "phone_evidence": {"provided": True},
                "evidence_fusion": {
                    "direct_overlap_ratio": 0.95,
                    "interpolation_ratio": 0.05,
                    "interpolation_count": 1,
                    "interpolations": [{"pitch_index": 19, "reason": "nearest"}],
                },
            }
        ),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    result = subprocess.run(
        [
            sys.executable,
            "tools/verify_miku_v4c_vsqx_smoke.py",
            str(vsqx),
            "--expected-smoke",
            "--product-provenance",
            str(provenance),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True, check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["product_provenance"]["ok"] is True


def test_product_verifier_rejects_non_finite_ratios(tmp_path: Path) -> None:
    provenance = tmp_path / "nan-provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "lyric_source": "user_lyrics",
                "pitch_evidence": {"provided": True},
                "phone_evidence": {"provided": True},
                "evidence_fusion": {
                    "direct_overlap_ratio": "NaN",
                    "interpolation_ratio": "NaN",
                    "interpolation_count": 0,
                    "interpolations": [],
                },
            }
        ),
        encoding="utf-8",
    )
    vsqx = tmp_path / "ok.vsqx"
    _write_smoke_vsqx(vsqx, ["x ei", "f u n", "n ei"])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    result = subprocess.run(
        [sys.executable, "tools/verify_miku_v4c_vsqx_smoke.py", str(vsqx), "--product-provenance", str(provenance)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert any("finite" in error for error in report["product_provenance"]["errors"])


def test_forced_product_provenance_requires_complete_evidence(tmp_path: Path) -> None:
    provenance = tmp_path / "forced-provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "product_mode": "forced_alignment_product",
                "lyric_source": "manual_alignment",
                "vocal_stem": {"path": "stem.wav", "sha256": "1" * 64},
                "pitch_evidence": {"provided": True},
                "phone_evidence": {"provided": True},
                "lyrics_timeline_evidence": {"provided": True, "item_count": 3, "audio_sha256": "2" * 64, "timeline_sha256": "3" * 64},
                "g2p_evidence": {"item_count": 3, "oov_items": [], "audio_sha256": "4" * 64, "lyrics_timeline_sha256": "5" * 64, "artifact_sha256": "6" * 64},
                "forced_alignment_evidence": {"provided": True, "covered_lyric_item_count": 3},
                "evidence_fusion": {"direct_overlap_ratio": 1.0, "interpolation_ratio": 0.0, "interpolation_count": 0, "interpolations": []},
                "manual_vocaloid4_import_required": True,
                "manual_acceptance_status": "pending_user_validation",
            }
        ),
        encoding="utf-8",
    )
    vsqx = tmp_path / "ok.vsqx"
    _write_smoke_vsqx(vsqx, ["x ei", "f u n", "n ei"])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")

    result = subprocess.run(
        [sys.executable, "tools/verify_miku_v4c_vsqx_smoke.py", str(vsqx), "--product-provenance", str(provenance)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert any(
        "qualifying forced-phone acceptance" in error
        for error in report["product_provenance"]["errors"]
    )


def test_human_acceptance_verifier_enforces_clip_range_and_full_predecessor(
    tmp_path: Path,
) -> None:
    stem = _review_wav(tmp_path / "stem.wav")
    stem_hash = hashlib.sha256(stem.read_bytes()).hexdigest()
    stem_evidence = {"path": str(stem), "sha256": stem_hash}

    short = _acceptance_artifact(
        tmp_path / "short-invalid.json",
        stem_hash=stem_hash,
        scope="short_clip",
        reviewed_range=[0.0, 9.9],
    )
    errors: list[str] = []
    _verify_human_review_acceptance(short, stem=stem_evidence, errors=errors)
    assert any("10-20 seconds" in error for error in errors)

    full = _acceptance_artifact(
        tmp_path / "full-missing-predecessor.json",
        stem_hash=stem_hash,
        scope="full_song",
        reviewed_range=[0.0, 12.0],
    )
    errors = []
    _verify_human_review_acceptance(full, stem=stem_evidence, errors=errors)
    assert any("no short-clip predecessor" in error for error in errors)


def test_human_acceptance_verifier_accepts_full_review_with_real_predecessor(
    tmp_path: Path,
) -> None:
    stem = _review_wav(tmp_path / "stem.wav")
    stem_hash = hashlib.sha256(stem.read_bytes()).hexdigest()
    stem_evidence = {"path": str(stem), "sha256": stem_hash}
    short = _acceptance_artifact(
        tmp_path / "short.json",
        stem_hash=stem_hash,
        scope="short_clip",
        reviewed_range=[0.0, 12.0],
    )
    full = _acceptance_artifact(
        tmp_path / "full.json",
        stem_hash=stem_hash,
        scope="full_song",
        reviewed_range=[0.0, 12.0],
        predecessor={
            "acceptance_path": str(short["acceptance_artifact_path"]),
            "acceptance_sha256": str(short["acceptance_artifact_sha256"]),
        },
    )

    errors: list[str] = []
    _verify_human_review_acceptance(full, stem=stem_evidence, errors=errors)
    assert errors == []


def test_human_acceptance_verifier_requires_full_song_start_at_zero(tmp_path: Path) -> None:
    stem = _review_wav(tmp_path / "stem.wav")
    stem_hash = hashlib.sha256(stem.read_bytes()).hexdigest()
    stem_evidence = {"path": str(stem), "sha256": stem_hash}
    short = _acceptance_artifact(
        tmp_path / "short.json",
        stem_hash=stem_hash,
        scope="short_clip",
        reviewed_range=[0.0, 12.0],
    )
    full = _acceptance_artifact(
        tmp_path / "full.json",
        stem_hash=stem_hash,
        scope="full_song",
        reviewed_range=[0.1, 12.0],
        predecessor={
            "acceptance_path": str(short["acceptance_artifact_path"]),
            "acceptance_sha256": str(short["acceptance_artifact_sha256"]),
        },
    )

    errors: list[str] = []
    _verify_human_review_acceptance(full, stem=stem_evidence, errors=errors)
    assert any("complete vocal stem" in error for error in errors)


def test_forced_acceptance_requires_real_generated_alignment_artifact(tmp_path: Path) -> None:
    stem = _review_wav(tmp_path / "stem.wav")
    stem_hash = hashlib.sha256(stem.read_bytes()).hexdigest()
    source = tmp_path / "source-alignment.json"
    source.write_text("{}", encoding="utf-8")
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text("{}", encoding="utf-8")

    errors: list[str] = []
    _verify_forced_acceptance(
        {
            "path": str(source),
            "source_artifact_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "artifact_path": str(tmp_path / "missing-generated-alignment.json"),
            "artifact_sha256": "0" * 64,
            "acceptance": {
                "schema_version": "forced_phone_acceptance_v1",
                "route": "human_reviewed_phone_alignment",
                "acceptance_artifact_path": str(acceptance),
                "acceptance_artifact_sha256": hashlib.sha256(acceptance.read_bytes()).hexdigest(),
            },
        },
        stem={"path": str(stem), "sha256": stem_hash},
        errors=errors,
    )
    assert any("forced alignment artifact path does not exist" in error for error in errors)
