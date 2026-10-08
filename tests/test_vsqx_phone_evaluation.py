from __future__ import annotations

import json
from pathlib import Path

from any2xsampa.vocaloid.vsqx_phone_evaluation import build_vsqx_phone_identity_evaluation


def test_human_vsqx_phone_annotations_score_top1_and_top3_separately(tmp_path: Path) -> None:
    vsqx_path = tmp_path / "reference.vsqx"
    vsqx_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<vsq4 xmlns="http://www.yamaha.co.jp/vocaloid/schema/vsq4/">
  <masterTrack><resolution>480</resolution><preMeasure>0</preMeasure><tempo><t>0</t><v>12000</v></tempo></masterTrack>
  <vsTrack><tNo>0</tNo><vsPart><t>0</t>
    <note><t>0</t><dur>480</dur><n>60</n><y>ma</y><p>m a</p></note>
    <note><t>480</t><dur>480</dur><n>62</n><y>ti</y><p>t i</p></note>
  </vsPart></vsTrack>
</vsq4>
""",
        encoding="utf-8",
    )
    phone_path = tmp_path / "phones.json"
    phone_path.write_text(
        json.dumps(
            {
                "mode": "infer-open",
                "audio_path": "audio.wav",
                "segments": [
                    _segment(0.0, 0.25, "m", ["m", "n", "N"]),
                    _segment(0.25, 0.5, "a", ["a", "o", "u"]),
                    _segment(0.5, 0.75, "k", ["k", "t", "p"]),
                    _segment(0.75, 1.0, "i", ["i", "e", "a"]),
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = build_vsqx_phone_identity_evaluation(phone_path, vsqx_path, profile="strict")
    summary = report["summary"]

    assert report["recognizer_redecoded"] is False
    assert report["vsqx_used_as_recognizer_input"] is False
    assert summary["note_count"] == 2
    assert summary["micro_top1_reference_phone_presence_ratio"] == 0.75
    assert summary["micro_top3_reference_phone_presence_ratio"] == 1.0
    assert summary["sequence_token_error_rate"] == 0.25
    assert summary["exact_winner_sequence_note_ratio"] == 0.5
    assert summary["top_substitutions"] == {"t->k": 1}


def test_reference_mapping_reports_unknown_v4c_tokens_without_counting_them_correct(tmp_path: Path) -> None:
    vsqx_path = tmp_path / "reference.vsqx"
    vsqx_path.write_text(
        """<vsq4 xmlns="http://www.yamaha.co.jp/vocaloid/schema/vsq4/"><masterTrack><resolution>480</resolution><tempo><t>0</t><v>12000</v></tempo></masterTrack><vsTrack><tNo>0</tNo><vsPart><t>0</t><note><t>0</t><dur>480</dur><n>60</n><y>x</y><p>7 a</p></note></vsPart></vsTrack></vsq4>""",
        encoding="utf-8",
    )
    phone_path = tmp_path / "phones.json"
    phone_path.write_text(
        json.dumps({"segments": [_segment(0.0, 0.5, "a", ["a", "i", "u"])]}),
        encoding="utf-8",
    )

    report = build_vsqx_phone_identity_evaluation(phone_path, vsqx_path, profile="strict")

    assert report["summary"]["unknown_reference_token_counts"] == {"7": 1}
    assert report["summary"]["configured_reference_token_ratio"] == 0.5
    assert report["notes"][0]["fully_mapped_reference"] is False
    assert report["notes"][0]["exact_winner_sequence_match"] is None


def _segment(start: float, end: float, winner: str, candidates: list[str]) -> dict[str, object]:
    return {
        "start_sec": start,
        "end_sec": end,
        "normalized_ipa": winner,
        "candidate_phones": [
            {"symbol": symbol, "confidence": round(0.8 - index * 0.2, 2)}
            for index, symbol in enumerate(candidates)
        ],
        "metadata": {
            "source_mode": "real",
            "real_backend": "posterior-json",
            "producer_backend_kind": "onnx-espeak",
            "producer_raw_model_output": {"raw_model_output_kind": "trained_onnx_ctc_model"},
        },
    }
