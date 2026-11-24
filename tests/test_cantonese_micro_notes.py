from dataclasses import asdict
import json
from xml.etree import ElementTree as ET

from any2xsampa.local.generated_vsqx import export_generated_score
from any2xsampa.local.vsqx_export import POLICY, make_vsqx_splice_plan
from any2xsampa.vocaloid.vsqx import VSQ4_NS


def test_v3_keeps_legacy_coda_plan_on_parent_without_pitch_bend(tmp_path):
    score_path = tmp_path / "score.json"
    output = tmp_path / "generated.vsqx"
    row = {
        "start_sec": 0.125,
        "duration_sec": 0.5,
        "pitch": 69,
        "lyric": "/pʰ a t̚/",
        "v4c_phonemes": ["p_h", "a", "t"],
    }
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    plan = make_vsqx_splice_plan([
        {"normalized_ipa": "pʰ"},
        {"normalized_ipa": "a"},
        {"normalized_ipa": "t̚", "subphone_boundaries": {"coda_closure_ms": 500}},
    ], policy)
    assert plan is not None
    row["splice_plan"] = asdict(plan)
    score_path.write_text(json.dumps({
        "schema_version": "local_note_score_v1",
        "source": "test",
        "tempo": 120,
        "ticks_per_quarter": 480,
        "notes": [row],
        "f0": [{"time_sec": 0.25, "hz": 449.0, "voiced": True}],
        "placeholder_note_count": 0,
        "unassigned_phone_count": 0,
    }), encoding="utf-8")

    assert export_generated_score(score_path, output)["splice_count"] == 0

    ns = {"v": VSQ4_NS}
    root = ET.parse(output)
    notes = root.findall(".//v:note", ns)
    assert [n.findtext("v:p", namespaces=ns) for n in notes] == ["p_h a t"]
    assert int(notes[0].findtext("v:t", namespaces=ns)) == 120
    assert int(notes[0].findtext("v:dur", namespaces=ns)) == 480
    assert int(notes[0].findtext("v:v", namespaces=ns)) == 64
    assert notes[0].findtext("v:nStyle/v:v[@id='opening']", namespaces=ns) == "115"
    controls = root.findall(".//v:cc/v:v", ns)
    assert [(v.get("id"), v.text) for v in controls] == [("S", "12")]
