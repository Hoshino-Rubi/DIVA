from __future__ import annotations

from copy import deepcopy
from html.parser import HTMLParser
import json
from pathlib import Path
import re

import pytest

from any2xsampa.gui.alignment_audit import (
    audit_row, audit_state, bind_responses, checked_object, copy_previews, write_page,
)
from any2xsampa.training.limited_data_artifacts import sha256_object, sha256_file


@pytest.mark.parametrize("machine", ["machine_aligned", "needs_review", "machine_suggested"])
def test_human_rejection_is_never_promoted(machine):
    state = audit_state({"status": "rejected"}, {"status": machine})
    assert state["status"] == "human_rejected"
    assert state["human_machine_conflict"] == (machine == "machine_aligned")
    assert state["training_eligible"] is False
    assert state["formal_reference_validated"] is False


def test_text_approval_remains_separate_from_phone_truth():
    human = {"status": "human_confirmed", "phones": [], "review_statement": "全部通过"}
    assert audit_state(human, {"status": "machine_aligned"})["text_approval_only"]
    assert not audit_state(human, {})["formal_reference_validated"]


def test_packet_hash_and_response_identity(tmp_path):
    packet = {"items": [{"review_id": "C01"}]}
    packet["packet_sha256"] = sha256_object(packet)
    path = tmp_path / "packet.json"
    path.write_text(json.dumps(packet), encoding="utf-8")
    assert checked_object(path, "packet_sha256") == packet
    response = {"schema_version": "original_segment_review_responses_v2", "packet_sha256": packet["packet_sha256"],
                "responses": [{"review_id": "C01", "status": "rejected", "review_statement": "错位"}]}
    answer = tmp_path / "response.json"
    answer.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")
    assert bind_responses(packet, answer)["C01"]["review_statement"] == "错位"
    response["packet_sha256"] = "f" * 64
    answer.write_text(json.dumps(response), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        bind_responses(packet, answer)
    packet["items"][0]["review_id"] = "C02"
    path.write_text(json.dumps(packet), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        checked_object(path, "packet_sha256")


def test_previews_are_copied_and_verified_without_touching_source(tmp_path):
    source = tmp_path / "packet"
    (source / "media").mkdir(parents=True)
    item = {"review_id": "C01", "previews": [{"role": "original", "duration_sec": 10}]}
    for field, name, key in (("audio_file", "audio.wav", "audio_sha256"), ("waveform_file", "wave.png", "waveform_sha256")):
        p = source / "media" / name
        p.write_bytes(b"unchanged preview fixture")
        item["previews"][0].update({field: name, key: sha256_file(p)})
    original = deepcopy(item)
    result = copy_previews(source, item, tmp_path / "gui")
    assert (tmp_path / "gui" / result[0]["audio_file"]).read_bytes() == b"unchanged preview fixture"
    assert item == original
    (source / "media/audio.wav").write_bytes(b"different")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        copy_previews(source, item, tmp_path / "other")


def test_media_path_cannot_escape_packet(tmp_path):
    item = {"review_id": "C01", "previews": [{"role": "original", "audio_file": "../outside.wav"}]}
    with pytest.raises(ValueError, match="media filename"):
        copy_previews(tmp_path, item, tmp_path / "out")


def test_notes_are_display_only_and_existing_human_text_is_preserved():
    song = {"song_id": "like-you", "split": "train", "original_audio": {"sha256": "a"*64},
            "human_vsqx": {"sha256": "b"*64}, "notes": [{"note_id": "n1", "raw_lyric": "笑",
            "mapping_status": "resolved_proxy", "model_target_ipa": ["s", "i", "u"], "raw_v4c_tokens": ["s", "iU"]}]}
    item = {"review_id": "C05", "song_id": "like-you", "split": "train", "candidate_references": [{"note_ids": ["n1"]}]}
    human = {"status": "human_confirmed", "phones": [], "review_statement": "笑 在原曲音频被吃掉了"}
    before = deepcopy((item, song, human))
    row = audit_row(item, {"packet_sha256": "c"*64}, human, {}, song, [])
    assert row["notes"][0]["lyric"] == "笑"
    assert row["human"] == human
    assert not row["state"]["training_eligible"]
    assert before == (item, song, human)
    song["split"] = "test"
    with pytest.raises(ValueError, match="sealed"):
        audit_row(item, {}, human, {}, song, [])


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def test_generated_page_is_offline_and_embedded_data_cannot_inject_html(tmp_path):
    assets = Path(__file__).resolve().parents[1] / "tools/alignment_audit_gui"
    bundle = {"bundle_id": "a"*64, "rows": [], "malicious_fixture": '</script><img src="remote">音素 ŋ'}
    write_page(bundle, tmp_path, assets)
    text = (tmp_path / "index.html").read_text(encoding="utf-8")
    parser = PageParser()
    parser.feed(text)
    assert not any(tag == "img" and attrs.get("src") == "remote" for tag, attrs in parser.tags)
    assert not any(attrs.get("src", "").startswith("http") for _, attrs in parser.tags)
    scripts = [attrs for tag, attrs in parser.tags if tag == "script"]
    assert len(scripts) == 3 and all("src" not in s for s in scripts)
    assert "__AUDIT_" not in text
    assert "connect-src 'none'" in text
    raw = re.search(r'<script id="audit-data" type="application/json">(.*?)</script>', text, re.S).group(1)
    assert json.loads(raw) == bundle
    # No missing literal form IDs between the HTML and the rendering code.
    ids = set(re.findall(r'id="([a-z][a-z-]+)"', text))
    used = set(re.findall(r'\$\("([a-z][a-z-]+)"\)', text))
    assert used <= ids
