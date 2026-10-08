from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree as ET

from any2xsampa.vocaloid.vsqx import MIKU_V4C_ENGINE_NAME, VSQ4_NS


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "import_miku_v4c_annotated_vsqx.py"


def test_imports_ordered_lyrics_and_preserves_raw_vsqx_phonemes(tmp_path: Path) -> None:
    vsqx = tmp_path / "annotated.vsqx"
    _write_vsqx(vsqx, [("wang", "uAN", True), ("a", "t @U \u014b", False)])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "miku_v4c_annotated_vsqx_manifest_v1",
                "corpus_id": "fixture",
                "expected_voice": MIKU_V4C_ENGINE_NAME,
                "segments": [
                    {
                        "segment_id": "fixture-p1",
                        "order": 1,
                        "vsqx": vsqx.name,
                        "lyrics": ["\u7532", "\u4e59"],
                        "variant_group": "fixture-v1",
                        "annotation_status": "user_supplied_human_authored_unscored",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.json"
    character_map = tmp_path / "character_map.json"

    completed = _run_tool(manifest, corpus, character_map)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(corpus.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["segment_count"] == 1
    assert payload["note_count"] == 2
    segment = payload["segments"][0]
    assert segment["lyrics"] == "\u7532\u4e59"
    assert segment["voice_name"] == MIKU_V4C_ENGINE_NAME
    assert segment["track_index"] == 1
    assert segment["alignment_mode"] == "one_note_per_character"
    assert segment["notes"][0]["lyric_character"] == "\u7532"
    assert segment["notes"][0]["vsqx_lyric"] == "wang"
    assert segment["notes"][0]["phoneme_text"] == "uAN"
    assert segment["notes"][0]["phoneme_lock"] is True
    assert segment["notes"][0]["absolute_start_tick"] == 3840
    assert segment["notes"][1]["phoneme_tokens"] == ["t", "@U", "\u014b"]
    assert segment["training_data"] is False

    mapped = json.loads(character_map.read_text(encoding="utf-8"))
    by_char = {row["character"]: row for row in mapped["characters"]}
    assert by_char["\u7532"]["phoneme_sequence_counts"] == {"uAN": 1}
    assert by_char["\u4e59"]["observations"][0]["phoneme_tokens"] == ["t", "@U", "\u014b"]
    assert mapped["training_data"] is False


def test_rejects_lyric_note_count_mismatch_but_writes_auditable_report(tmp_path: Path) -> None:
    vsqx = tmp_path / "annotated.vsqx"
    _write_vsqx(vsqx, [("a", "a", True), ("b", "p a", True)])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "miku_v4c_annotated_vsqx_manifest_v1",
                "segments": [
                    {
                        "segment_id": "bad-count",
                        "vsqx": str(vsqx),
                        "lyrics": "\u7532\u4e59\u4e19",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.json"
    character_map = tmp_path / "character_map.json"

    completed = _run_tool(manifest, corpus, character_map)

    assert completed.returncode == 1
    payload = json.loads(corpus.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["segment_count"] == 0
    assert "lyric/note count mismatch" in payload["errors"][0]
    assert json.loads(character_map.read_text(encoding="utf-8"))["observation_count"] == 0


def test_supports_explicit_repeated_character_alignment(tmp_path: Path) -> None:
    vsqx = tmp_path / "split.vsqx"
    _write_vsqx(vsqx, [("a", "ts", True), ("a", "i", True), ("b", "n", True)])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "segment_id": "split",
                        "vsqx": str(vsqx),
                        "lyrics": "\u7532\u4e59",
                        "note_character_indexes": [1, 1, 2],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.json"
    character_map = tmp_path / "character_map.json"

    completed = _run_tool(manifest, corpus, character_map)

    assert completed.returncode == 0, completed.stderr
    segment = json.loads(corpus.read_text(encoding="utf-8"))["segments"][0]
    assert segment["alignment_mode"] == "explicit_note_character_indexes"
    assert segment["note_character_indexes"] == [1, 1, 2]
    assert [row["lyric_character"] for row in segment["notes"]] == ["\u7532", "\u7532", "\u4e59"]
    by_char = {
        row["character"]: row
        for row in json.loads(character_map.read_text(encoding="utf-8"))["characters"]
    }
    assert by_char["\u7532"]["observation_count"] == 2


def _write_vsqx(path: Path, rows: list[tuple[str, str, bool]]) -> None:
    q = lambda tag: f"{{{VSQ4_NS}}}{tag}"
    root = ET.Element(q("vsq4"))
    table = ET.SubElement(root, q("vVoiceTable"))
    voice = ET.SubElement(table, q("vVoice"))
    ET.SubElement(voice, q("id")).text = "fixture-voice"
    ET.SubElement(voice, q("name")).text = MIKU_V4C_ENGINE_NAME
    master = ET.SubElement(root, q("masterTrack"))
    ET.SubElement(master, q("resolution")).text = "480"
    tempo = ET.SubElement(master, q("tempo"))
    ET.SubElement(tempo, q("v")).text = "12000"
    ET.SubElement(root, q("vsTrack"))
    track = ET.SubElement(root, q("vsTrack"))
    part = ET.SubElement(track, q("vsPart"))
    ET.SubElement(part, q("t")).text = "3840"
    for index, (lyric, phonemes, locked) in enumerate(rows):
        note = ET.SubElement(part, q("note"))
        ET.SubElement(note, q("t")).text = str(index * 480)
        ET.SubElement(note, q("dur")).text = "480"
        ET.SubElement(note, q("n")).text = str(64 + index)
        ET.SubElement(note, q("v")).text = "64"
        ET.SubElement(note, q("y")).text = lyric
        phoneme = ET.SubElement(note, q("p"), {"lock": "1"} if locked else {})
        phoneme.text = phonemes
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _run_tool(manifest: Path, corpus: Path, character_map: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return subprocess.run(
        [
            sys.executable,
            str(TOOL),
            str(manifest),
            "--output",
            str(corpus),
            "--character-map-output",
            str(character_map),
            "--pretty",
        ],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        env=env,
    )
