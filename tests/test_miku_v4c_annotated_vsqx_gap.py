from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "report_miku_v4c_annotated_vsqx_gap.py"


def test_reports_exact_and_missing_manual_sequences_by_character_slot(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "segment_id": "fixture",
                        "variant_group": "fixture",
                        "notes": [
                            {
                                "note_index": 1,
                                "character_index": 1,
                                "lyric_character": "\u7532",
                                "phoneme_text": "x ei",
                                "phoneme_tokens": ["x", "ei"],
                                "pitch": 64,
                            },
                            {
                                "note_index": 2,
                                "character_index": 2,
                                "lyric_character": "\u4e59",
                                "phoneme_text": "z z",
                                "phoneme_tokens": ["z", "z"],
                                "pitch": 65,
                            },
                        ],
                    }
                ],
                "model_output": False,
                "training_data": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    lexicon = tmp_path / "lexicon.json"
    lexicon.write_text(
        json.dumps({"entries": {"\u7532": ["hei2"], "\u4e59": ["nei5"]}}, ensure_ascii=False),
        encoding="utf-8",
    )
    output = tmp_path / "gap.json"

    completed = _run_tool(corpus, lexicon, output)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["slot_count"] == 2
    assert payload["resolved_slot_count"] == 2
    assert payload["manual_exact_lattice_match_count"] == 1
    assert payload["manual_sequence_not_in_lattice_count"] == 1
    assert payload["rows"][0]["manual_top_candidate_match"] is True
    assert payload["rows"][1]["manual_exact_lattice_match"] is False
    assert payload["training_data"] is False


def test_concatenates_repeated_notes_before_resolver_comparison(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "segment_id": "split",
                        "notes": [
                            {
                                "note_index": 1,
                                "character_index": 1,
                                "lyric_character": "\u7532",
                                "phoneme_text": "x",
                                "phoneme_tokens": ["x"],
                            },
                            {
                                "note_index": 2,
                                "character_index": 1,
                                "lyric_character": "\u7532",
                                "phoneme_text": "ei",
                                "phoneme_tokens": ["ei"],
                            },
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    lexicon = tmp_path / "lexicon.json"
    lexicon.write_text(json.dumps({"entries": {"\u7532": ["hei2"]}}, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "gap.json"

    completed = _run_tool(corpus, lexicon, output)

    assert completed.returncode == 0, completed.stderr
    row = json.loads(output.read_text(encoding="utf-8"))["rows"][0]
    assert row["note_count"] == 2
    assert row["note_indexes"] == [1, 2]
    assert row["manual_phoneme_sequence"] == "x ei"
    assert row["manual_top_candidate_match"] is True


def _run_tool(corpus: Path, lexicon: Path, output: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return subprocess.run(
        [
            sys.executable,
            str(TOOL),
            str(corpus),
            "--lexicon",
            str(lexicon),
            "--output",
            str(output),
            "--pretty",
        ],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        env=env,
    )
