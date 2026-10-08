from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "report_miku_v4c_resolver_coverage.py"


def test_resolver_coverage_reports_unresolved_chars_and_patch_template(tmp_path: Path) -> None:
    output = tmp_path / "coverage.json"
    patch = tmp_path / "patch.json"

    completed = _run_tool(
        "--phrase",
        "\u559c\u6b61\u4f60",
        "--phrase",
        "\u4eca\u65e5",
        "--output",
        str(output),
        "--patch-output",
        str(patch),
        "--pretty",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "miku_v4c_resolver_coverage_report_v1"
    assert payload["phrase_count"] == 2
    assert payload["full_coverage_phrase_count"] == 1
    assert payload["zero_coverage_phrase_count"] == 1
    unresolved = {row["char"]: row for row in payload["unresolved_chars"]}
    assert set(unresolved) == {"\u4eca", "\u65e5"}
    assert unresolved["\u4eca"]["example_phrases"] == ["\u4eca\u65e5"]
    assert payload["lexicon_patch_template"]["entries"] == {"\u4eca": [], "\u65e5": []}
    assert json.loads(patch.read_text(encoding="utf-8"))["entries"] == {"\u4eca": [], "\u65e5": []}
    assert payload["training_data"] is False


def test_resolver_coverage_accepts_phrase_batch_json_and_audio_title_fallback(tmp_path: Path) -> None:
    phrase_batch = tmp_path / "phrases.json"
    phrase_batch.write_text(
        json.dumps(
            {
                "schema_version": "miku_v4c_phrase_batch_v1",
                "phrases": [
                    {"phrase_id": "covered", "lyrics": "\u6d77\u95ca\u5929\u7a7a"},
                    {"phrase_id": "missing", "lyrics": "\u4eca\u65e5"},
                ],
                "training_data": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "\u559c\u6b61\u59b3.flac").write_bytes(b"fake")
    (audio_dir / "Amani.flac").write_bytes(b"fake")
    output = tmp_path / "coverage.json"

    completed = _run_tool(
        str(phrase_batch),
        str(audio_dir),
        "--include-audio-titles",
        "--output",
        str(output),
        "--pretty",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    phrases = {row["lyrics"]: row for row in payload["phrases"]}
    assert set(phrases) == {"\u6d77\u95ca\u5929\u7a7a", "\u4eca\u65e5", "\u559c\u6b61\u59b3"}
    assert phrases["\u559c\u6b61\u59b3"]["source"] == "audio_title_fallback"
    assert phrases["\u6d77\u95ca\u5929\u7a7a"]["resolver_coverage"] == 1.0
    assert phrases["\u4eca\u65e5"]["resolver_coverage"] == 0.0


def test_resolver_coverage_can_use_explicit_lexicon_what_if(tmp_path: Path) -> None:
    lexicon = tmp_path / "lexicon.json"
    lexicon.write_text(
        json.dumps({"schema_version": "cantonese_v4c_lexicon_v1", "entries": {"\u4eca": ["gam1"], "\u65e5": ["jat6"]}}, ensure_ascii=False),
        encoding="utf-8",
    )
    output = tmp_path / "coverage.json"

    completed = _run_tool(
        "--phrase",
        "\u4eca\u65e5",
        "--lexicon",
        str(lexicon),
        "--output",
        str(output),
        "--pretty",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["lexicon"] == str(lexicon)
    assert payload["full_coverage_phrase_count"] == 1
    assert payload["unresolved_char_count"] == 0
    assert payload["phrases"][0]["resolver_coverage"] == 1.0


def _run_tool(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        env=env,
    )
