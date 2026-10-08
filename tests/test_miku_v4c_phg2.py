from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import tools.probe_miku_v4c_phonemes as probe_module
from any2xsampa.vocaloid.phg2 import (
    PHG2ParseError,
    PHG2_SCHEMA_VERSION,
    parse_phg2_bytes,
    parse_phg2_path,
)
from any2xsampa.vocaloid.vsqx import export_vsqx_from_template as real_export_vsqx_from_template


REPO_ROOT = Path(__file__).resolve().parents[1]




def _synthetic_binary_ddi(groups: list[tuple[str, list[str]]], *, prefix: bytes = b"DDI\x00public-header") -> bytes:
    body = len(groups).to_bytes(4, byteorder="little")
    token_id = 1
    for group_name, tokens in groups:
        name_bytes = group_name.encode("ascii")
        body += len(name_bytes).to_bytes(4, byteorder="little") + name_bytes
        body += len(tokens).to_bytes(4, byteorder="little")
        body += (token_id + 100).to_bytes(4, byteorder="little")
        for token in tokens:
            token_bytes = token.encode("ascii")
            body += len(token_bytes).to_bytes(4, byteorder="little") + token_bytes
            body += token_id.to_bytes(4, byteorder="little")
            token_id += 1
    return prefix + b"PHG2" + len(body).to_bytes(4, byteorder="little") + body + b"license=DO_NOT_LEAK"


def _synthetic_ddi(payload: dict[str, object], *, prefix: bytes = b"DDI\x00public-header") -> bytes:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return prefix + b"PHG2" + len(body).to_bytes(4, byteorder="little") + body + b"license=DO_NOT_LEAK"


def test_parse_phg2_bytes_returns_public_token_schema_without_sensitive_strings() -> None:
    ddi = _synthetic_ddi(
        {
            "schema_version": "synthetic",
            "license": "SHOULD_NOT_LEAK",
            "serial": "SN-1234",
            "activation": "ACT-5678",
            "groups": [
                {"name": "initials", "tokens": ["x", "f", "n"]},
                {"name": "finals", "tokens": ["ei", "u", "aI", "@N", "i\\", "i`"]},
            ],
        }
    )

    parsed = parse_phg2_bytes(ddi)
    rendered = json.dumps(parsed, ensure_ascii=False, sort_keys=True)

    assert parsed["schema_version"] == PHG2_SCHEMA_VERSION
    assert parsed["groups"] == [
        {"name": "initials", "tokens": ["x", "f", "n"], "token_count": 3},
        {"name": "finals", "tokens": ["ei", "u", "aI", "@N", "i\\", "i`"], "token_count": 6},
    ]
    assert parsed["tokens"] == ["x", "f", "n", "ei", "u", "aI", "@N", "i\\", "i`"]
    assert "SHOULD_NOT_LEAK" not in rendered
    assert "SN-1234" not in rendered
    assert "ACT-5678" not in rendered
    assert "license=DO_NOT_LEAK" not in rendered


def test_parse_phg2_path_accepts_synthetic_ddi_file(tmp_path: Path) -> None:
    ddi_path = tmp_path / "MIKU_V4C.ddi"
    ddi_path.write_bytes(
        _synthetic_ddi(
            {
                "groups": {
                    "onsets": ["p", "t", "k"],
                    "finals": ["a", "ei", "u@N"],
                },
            }
        )
    )

    parsed = parse_phg2_path(ddi_path)

    assert parsed["section"]["magic"] == "PHG2"
    assert parsed["tokens"] == ["p", "t", "k", "a", "ei", "u@N"]


def test_probe_draft_uses_short_multi_token_labels_not_isolated_characters() -> None:
    phg2_payload = parse_phg2_bytes(
        _synthetic_ddi(
            {
                "groups": [
                    {"name": "smoke", "tokens": ["x", "ei", "f", "u", "n", "l", "aI"]},
                ],
            }
        )
    )

    draft = probe_module.build_probe_draft_payload(phg2_payload, tokens_per_note=3)
    notes = draft["tracks"][0]["notes"]

    assert [note["phonemes"] for note in notes] == [["x", "ei", "f"], ["u", "n", "l", "aI"]]
    assert all(len(note["phonemes"]) >= 2 for note in notes)
    assert all(len(note["lyric"]) > 1 for note in notes)
    assert all(note["metadata"]["probe_kind"] == "short_phrase_or_label" for note in notes)
    assert draft["provenance"]["xsampa_role"] == "export_only_not_internal_truth"


def test_probe_cli_writes_tokens_and_draft_from_synthetic_ddi(tmp_path: Path) -> None:
    ddi_path = tmp_path / "MIKU_V4C.ddi"
    tokens_path = tmp_path / "tokens.json"
    draft_path = tmp_path / "draft.json"
    ddi_path.write_bytes(_synthetic_ddi({"groups": {"smoke": ["x", "ei", "f", "u", "n", "aI"]}}))

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [
            sys.executable,
            "tools/probe_miku_v4c_phonemes.py",
            "--ddi",
            str(ddi_path),
            "--output-tokens-json",
            str(tokens_path),
            "--output-draft",
            str(draft_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    tokens_payload = json.loads(tokens_path.read_text(encoding="utf-8"))
    draft_payload = json.loads(draft_path.read_text(encoding="utf-8"))

    assert summary["tokens"] == 6
    assert summary["outputs"]["tokens_json"] == str(tokens_path)
    assert tokens_payload["tokens"] == ["x", "ei", "f", "u", "n", "aI"]
    assert draft_payload["schema_version"] == "vocal_track_draft_v1"
    assert all(len(note["phonemes"]) > 1 for note in draft_payload["tracks"][0]["notes"])


def test_output_vsqx_uses_template_exporter(monkeypatch, tmp_path: Path) -> None:
    ddi_path = tmp_path / "MIKU_V4C.ddi"
    template_path = tmp_path / "template.vsqx"
    output_path = tmp_path / "probe.vsqx"
    ddi_path.write_bytes(_synthetic_ddi({"groups": {"smoke": ["x", "ei", "f", "u", "n", "aI"]}}))
    template_path.write_text("<vsq4 />", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_export(draft, output, *, template_path):
        calls.append(
            {
                "note_count": sum(len(track.notes) for track in draft.tracks),
                "output": Path(output),
                "template_path": Path(template_path),
            }
        )
        Path(output).write_text("fake-vsqx", encoding="utf-8")
        return Path(output)

    monkeypatch.setattr(probe_module, "export_vsqx_from_template", fake_export)

    rc = probe_module.main(
        [
            "--ddi",
            str(ddi_path),
            "--output-vsqx",
            str(output_path),
            "--template",
            str(template_path),
        ]
    )

    assert rc == 0
    assert output_path.read_text(encoding="utf-8") == "fake-vsqx"
    assert calls == [{"note_count": 2, "output": output_path, "template_path": template_path}]
    assert probe_module.export_vsqx_from_template is not real_export_vsqx_from_template


def test_parse_phg2_bytes_understands_observed_binary_table_layout() -> None:
    ddi = _synthetic_binary_ddi(
        [
            ("Vowels", ["a", "o", "7", "aI", "ei", "@`", "@N", "i\\", "i`"]),
            ("UnvoicedFricatives", ["s\\", "f", "x", "s", "s`"]),
        ]
    )

    parsed = parse_phg2_bytes(ddi)

    assert parsed["groups"] == [
        {
            "name": "Vowels",
            "tokens": ["a", "o", "7", "aI", "ei", "@`", "@N", "i\\", "i`"],
            "token_count": 9,
        },
        {
            "name": "UnvoicedFricatives",
            "tokens": ["s\\", "f", "x", "s", "s`"],
            "token_count": 5,
        },
    ]
    assert parsed["tokens"] == ["a", "o", "7", "aI", "ei", "@`", "@N", "i\\", "i`", "s\\", "f", "x", "s", "s`"]
    assert parsed["section"]["declared_length"] is not None


def test_parse_phg2_json_rejects_unreasonable_group_and_section_sizes(monkeypatch) -> None:
    import any2xsampa.vocaloid.phg2 as phg2

    monkeypatch.setattr(phg2, "_MAX_GROUPS", 2)
    with pytest.raises(PHG2ParseError, match="group count"):
        parse_phg2_bytes(
            _synthetic_ddi({"groups": [{"tokens": ["a"]}] * 3})
        )

    marker = b"PHG2" + (phg2._MAX_SECTION_BYTES + 1).to_bytes(4, "little")
    with pytest.raises(PHG2ParseError, match="section is too large"):
        parse_phg2_bytes(marker)


def test_parse_phg2_json_rejects_unreasonable_token_counts(monkeypatch) -> None:
    import any2xsampa.vocaloid.phg2 as phg2

    monkeypatch.setattr(phg2, "_MAX_TOKENS_PER_GROUP", 2)
    with pytest.raises(PHG2ParseError, match="token count"):
        parse_phg2_bytes(_synthetic_ddi({"groups": [{"tokens": ["a", "b", "c"]}]}))



def test_probe_draft_can_exclude_known_silent_tokens() -> None:
    phg2_payload = parse_phg2_bytes(
        _synthetic_ddi(
            {
                "groups": [
                    {"name": "smoke", "tokens": ["x", "ei", "i\\", "i`", "UN", "f", "u", "n"]},
                ],
            }
        )
    )

    draft = probe_module.build_probe_draft_payload(
        phg2_payload,
        tokens_per_note=3,
        exclude_tokens={"i\\", "i`", "UN"},
    )
    flat_tokens = [token for note in draft["tracks"][0]["notes"] for token in note["phonemes"]]

    assert "i\\" not in flat_tokens
    assert "i`" not in flat_tokens
    assert "UN" not in flat_tokens
    assert draft["provenance"]["excluded_tokens"] == ["UN", "i\\", "i`"]
