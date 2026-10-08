from __future__ import annotations

import json
from pathlib import Path

from tools.prepare_cantonese_singing_ctc_dataset import main


def _human(example_id: str) -> dict[str, object]:
    return {
        "example_id": example_id,
        "group_id": example_id,
        "singer_id": f"singer-{example_id}",
        "song_id": f"song-{example_id}",
        "phrase_id": f"phrase-{example_id}",
        "target_phone_tokens": ["a"],
        "internal_ipa_tokens": ["a"],
        "articulatory_features": [{"class": "fixture"}],
        "human_authored_annotation_evidence": True,
        "label_provenance": {
            "schema_version": "human_phone_label_v1",
            "kind": "human",
            "source_id": example_id,
            "source_revision": "r1",
            "annotation_sha256": "a" * 64,
            "policy_version": "fixture-v1",
            "annotator_id": "tester",
            "review_status": "approved",
        },
    }


def test_cli_combines_examples_and_refuses_overwrite(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "ctc_dataset.json"
    first.write_text(json.dumps({"examples": [_human("one")]}), encoding="utf-8")
    second.write_text(json.dumps(_human("two")) + "\n", encoding="utf-8")

    assert main([str(first), str(second), "--output", str(output), "--validation-ratio", "0.5"]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cantonese_singing_ctc_dataset_v1"
    assert {row["split"] for row in payload["examples"]} == {"train", "val"}
    assert main([str(first), str(second), "--output", str(output), "--validation-ratio", "0.5"]) == 2
    assert not list(tmp_path.glob("*.tmp"))
