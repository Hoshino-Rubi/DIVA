from __future__ import annotations

import json
from pathlib import Path

from tools.build_automatic_original_alignment import main


def test_cli_writes_machine_manifest_and_exception_queue(tmp_path: Path, monkeypatch) -> None:
    reference = tmp_path / "reference.json"
    proposals = tmp_path / "proposals.json"
    correspondence_dir = tmp_path / "correspondence"
    output = Path(".state/runs") / "automatic-alignment-cli-test"
    if output.exists():
        import shutil
        shutil.rmtree(output)
    correspondence_dir.mkdir()
    reference.write_text(json.dumps({"songs": [{
        "song_id": "song-a", "split": "train",
        "original_audio": {"path": "original.wav", "sha256": "a" * 64},
        "human_vsqx": {"sha256": "b" * 64},
        "notes": [{"note_id": "song-a-n1", "mapping_status": "resolved_proxy", "model_target_ipa": ["a"]}],
    }]}, ensure_ascii=False), encoding="utf-8")
    proposals.write_text(json.dumps({"candidates": [{
        "candidate_id": "candidate-1", "example_id": "song-a-original-01", "song_id": "song-a", "split": "train",
        "start_sec": 1.0, "end_sec": 3.0,
        "candidate_references": [{"factor": 4, "rendered_start_sec": 4.0, "rendered_end_sec": 6.0, "note_ids": ["song-a-n1"], "mapping_complete": True, "valid_rendered_range": True}],
    }]}, ensure_ascii=False), encoding="utf-8")
    (correspondence_dir / "song-a.json").write_text(json.dumps({
        "song_id": "song-a", "split": "train",
        "clips": [{"example_id": "song-a-original-01", "resolution_consistent": True, "resolution_disagreement_sec": 0.0}],
        "probes": [{"validation": [{"example_id": "song-a-original-01", "local_best": {"cosine_similarity": 0.8}}]}],
    }), encoding="utf-8")
    try:
        monkeypatch.setattr("sys.argv", ["build_automatic_original_alignment.py", "--reference", str(reference), "--candidate-proposals", str(proposals), "--correspondence-dir", str(correspondence_dir), "--output-dir", str(output)])
        assert main() == 0
        result = json.loads((output / "machine-aligned-reference.json").read_text(encoding="utf-8"))
        status = json.loads((output / "STATUS.json").read_text(encoding="utf-8"))
        assert result["counts"]["machine_aligned"] == 1
        assert status["model_jobs_started"] is False
        assert (output / "exception-review-queue.json").exists()
    finally:
        import shutil
        if output.exists():
            shutil.rmtree(output)
