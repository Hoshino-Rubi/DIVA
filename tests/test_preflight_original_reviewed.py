from __future__ import annotations

from pathlib import Path

from tools.preflight_original_reviewed import _discover_response


def test_discovery_ignores_non_response_json(tmp_path: Path) -> None:
    packet = tmp_path / "review-packet.json"
    packet.write_text("{}", encoding="utf-8")
    (tmp_path / "candidate-proposals.json").write_text('{"schema_version":"original_training_candidate_proposals_v2"}', encoding="utf-8")
    (tmp_path / "responses-template.json").write_text('{"schema_version":"original_segment_review_responses_v2"}', encoding="utf-8")
    assert _discover_response(packet, None) is None


def test_discovery_finds_exact_response_schema(tmp_path: Path) -> None:
    packet = tmp_path / "review-packet.json"
    packet.write_text("{}", encoding="utf-8")
    answer = tmp_path / "responses-from-user.json"
    answer.write_text('{"schema_version":"original_segment_review_responses_v2"}', encoding="utf-8")
    assert _discover_response(packet, None) == answer
