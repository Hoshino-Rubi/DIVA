"""Build machine-aligned original-song proxy references and an exception queue."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.automatic_original_alignment import build_edge_review_queue, build_machine_alignment  # noqa: E402
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, write_json_new  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Transfer VSQX phones through audio-only original/rendered correspondence.")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate-proposals", required=True, type=Path)
    parser.add_argument("--correspondence-dir", required=True, type=Path)
    parser.add_argument("--shalong-packet", type=Path, help="optional B01-B06 packet for machine edge suggestions")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / ".state/runs") or output.exists():
        parser.error("output must be a new directory inside .state/runs")
    reference = read_json(args.reference)
    proposal = read_json(args.candidate_proposals)
    candidates = proposal.get("candidates")
    if not isinstance(candidates, list) or len(candidates) > 20:
        raise ValueError("candidate proposals must contain at most twenty candidates")
    correspondences: dict[str, dict[str, Any]] = {}
    correspondence_bindings: dict[str, Any] = {}
    for song in sorted({row.get("song_id") for row in candidates}):
        if not isinstance(song, str) or not song:
            raise ValueError("candidate has invalid song_id")
        path = args.correspondence_dir / f"{song}.json"
        data = read_json(path)
        if data.get("song_id") != song or data.get("split") == "test":
            raise ValueError("correspondence song/split mismatch or sealed test input")
        correspondences[song] = data
        correspondence_bindings[song] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    bindings = {
        "reference": {"path": str(args.reference.resolve()), "sha256": sha256_file(args.reference)},
        "candidate_proposals": {"path": str(args.candidate_proposals.resolve()), "sha256": sha256_file(args.candidate_proposals)},
        "correspondence": correspondence_bindings,
    }
    result = build_machine_alignment(reference=reference, candidates=candidates,
                                     correspondences=correspondences,
                                     source_bindings=bindings)
    output.mkdir(parents=True)
    digest = write_json_new(output / "machine-aligned-reference.json", result)
    exceptions = [row for row in result["examples"] if row["status"] == "needs_review"]
    exceptions.extend(row for row in result["errors"] if row["status"] == "needs_review")
    queue = {
        "schema_version": "original_alignment_exception_queue_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": digest,
        "items": exceptions,
        "human_review_required": bool(exceptions),
        "official_evaluation_eligible": False,
    }
    write_json_new(output / "exception-review-queue.json", queue)
    edge_status = None
    if args.shalong_packet:
        shalong_packet = read_json(args.shalong_packet)
        shalong_corr_path = args.correspondence_dir / "shalong.json"
        edge_status = build_edge_review_queue(
            packet=shalong_packet,
            reference=reference,
            correspondence={"shalong": read_json(shalong_corr_path)},
            source_bindings={
                "reference": {"path": str(args.reference.resolve()), "sha256": sha256_file(args.reference)},
                "shalong_packet": {"path": str(args.shalong_packet.resolve()), "sha256": sha256_file(args.shalong_packet)},
                "shalong_correspondence": {"path": str(shalong_corr_path.resolve()), "sha256": sha256_file(shalong_corr_path)},
            },
        )
        write_json_new(output / "shalong-machine-edge-suggestions.json", edge_status)
    status = {
        "schema_version": "original_alignment_run_status_v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine_alignment_sha256": digest,
        "counts": result["counts"],
        "training_proxy_rows": result["training"]["machine_proxy_rows"],
        "exception_queue": str((output / "exception-review-queue.json").resolve()),
        "shalong_edge_suggestions": edge_status["counts"] if edge_status else None,
        "model_jobs_started": False,
        "test_predictions_opened": False,
        "source_wav_vsqx_modified": False,
        "default_model_replaced": False,
    }
    write_json_new(output / "STATUS.json", status)
    edge_line = ""
    if edge_status:
        edge_line = (f"- 《沙龙》边缘建议：{edge_status['counts']['machine_suggested']} 项机器建议，"
                     f"{edge_status['counts']['needs_review']} 项阻塞\n")
    report = (
        "# 自动原唱对齐结果\n\n"
        "本产物是 `machine_aligned_proxy`，不是人工原唱真值。高置信片段可用于机器代理训练；正式原唱评测仍要求人工确认。\n\n"
        f"- 总候选：{result['counts']['total']}\n"
        f"- 自动通过：{result['counts']['machine_aligned']}\n"
        f"- 需要复核：{result['counts']['needs_review']}\n"
        f"- 拒绝：{result['counts']['rejected']}\n"
        + edge_line
        + f"- machine-aligned-reference.json SHA-256：`{digest}`\n"
    )
    (output / "RESULTS.md").write_text(report, encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
