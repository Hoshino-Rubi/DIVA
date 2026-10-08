"""Build one conservative replacement review window after a rejected candidate."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from any2xsampa.training.limited_data_artifacts import (
    read_json,
    sha256_file,
    sha256_object,
    verify_file,
    write_json_new,
)
from any2xsampa.training.replacement_candidates import overlaps, select_candidates
from tools.batch_training_review import render_page, verify_sources_and_media
from tools.build_reviewed_batch_followup import choice_for, proposals
from tools.prepare_original_training_candidates import _preview
from tools.reconcile_replacement_edges import load_verified_packet


BATCH = ROOT / ".state/runs/batch-training-review-20260918T001300"
CUMULATIVE = ROOT / ".state/runs/reviewed-batch-followup-features-20260918T141000/CUMULATIVE_DATA.json"
PROGRESS = ROOT / ".state/runs/reviewed-training-progress-20260917T230000"
HISTORY = ROOT / ".state/runs/training-replacements-20260917T211100/HISTORY.json"
N06_PACKET = ROOT / ".state/runs/final-training-window-review-20260918T150000/packet.json"


def build(output: Path) -> dict:
    raise RuntimeError(
        'Retired: this builder repeats rejected correspondence hypotheses. '
        'Use tools/diagnose_reviewed_correspondence.py to validate localization; '
        'do not generate another review from the old saved-path pool.'
    )
    if output.exists() or not output.is_relative_to(ROOT / ".state/runs"):
        raise ValueError("new run required")
    packet = read_json(BATCH / "packet.json")
    verified = verify_sources_and_media(BATCH, packet)
    cumulative = read_json(CUMULATIVE)
    if cumulative["manifest_hash"] != sha256_object({k: v for k, v in cumulative.items() if k != "manifest_hash"}):
        raise ValueError("cumulative hash mismatch")
    history = read_json(HISTORY)
    contexts_doc = read_json(Path(packet["source_bindings"]["contexts"]["path"]))
    audit = read_json(PROGRESS / "MULTIPATH_AUDIT.json")
    policy = deepcopy(packet["policy"])
    reference = verified["songs"]
    protected = history["retained"] + read_json(Path(packet["source_bindings"]["location_review"]["path"]))["items"]
    protected += cumulative["rows"] + packet["items"]
    n06 = read_json(N06_PACKET)["items"][0]
    protected.append({"song_id": n06["song_id"], "start_sec": n06["start_sec"], "end_sec": n06["end_sec"]})
    pool, excluded = [], []
    for context in contexts_doc["contexts"]:
        if context["split"] != "train":
            continue
        alignment = read_json(Path(packet["source_bindings"][context["context_id"] + "_saved_paths"]["path"]))
        paths = {"mixed": alignment["paths"]["mixed"]["candidates"][0],
                 "vocal": alignment["paths"]["vocal"]["candidates"][0]}
        # Rebuild the same bounded alternatives used by the previous review builder.
        values, failures = proposals(
            context,
            reference[context["song_id"]],
            alignment,
            protected,
            history["rejected"],
            audit["anchors"],
            policy,
        )
        pool.extend(values)
        excluded.extend(failures)
    if not pool:
        raise ValueError("no replacement candidates")
    selected = select_candidates(pool, cumulative["rows"], contexts_doc["split"]["train"], 1)
    if len(selected) != 1:
        raise ValueError("one replacement candidate required")
    row = deepcopy(selected[0])
    row["review_id"] = "N07"
    row["kind"] = "replacement"
    row["correspondence_preconfirmed"] = False
    row["previous_audit"] = None
    row["review_focus"] = "N06 不对应后，审查同一上下文中另一段不重叠窗口；不得沿用 N06 的对应结论。"
    song = reference[row["song_id"]]
    row.update(original_audio=deepcopy(song["original_audio"]), source_vsqx=deepcopy(song["human_vsqx"]), training_eligible=False)
    output.mkdir(parents=True)
    media = output / "media"
    media.mkdir()
    row["original"] = _preview(song["original_audio"], row, media, "N07-original", "original")
    row["edge_media"] = {
        side: _preview(song["original_audio"], {"start_sec": row[side + "_sec"] - 1.25,
                                                "end_sec": row[side + "_sec"] + 1.25}, media,
                      "N07-" + side, "original")
        for side in ("start", "end")
    }
    for choice in row["alternatives"]:
        choice["preview"] = _preview(song["rendered_audio"], choice, media, "N07-" + choice["choice_id"], "rendered")
    sources = deepcopy(packet["source_bindings"])
    for name, path in [("prior_batch", BATCH / "packet.json"),
                       ("current_cumulative", CUMULATIVE),
                       ("rejected_N06_packet", N06_PACKET),
                       ("builder", Path(__file__))]:
        sources[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    policy_out = {
        "window_sec": 10,
        "active_slots_before": cumulative["valid_windows"],
        "active_slots_after_if_confirmed": cumulative["valid_windows"] + 1,
        "previous_rejected_windows_protected": True,
        "selection_uses_phone_predictions": False,
        "training_started": False,
        "test_songs_opened": False,
        "candidate_count": len(pool),
        "excluded_count": len(excluded),
    }
    write_json_new(output / "FROZEN_POLICY.json", policy_out)
    write_json_new(output / "PROPOSAL_AUDIT.json", {
        "pool_count": len(pool),
        "excluded_count": len(excluded),
        "selected_new_windows": [{k: row[k] for k in ("review_id", "song_id", "context_id", "start_sec", "end_sec")}],
        "selection_uses_phone_predictions": False,
        "rejected_N06_protected": True,
    })
    result = {
        "schema_version": "batch_training_review_packet_v1",
        "source_bindings": sources,
        "items": [row],
        "prior_valid_windows": cumulative["valid_windows"],
        "active_slots_after": cumulative["valid_windows"] + 1,
        "new_slot_count": 1,
        "policy": policy_out,
        "training_started": False,
        "test_songs_opened": False,
    }
    result["packet_sha256"] = sha256_object(result)
    write_json_new(output / "packet.json", result)
    render_page(output, result)
    for spec in sources.values():
        verify_file(Path(spec["path"]), spec["sha256"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.output.resolve())
    row = result["items"][0]
    print("review_id", row["review_id"], "song", row["song_id"], "start", row["start_sec"], "end", row["end_sec"])
    print("packet", result["packet_sha256"])
