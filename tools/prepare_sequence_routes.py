"""Freeze continuation routes without opening held-out final-test predictions."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.training.limited_data_artifacts import read_json, write_json_new, verify_file, sha256_file


def prepare(run: Path) -> None:
    run.mkdir(parents=True, exist_ok=False)
    parents = {"singing": ROOT / ".state/runs/singing-domain-20260920T095918143867Z",
               "powsm": ROOT / ".state/runs/powsm-proxy18080-20260920T125400Z"}
    preserved = {}
    for parent in parents.values():
        delivery = read_json(parent / "DELIVERY.json")
        for row in delivery["source_files"] + delivery["root_artifacts"]:
            verify_file(Path(row["path"]), row["sha256"])
            preserved[row["path"]] = row["sha256"]
    spec = read_json(ROOT / "src/any2xsampa/config/sequence_routes_v1.json")
    write_json_new(run / "POLICY.json", spec)
    old = parents["singing"]
    requests = [r for r in read_json(old / "AUDIO_REQUESTS.json")["rows"] if r["audio_role"] == "original"]
    refs = [r for r in read_json(old / "REFERENCES.json")["rows"] if r["audio_role"] == "original"]
    if len(requests) != 20 or len(refs) != 20 or sum(len(r["target_ipa"]) for r in refs) != 1235:
        raise ValueError("confirmed training cohort changed")
    if any(r["split"] != "train" or r["song_id"] not in spec["songs"] for r in requests):
        raise ValueError("training-song leakage")
    for row in requests:
        verify_file(Path(row["audio_path"]), row["audio_sha256"])
        preserved[row["audio_path"]] = row["audio_sha256"]
    write_json_new(run / "POWSM_AUDIO_REQUESTS.json", {"rows": requests, "labels_received": False})
    write_json_new(run / "POWSM_SCORING_REFERENCES.json", {"rows": refs, "input_to_worker": False})
    write_json_new(run / "PRESERVATION_BEFORE.json", {"files": preserved})
    write_json_new(run / "FREEZE.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parents": {k: str(v) for k,v in parents.items()}, "policy_sha256": sha256_file(run / "POLICY.json"),
        "original_requests_sha256": sha256_file(run / "POWSM_AUDIO_REQUESTS.json"),
        "original_references_sha256": sha256_file(run / "POWSM_SCORING_REFERENCES.json"),
        "powsm_python": str(parents["powsm"] / "runtime/Scripts/python.exe"),
        "training_python": str(ROOT / ".state/runs/limited-data-20260906T050251/runtime/Scripts/python.exe"),
        "proxy_for_any_future_download": "http://127.0.0.1:18080", "training_started": False})
    print(str(run))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    prepare(parser.parse_args().run.resolve())
