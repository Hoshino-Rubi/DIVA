"""Bind a downloaded challenger to exact code, data, policy and dependencies."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, write_json_new, verify_file


def freeze(run: Path) -> None:
    import yaml
    metadata = read_json(run / "espnet-powsm_ctc.metadata.json")
    downloads = read_json(run / "DOWNLOADS.json")
    for record in downloads["files"]:
        verify_file(Path(record["path"]), record["sha256"])
    for member in metadata["siblings"]:
        path = run / "model" / member["rfilename"]
        if path.stat().st_size != member["size"]:
            raise ValueError("snapshot size mismatch: " + member["rfilename"])
        if "lfs" in member:
            verify_file(path, member["lfs"]["sha256"])
        else:
            data = path.read_bytes()
            digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if digest != member["blobId"]:
                raise ValueError("snapshot git blob mismatch: " + member["rfilename"])
    config_file = next(x["rfilename"] for x in metadata["siblings"] if x["rfilename"].endswith("config.yaml"))
    weight_file = next(x["rfilename"] for x in metadata["siblings"] if x["rfilename"].endswith(".pth"))
    config = yaml.safe_load((run / "model" / config_file).read_text(encoding="utf-8"))
    write_json_new(run / "vocabulary.json", config["token_list"])
    source = run / "source" / ("espnet-" + downloads["espnet_revision"])
    files = [p for base in (run / "model", source) for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files += [run / "vocabulary.json", run / "POLICY.json"]
    lock = {"schema_version": "powsm_snapshot_lock_v1", "model_id": "espnet/powsm_ctc",
        "revision": metadata["sha"], "license": metadata["cardData"]["license"],
        "espnet_revision": downloads["espnet_revision"],
        "source_archive_sha256": sha256_file(run / "espnet-source.zip"),
        "source_directory": source.relative_to(run).as_posix(),
        "config_file": "model/" + config_file, "weight_file": "model/" + weight_file,
        "weight_sha256": sha256_file(run / "model" / weight_file),
        "vocabulary_size": len(config["token_list"]),
        "files": [{"file": p.relative_to(run).as_posix(), "sha256": sha256_file(p)} for p in sorted(files)]}
    write_json_new(run / "MODEL_LOCK.json", lock)
    prior = ROOT / ".state/runs/singing-domain-20260920T095918143867Z"
    requests = read_json(prior / "DEVELOPMENT_REQUESTS.json")["rows"]
    original = [r for r in requests if r["audio_role"] == "original"]
    vocal = [r for r in requests if r["audio_role"] not in {"original", "rendered"}]
    if len(original) != 3 or len(vocal) != 3:
        raise ValueError("expected exactly three existing original/vocal pairs")
    # The explicit role rename changes metadata only; all source clocks/hashes remain.
    vocal = [{**r, "audio_role": "separated_vocal"} for r in vocal]
    first = next(r for r in read_json(prior / "AUDIO_REQUESTS.json")["rows"] if r["audio_role"] == "original")
    for seconds in (8, 20):
        row = {**first, "example_id": "training-smoke-" + str(seconds), "end_sec": first["start_sec"] + seconds}
        write_json_new(run / ("smoke-" + str(seconds) + ".json"), {"rows": [row]})
    write_json_new(run / "original-development.json", {"rows": original})
    write_json_new(run / "vocal-development.json", {"rows": vocal})
    refs = read_json(prior / "DEVELOPMENT_REFERENCES.json")
    rows = [r for r in refs["rows"] if r["cohort"] in {"original_dev", "vocal_dev"}]
    if sum(len(r["target_ipa"]) for r in rows if r["cohort"] == "original_dev") != 214:
        raise ValueError("development reference denominator changed")
    write_json_new(run / "SCORING_REFERENCES.json", {"rows": rows,
        "source": str(prior / "DEVELOPMENT_REFERENCES.json"), "source_sha256": sha256_file(prior / "DEVELOPMENT_REFERENCES.json")})
    package_result = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"],
        check=True, capture_output=True, text=True, encoding="utf-8")
    (run / "requirements.lock.txt").write_text(package_result.stdout, encoding="utf-8")
    bindings = {str(p.resolve()): sha256_file(p) for p in [prior / "DEVELOPMENT_REQUESTS.json",
        prior / "DEVELOPMENT_REFERENCES.json", prior / "DELIVERY.json", prior / "AUDIO_REQUESTS.json"]}
    for row in [first] + original + vocal:
        verify_file(Path(row["audio_path"]), row["audio_sha256"])
        bindings[row["audio_path"]] = row["audio_sha256"]
    write_json_new(run / "FREEZE.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_bindings": bindings, "policy_sha256": sha256_file(run / "POLICY.json"),
        "model_lock_sha256": sha256_file(run / "MODEL_LOCK.json"),
        "requirements_sha256": sha256_file(run / "requirements.lock.txt"),
        "no_reference_supplied_to_worker": True, "final_test_predictions_allowed": False})
    print("frozen", len(files), "snapshot files; 3 original + 3 vocal development windows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    freeze(parser.parse_args().run.resolve())
