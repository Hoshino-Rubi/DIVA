from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import finite_number, read_json, require_digest, sha256_file, sha256_object, verify_file, write_json_new  # noqa: E402
from any2xsampa.training.limited_data_reference import SONG_SPLIT, SPLITS  # noqa: E402


AUDIO_KEYS = {"example_id", "song_id", "split", "audio_role", "audio_path", "audio_sha256", "start_sec", "end_sec"}


def validate_requests(requests: dict) -> None:
    expected_keys = {
        "schema_version", "purpose", "source_dataset_hash", "source_manifest_file_sha256",
        "split_sha256", "rows", "contains_labels_or_score_context",
    }
    if set(requests) != expected_keys:
        raise ValueError("unexpected request fields; acoustic workers do not accept reference context")
    if requests.get("schema_version") != "phoneticxeus_audio_requests_v1":
        raise ValueError("unsupported acoustic request schema")
    if requests.get("contains_labels_or_score_context") is not False:
        raise ValueError("acoustic requests must not contain labels/score context")
    if requests.get("split_sha256") != sha256_object(SPLITS):
        raise ValueError("unrecognized song split")
    if requests.get("purpose") not in {"smoke8", "smoke20", "original-baselines", "original-reviewed-v2", "rendered-features"}:
        raise ValueError("unsupported acoustic request purpose")
    seen: set[str] = set()
    for key in ("source_dataset_hash", "source_manifest_file_sha256", "split_sha256"):
        require_digest(requests[key], key)
    counts: dict[str, int] = {}
    for row in requests["rows"]:
        if set(row) != AUDIO_KEYS:
            raise ValueError("unexpected acoustic input fields; labels and score are forbidden")
        if row["example_id"] in seen or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,119}", row["example_id"]) is None:
            raise ValueError("invalid or duplicate example ID")
        seen.add(row["example_id"])
        if SONG_SPLIT.get(row["song_id"]) != row["split"]:
            raise ValueError("song split mismatch")
        counts[row["song_id"]] = counts.get(row["song_id"], 0) + 1
        require_digest(row["audio_sha256"], "audio_sha256")
        start = finite_number(row["start_sec"], "start_sec", minimum=0)
        end = finite_number(row["end_sec"], "end_sec", minimum=0)
        if not 0 < end - start <= 20.000001:
            raise ValueError("acoustic request duration must be >0 and <=20 seconds")
        if requests["purpose"] == "rendered-features":
            if row["split"] not in {"train", "dev"} or row["audio_role"] != "rendered":
                raise ValueError("test/original audio cannot enter rendered feature training")
        elif row["audio_role"] != "original":
            raise ValueError("baseline input must be original audio")
    if not seen:
        raise ValueError("empty acoustic request")
    if requests["purpose"] == "original-baselines" and counts != {song: 3 for song in SONG_SPLIT}:
        raise ValueError("original baseline batch must cover the 24 predeclared clips")
    if requests["purpose"] in {"smoke8", "smoke20"}:
        expected_duration = 8 if requests["purpose"] == "smoke8" else 20
        row = requests["rows"][0]
        if len(seen) != 1 or row["song_id"] != "freedom-flower" or abs(row["end_sec"] - row["start_sec"] - expected_duration) > 1e-6:
            raise ValueError("smoke request differs from its predeclared train-song duration")


def _run_batch(args: argparse.Namespace, destination: Path, requests: dict) -> int:
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor

    with ProcessResourceMonitor(destination, timeout_seconds=2400), ExitStack() as stack:
        cache = None
        if getattr(args, "audio_cache", False):
            from any2xsampa.adapters.locked_audio_cache import LockedAudioCache
            cache = stack.enter_context(LockedAudioCache(max_bytes=8 * 1024 ** 2, max_sources=8))
        return _run_monitored_batch(args, destination, requests, audio_cache=cache)


def _run_monitored_batch(args: argparse.Namespace, destination: Path, requests: dict, *, audio_cache=None) -> int:
    started = datetime.now(timezone.utc).isoformat()
    before = time.monotonic()
    from any2xsampa.adapters.phoneticxeus_runtime import PhoneticXeusRuntime
    from safetensors.torch import save_file

    runtime = PhoneticXeusRuntime(args.snapshot, cache_dir=destination / "runtime-cache", audio_cache=audio_cache)
    head = runtime.model.model.ctc.ctc_lo
    head_path = destination / "pretrained-head.safetensors"
    save_file({name: tensor.detach().cpu().contiguous() for name, tensor in head.state_dict().items()}, str(head_path))
    entries = []
    for index, row in enumerate(requests["rows"], 1):
        sealed = row["split"] == "test"
        path = destination / ("sealed" if sealed else "evidence") / row["example_id"]
        record = runtime.save_example(row, path, sealed=sealed)
        entries.append(record)
        print(f"completed={index}/{len(requests['rows'])}; id={row['example_id']}; frames={record['frames']}; sealed={sealed}", flush=True)
    manifest = {
        "schema_version": "phoneticxeus_feature_index_v1",
        "purpose": requests["purpose"],
        "source_dataset_hash": requests["source_dataset_hash"],
        "requests_sha256": args.requests_sha256,
        "split_sha256": requests["split_sha256"],
        "model_metadata": runtime.identity,
        "head": {"path": str(head_path), "sha256": sha256_file(head_path)},
        "entries": entries, "runtime": runtime.runtime,
        "started_at_utc": started, "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - before,
        "all_parameters_frozen": not any(p.requires_grad for p in runtime.model.parameters()),
        "no_labels_or_score_received": True,
        "audio_cache": audio_cache.stats() if audio_cache is not None else None,
        "audio_cache_opt_in": audio_cache is not None,
    }
    print(f"index_sha256={write_json_new(destination / 'index.json', manifest)}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated offline PhoneticXeus audio-only producer.")
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--requests-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=["cuda"])
    parser.add_argument("--audio-cache", action="store_true", help="Opt-in Windows read-locked audio LRU (8 MiB); never caches model features/logits.")
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if not destination.is_relative_to(ROOT / ".state/runs"):
        parser.error("model outputs must remain under .state/runs")
    if destination.exists():
        parser.error("refusing an existing output directory")
    verify_file(args.requests, args.requests_sha256)
    requests = read_json(args.requests)
    validate_requests(requests)
    destination.mkdir(parents=True)
    return _run_batch(args, destination, requests)


if __name__ == "__main__":
    raise SystemExit(main())
