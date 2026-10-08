from __future__ import annotations

import copy
import json
from pathlib import Path
import threading
import wave

import numpy as np
import pytest
from safetensors.numpy import save_file

from any2xsampa.adapters.phoneticxeus import emission_payload
from any2xsampa.local.common import Cancelled, read, sha, write
from any2xsampa.local.engine import LocalEngine
from any2xsampa.local.whole_audio import POLICY, merge_results, plan_chunks, whole_input
from any2xsampa.local.whole_task import replay_whole, run_whole


@pytest.fixture
def song(tmp_path: Path) -> Path:
    file = tmp_path / "whole 歌曲.wav"
    with wave.open(str(file), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(16000 * 40 * 2))
    return file


def test_plan_has_exact_coverage_and_bounded_windows(song: Path) -> None:
    info = whole_input(song)
    chunks = plan_chunks(info)
    assert len(chunks) == 3
    assert [(x["start_sec"], x["end_sec"]) for x in chunks] == [
        (0, 20), (18, 38), (36, 40)
    ]
    assert [(x["keep_start_sec"], x["keep_end_sec"]) for x in chunks] == [
        (0, 19), (19, 37), (37, 40)
    ]
    assert all(.1 <= x["end_sec"] - x["start_sec"] <= 20 for x in chunks)


def test_plan_rejects_subminimum_tail_and_keeps_long_song_finite(tmp_path: Path) -> None:
    from any2xsampa.local.whole_audio import plan_chunks
    assert len(plan_chunks({"sample_rate": 16000, "frames": 18*16000+800})) == 1
    chunks = plan_chunks({"sample_rate": 16000, "frames": 20*16000+800})
    assert len(chunks) == 2 and chunks[-1]["end_sec"]-chunks[-1]["start_sec"] >= .1
    long = plan_chunks({"sample_rate": 96000, "frames": 253*96000+5760})
    assert len(long) == 14 and all(x["end_sec"]-x["start_sec"] <= 20 for x in long)


def candidate_manifest(song: Path, tmp_path: Path) -> dict:
    info = whole_input(song)
    chunks = plan_chunks(info)
    vocabulary = {"<blank>": 0, "a": 1, "b": 2}
    geometry = {"stride_samples": 320, "receptive_field_samples": 400}
    identity = {"vocabulary_sha256": ""}
    manifest = {
        "schema_version": "local_whole_evidence_v1", "policy": POLICY,
        "input": info, "chunks": chunks, "completed": [],
        "backend": "guarded-phoneticxeus", "model_id": "mock",
        "package_sha256": "fixture-package", "device": "cuda",
    }
    for chunk in chunks:
        directory = tmp_path / f"chunk-{chunk['index']}"
        directory.mkdir(parents=True)
        vocab_file = directory / "vocabulary.json"
        write(vocab_file, vocabulary)
        identity["vocabulary_sha256"] = sha(vocab_file)
        size = (chunk["source_end_sample_exclusive"] - chunk["source_start_sample"])
        frame_count = (size - 400) // 320 + 1
        logits = np.full((frame_count, len(vocabulary)), -8, dtype=np.float32)
        for frame in range(frame_count):
            global_id = chunk["source_start_sample"] // 320 + frame
            # An uninterrupted 'a' crosses the 19-second seam; the blank
            # between two later a-runs must still create two emissions.
            label = 1 if 940 <= global_id < 970 else 0
            if 970 <= global_id < 975:
                label = 0
            if 975 <= global_id < 990:
                label = 1
            if 1840 <= global_id < 1860:
                label = 2
            logits[frame, label] = 8
        evidence = directory / "evidence.safetensors"
        save_file({"frame_logits": logits}, str(evidence))
        crop = {
            "source_start_sample": chunk["source_start_sample"],
            "source_end_sample_exclusive": chunk["source_end_sample_exclusive"],
            "effective_start_sec": chunk["start_sec"],
            "effective_end_sec": chunk["end_sec"],
        }
        posterior = emission_payload(
            logits, vocabulary, input_audio=info, crop=crop,
            model_metadata=identity,
            geometry={**geometry, "frame_count": frame_count}, runtime={},
        )
        posterior["raw_evidence"] = {
            "file": evidence.name, "sha256": sha(evidence),
            "vocabulary_file": vocab_file.name,
        }
        pp = directory / "posterior.json"
        write(pp, posterior)
        result = {
            "schema_version": "local_task_result_v1", "mode": "real",
            "input": {
                **info, "source_start_sample": chunk["source_start_sample"],
                "source_end_sample_exclusive": chunk["source_end_sample_exclusive"],
            },
            "time_basis": "original_audio", "model_id": "mock",
            "package_sha256": "fixture-package", "device": "cuda",
            "posterior_path": str(pp), "posterior_sha256": sha(pp),
            "segments": [], "quality": "fixture-only", "load_count": 1,
        }
        result_file = directory / "result.json"
        write(result_file, result)
        manifest["completed"].append({
            "index": chunk["index"], "result_path": str(result_file),
            "result_sha256": sha(result_file),
        })
    return manifest


def test_candidate_stitches_frames_not_phone_strings(song: Path, tmp_path: Path) -> None:
    manifest = candidate_manifest(song, tmp_path)
    result = merge_results(manifest, threading.Event())
    assert [x["normalized_ipa"] for x in result["segments"]] == ["a", "a", "b"]
    assert len(result["merge_summary"]["ranges"]) == 3
    assert result["merge_summary"]["frame_count"] == 1999
    assert result["segments"][0]["end_sec"] > 19
    assert all(x["metadata"]["human_reference"] is False for x in result["segments"])


def test_rejects_missing_window_and_modified_evidence(song: Path, tmp_path: Path) -> None:
    manifest = candidate_manifest(song, tmp_path)
    missing = copy.deepcopy(manifest)
    missing["completed"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        merge_results(missing, threading.Event())
    first = Path(manifest["completed"][0]["result_path"])
    first.write_text(first.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        merge_results(manifest, threading.Event())


def test_replay_whole_is_independent_and_never_calls_model(song: Path, tmp_path: Path) -> None:
    manifest = candidate_manifest(song, tmp_path)
    source = tmp_path / "whole-evidence.json"
    write(source, manifest)
    engine = LocalEngine()
    evidence = merge_results(manifest, threading.Event())
    evidence.update(whole_evidence_path=str(source), whole_evidence_sha256=sha(source))
    first_dir = tmp_path / "first"
    first_dir.mkdir()
    engine.save_result(first_dir, "real", manifest["input"], evidence)
    original = first_dir / "result.json"
    result = replay_whole(engine, {"fixture": str(original), "output_dir": str(tmp_path / "replay")},
                          threading.Event(), lambda _: None)
    assert result["mode"] == "replay"
    assert read(Path(result["result_path"]))["segments"] == read(original)["segments"]
    assert read(Path(result["output_dir"]) / "EXIT.json")["model_forwards"] == 0


def test_onnx_blank_and_repeated_phone_across_seam(song: Path, tmp_path: Path) -> None:
    info = whole_input(song)
    chunks = plan_chunks(info)
    manifest = {"schema_version":"local_whole_evidence_v1","policy":POLICY,
                "input":info,"chunks":chunks,"completed":[],"backend":"onnx-espeak",
                "model_id":"onnx","package_sha256":"mock","device":"cpu"}
    for chunk in chunks:
        directory=tmp_path / str(chunk["index"])
        directory.mkdir()
        frames=[]
        for frame in range(int((chunk["end_sec"]-chunk["start_sec"])*50)):
            global_frame=int(chunk["start_sec"]*50)+frame
            symbol = "a" if 940 <= global_frame < 965 or 975 <= global_frame < 990 else None
            frames.append({"start_index":frame*320,"end_index":(frame+1)*320,
                           "symbol":symbol,"candidates":[{"symbol":"a","confidence":.8}] if symbol else []})
        frame_path=directory/"onnx-frames.json"
        write(frame_path,{"schema_version":"local_onnx_frames_v1","frames":frames})
        result={"mode":"real","input":{**info,"source_start_sample":chunk["source_start_sample"],
                  "source_end_sample_exclusive":chunk["source_end_sample_exclusive"]},
                "time_basis":"original_audio","package_sha256":"mock","model_id":"onnx","device":"cpu",
                "frame_evidence_path":str(frame_path),"frame_evidence_sha256":sha(frame_path),"segments":[]}
        result_path=directory/"result.json"
        write(result_path,result)
        manifest["completed"].append({"result_path":str(result_path),"result_sha256":sha(result_path)})
    merged=merge_results(manifest,threading.Event())
    assert [s["normalized_ipa"] for s in merged["segments"]]==["a","a"]
    assert merged["merge_summary"]["blank_frames"] > 0


def test_run_whole_serial_progress_cancel_preserves_partial(song: Path, tmp_path: Path, monkeypatch) -> None:
    from any2xsampa.local import whole_task

    manifest = candidate_manifest(song, tmp_path / "mock")
    monkeypatch.setattr(whole_task, "package", lambda _: {
        "backend": "guarded-phoneticxeus", "devices": ["cuda"],
        "package_sha256": "fixture-package",
    })
    class Engine:
        def __init__(self):
            self.calls = 0
            self.closed = 0
        def execute(self, action, params, cancel, progress):
            assert action == "recognize"
            assert params["start"] == manifest["chunks"][self.calls]["start_sec"]
            assert params["duration"] <= 20
            row = manifest["completed"][self.calls]
            self.calls += 1
            cancel.set()
            return {"result_path": row["result_path"]}
        def close(self): self.closed += 1
    engine = Engine()
    flag = threading.Event()
    with pytest.raises(Cancelled):
        run_whole(engine, {"audio": str(song), "model": "mock", "device": "cuda",
                           "output_dir": str(tmp_path / "out")}, flag, lambda _: None)
    assert engine.calls == 1 and engine.closed == 1
    assert read(tmp_path / "out/EXIT.json")["chunks_completed"] == 1
    assert read(tmp_path / "out/EXIT.json")["status"] == "cancelled"
    assert not (tmp_path / "out/result.json").exists()


def test_run_whole_success_keeps_one_worker_and_replays(song: Path, tmp_path: Path, monkeypatch) -> None:
    from any2xsampa.local import whole_task

    manifest = candidate_manifest(song, tmp_path / "mock")
    monkeypatch.setattr(whole_task, "package", lambda _: {
        "backend": "guarded-phoneticxeus", "devices": ["cuda"],
        "package_sha256": "fixture-package",
    })

    class Engine(LocalEngine):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def execute(self, action, params, cancel, progress):
            assert action == "recognize" and params["duration"] <= 20
            row = manifest["completed"][self.calls]
            self.calls += 1
            return {"result_path": row["result_path"]}

    engine = Engine()
    events = []
    result = run_whole(engine, {"audio": str(song), "model": "mock", "device": "cuda",
                "output_dir": str(tmp_path / "whole")}, threading.Event(), events.append)
    assert engine.calls == 3
    assert [event["chunk_index"] for event in events if event["stage"] == "chunk_done"] == [1, 2, 3]
    assert read(tmp_path / "whole/EXIT.json")["model_forwards"] == 3
    assert [row["normalized_ipa"] for row in read(Path(result["result_path"]))["segments"]] == ["a", "a", "b"]
    replay = replay_whole(engine, {"fixture": result["result_path"],
               "output_dir": str(tmp_path / "whole-replay")}, threading.Event(), events.append)
    assert read(Path(replay["result_path"]))["segments"] == read(Path(result["result_path"]))["segments"]
