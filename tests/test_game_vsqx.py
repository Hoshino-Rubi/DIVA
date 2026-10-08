from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import threading
import wave

import pytest

from any2xsampa.local.common import ROOT, read, sha, write
from any2xsampa.local.game_score_tasks import build_game_score, export_saved_notes
from any2xsampa.local.generated_vsqx import export_generated_score
from any2xsampa.local.note_score import load_prediction, score_from_notes
from any2xsampa.local.note_windows import note_windows, stitch_note_windows
from any2xsampa.vocaloid.vsqx_alignment import parse_vsqx_timeline


def note(start, end, midi=60, index=0):
    return {"region_index": index, "start_sec": start, "end_sec": end,
            "duration_sec": end-start, "pitch_midi": midi, "pitch_midi_float": float(midi),
            "presence": True, "touches_input_start": False, "touches_input_end": False}


@pytest.fixture
def evidence(tmp_path):
    audio = tmp_path / "原唱 space.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        handle.writeframes(bytes(16000 * 2 * 35))
    prediction = {"schema_version": "note_transcription_v1", "mode": "fixture",
        "model_id": "fixture", "time_basis": "original_audio",
        "source_start_sec": 5.0, "source_end_sec": 8.0,
        "input": {"audio_path": str(audio), "audio_sha256": sha(audio)},
        "notes": [note(5.1, 5.6, 48, 0), note(5.6, 5.67, 48, 1), note(6.2, 7.9, 75, 3)]}
    phones = {"schema_version": "local_task_result_v1", "mode": "replay",
        "time_basis": "original_audio", "input": {"path": str(audio), "sha256": sha(audio),
        "start_sec": 5.0, "end_sec": 8.0}, "segments": [
        {"start_sec": 5.2, "end_sec": 5.21, "normalized_ipa": "i"},
        {"start_sec": 5.63, "end_sec": 5.64, "normalized_ipa": "i"},
        {"start_sec": 6.5, "end_sec": 6.51, "normalized_ipa": "u"}]}
    notes_file, phones_file = tmp_path/"预测.json", tmp_path/"结果.json"
    write(notes_file, prediction)
    write(phones_file, phones)
    return notes_file, phones_file


def test_notes_export_preserves_pitch_rest_short_and_repeated_notes(evidence, tmp_path):
    notes_file, phones_file = evidence
    before = (notes_file.read_bytes(), phones_file.read_bytes())
    output = tmp_path/"new song.vsqx"
    result = export_saved_notes({"result_path": str(notes_file), "phones_path": str(phones_file),
                                "output_path": str(output), "tempo": 137.25})
    restored = parse_vsqx_timeline(output)["notes"]
    source = read(notes_file)["notes"]
    assert [n["pitch"] for n in restored] == [48, 48, 75]
    assert len(restored) == 3 and result["placeholder_note_count"] == 0
    for a, b in zip(source, restored):
        assert b["start_sec"] == pytest.approx(a["start_sec"], abs=result["maximum_time_rounding_sec"]+1e-6)
        assert b["end_sec"] == pytest.approx(a["end_sec"], abs=result["maximum_time_rounding_sec"]+1e-6)
    assert restored[2]["start_sec"] > restored[1]["end_sec"] + .5
    assert before == (notes_file.read_bytes(), phones_file.read_bytes())
    proof = read(output.with_name(output.name+".provenance.json"))
    assert proof["ctc_times_used_as_note_durations"] is False
    assert proof["pitch_and_timing_readback_passed"] and proof["model_forwards"] == 0
    assert proof["manual_review_required"] is False and proof["quality_validated"] is False


def test_cli_exports_note_json_without_template_or_model(evidence, tmp_path):
    output = tmp_path/"命令行 导出.vsqx"
    completed = subprocess.run([sys.executable, "-B", "-m", "any2xsampa.cli", "export-vsqx",
        str(evidence[0]), "--phones", str(evidence[1]), "--output", str(output)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["note_count"] == 3 and payload["model_forwards"] == 0 and output.is_file()


def test_notes_only_export_is_explicit_about_missing_phones(evidence, tmp_path):
    output = tmp_path/"notes.vsqx"
    result = export_saved_notes({"result_path": str(evidence[0]), "output_path": str(output)})
    score = read(output.with_name(output.name+".score.json"))
    assert result["placeholder_note_count"] == 3
    assert score["phone_evidence_supplied"] is False
    assert [n["pitch"] for n in score["notes"]] == [48, 48, 75]
    assert all(n["lyric"] == "待定" and n["placeholder"] for n in score["notes"])


@pytest.mark.parametrize("field,value", [
    ("pitch_midi", 128), ("pitch_midi", True), ("pitch_midi", 60.5),
    ("start_sec", float("nan")), ("start_sec", 4.9), ("end_sec", 9.0),
    ("duration_sec", .4), ("presence", False)])
def test_invalid_notes_are_rejected(evidence, field, value):
    prediction = read(evidence[0])
    prediction["notes"][0][field] = value
    with pytest.raises(ValueError):
        score_from_notes(prediction, read(evidence[1]))


def test_empty_notes_never_generate_fixed_pitch(evidence):
    prediction = read(evidence[0])
    prediction["notes"] = []
    with pytest.raises(ValueError, match="No voiced pitch"):
        score_from_notes(prediction, read(evidence[1]))


def test_phone_mismatch_cannot_move_or_replace_the_notes(evidence):
    prediction, phones = read(evidence[0]), read(evidence[1])
    phones["input"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different audio"):
        score_from_notes(prediction, phones)


def test_overlapping_notes_rejected(evidence):
    prediction = read(evidence[0])
    prediction["notes"][1] = note(5.5, 6, 48, 1)
    with pytest.raises(ValueError, match="Overlapping"):
        score_from_notes(prediction, None)


def test_audio_tampering_rejected(evidence):
    prediction = read(evidence[0])
    Path(prediction["input"]["audio_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="audio hash"):
        load_prediction(evidence[0])


def test_no_overwrite_of_vsqx_or_sidecar(evidence, tmp_path):
    output = tmp_path/"protected.vsqx"
    proof = output.with_name(output.name+".provenance.json")
    proof.write_text("existing")
    with pytest.raises(FileExistsError):
        export_saved_notes({"result_path": str(evidence[0]), "output_path": str(output)})
    assert proof.read_text() == "existing" and not output.exists()


def test_invalid_grid_rejected_before_output(evidence, tmp_path):
    score = score_from_notes(read(evidence[0]), read(evidence[1]))
    score["tempo"] = 121.123
    file = tmp_path/"score.json"; write(file, score)
    with pytest.raises(ValueError, match="Tempo"):
        export_generated_score(file, tmp_path/"invalid.vsqx")
    assert not (tmp_path/"invalid.vsqx").exists()


def test_score_replay_never_starts_worker_or_modifies_evidence(evidence, tmp_path, monkeypatch):
    import any2xsampa.local.game_score_tasks as jobs
    def forbidden(*args, **kwargs):
        pytest.fail("saved-note export must never start inference")
    monkeypatch.setattr(jobs, "WorkerClient", forbidden)
    before = [p.read_bytes() for p in evidence]
    info = build_game_score({"result_path": str(evidence[1]), "notes_path": str(evidence[0]),
        "output_dir": str(tmp_path/"job")}, threading.Event(), lambda _: None)
    assert info["model_forwards"] == 0 and info["graph_calls"] == 0
    assert Path(info["vsqx_path"]).is_file()
    assert before == [p.read_bytes() for p in evidence]
    assert not (evidence[1].parent/"score-link.json").exists()


def test_window_plan_covers_whole_song_and_keeps_model_bounded():
    windows = note_windows(4.2, 83.17, 44100)
    assert windows[0]["keep_start_sec"] == 4.2
    assert windows[-1]["keep_end_sec"] == pytest.approx(83.17)
    assert all(.1 <= w["end_sec"]-w["start_sec"] <= 20.000001 for w in windows)
    assert all(a["keep_end_sec"] == b["keep_start_sec"] for a,b in zip(windows,windows[1:]))


def test_continuing_note_is_not_duplicated_at_artificial_seam():
    windows = note_windows(0, 35, 16000)
    merged = stitch_note_windows([{"notes": [note(17.5, 20, 64)]},
                                  {"notes": [note(18, 22.5, 64)]}], windows)
    assert len(merged) == 1 and merged[0]["start_sec"] == 17.5 and merged[0]["end_sec"] == 22.5
    assert merged[0]["joined_across_artificial_seam"] is True


def test_real_same_pitch_repeat_at_seam_is_not_merged():
    windows = note_windows(0, 35, 16000)
    merged = stitch_note_windows([{"notes": [note(17.5, 19, 64)]},
                                  {"notes": [note(19, 22, 64)]}], windows)
    assert len(merged) == 2 and merged[0]["end_sec"] == merged[1]["start_sec"] == 19


def test_different_pitch_and_rest_at_seam_are_preserved():
    windows = note_windows(0, 35, 16000)
    notes = stitch_note_windows([{"notes": [note(17, 18.5, 64)]},
                                 {"notes": [note(19.5, 22, 65)]}], windows)
    assert len(notes) == 2 and notes[0]["end_sec"] == 18.5 and notes[1]["start_sec"] == 19.5


def test_cancel_before_work_creates_no_job(evidence, tmp_path):
    from any2xsampa.local.common import Cancelled
    cancel = threading.Event(); cancel.set()
    with pytest.raises(Cancelled):
        build_game_score({"result_path": str(evidence[1]), "notes_path": str(evidence[0]),
                          "output_dir": str(tmp_path/"cancelled")}, cancel, lambda _: None)
    assert not (tmp_path/"cancelled").exists()


def test_full_worker_and_whole_job_roundtrip_with_fixture_runtime(evidence, tmp_path, monkeypatch):
    """Exercise the real orchestration/codec with no ONNX session or forward."""
    import numpy as np
    import any2xsampa.adapters.game_onnx as adapter
    import any2xsampa.local.game_score_tasks as jobs
    from any2xsampa.local.game_worker import GameEngine

    snapshot = tmp_path/"snapshot"; snapshot.mkdir(); write(snapshot/"snapshot-lock.json", {"fixture": True})
    monkeypatch.setattr(jobs, "runtime_config", lambda _: (snapshot, sys.executable))
    file = evidence[1]; payload = read(file)
    payload["input"]["start_sec"] = 0
    payload["input"]["end_sec"] = 35
    write(file, payload, replace=True)
    forward_lengths, requests = [], []

    class Runtime:
        def __init__(self, snapshot, *, device):
            assert device == "cpu"
            self.calls = []
        def infer(self, waveform):
            frames = round(waveform.shape[1]/441)
            boundary = frames//2
            cuts = np.zeros((1,frames),dtype=bool); cuts[0,boundary] = True
            self.calls.extend([{"graph":"fixture"}]*11)
            forward_lengths.append(waveform.shape[1]/44100)
            return {"durations":np.array([[boundary*.01,(frames-boundary)*.01]],np.float32),
                "presence":np.array([[True,True]]), "scores":np.array([[60,62]],np.float32),
                "maskN":np.ones((1,2),bool), "maskT":np.ones((1,frames),bool), "boundaries":cuts}

    def audio(req, destination):
        requests.append(dict(req))
        duration = req["end_sec"]-req["start_sec"]
        return np.zeros((1,round(duration*44100)),np.float32), {
            **req,"effective_start_sec":req["start_sec"],"effective_end_sec":req["end_sec"],
            "source_sample_rate":16000,"model_duration_sec":duration}

    monkeypatch.setattr(adapter, "GameOnnxRuntime", Runtime)
    monkeypatch.setattr(adapter, "load_audio", audio)
    closed = []
    class Client:
        def __init__(self, python, log, **kw):
            assert kw["module"] == "any2xsampa.local.game_worker"
            self.engine = GameEngine()
        def call(self, action, params, cancel, progress, **kw):
            assert kw["timeout"] == 180
            return self.engine.execute(action, params, cancel, progress)
        def close(self):
            self.engine.close(); closed.append(True)
    monkeypatch.setattr(jobs, "WorkerClient", Client)
    info = jobs.build_game_score({"result_path":str(file),"output_dir":str(tmp_path/"whole")},
                                threading.Event(),lambda _:None)
    assert closed == [True] and forward_lengths == [20,17]
    assert all(set(r) == {"audio_path","audio_sha256","start_sec","end_sec"} for r in requests)
    assert info["model_forwards"] == 2 and info["graph_calls"] == 22
    prediction_file = tmp_path/"whole/note-prediction.json"
    replay, _ = load_prediction(prediction_file)
    assert replay["source_end_sec"] == 35
    actual = parse_vsqx_timeline(Path(info["vsqx_path"]))["notes"]
    assert [n["pitch"] for n in actual] == [r["pitch_midi"] for r in replay["notes"]]
    changed = read(prediction_file)
    changed["notes"][0]["pitch_midi"] = 70
    write(prediction_file,changed,replace=True)
    with pytest.raises(ValueError,match="independently replayed"):
        load_prediction(prediction_file)


@pytest.mark.parametrize("field", ["lyrics", "phones", "pitch", "vsqx", "reference"])
def test_worker_rejects_decoding_conditions(field, tmp_path):
    from any2xsampa.local.game_worker import GameEngine
    params = {"audio_path":"x.wav","audio_sha256":"a"*64,"start_sec":0,"end_sec":1,
              "snapshot":str(tmp_path),"output_dir":str(tmp_path/"job"),field:[]}
    with pytest.raises(ValueError,match="only source audio"):
        GameEngine().execute("transcribe_notes",params,threading.Event(),lambda _:None)
    assert not (tmp_path/"job").exists()


def test_worker_failure_does_not_retry_or_report_success(evidence, tmp_path, monkeypatch):
    import any2xsampa.local.game_score_tasks as jobs
    snapshot = tmp_path/"snapshot"; snapshot.mkdir()
    monkeypatch.setattr(jobs,"runtime_config",lambda _: (snapshot,sys.executable))
    calls,closed=[],[]
    class Broken:
        def __init__(self,*args,**kw):pass
        def call(self,*args,**kw):
            calls.append(True);raise RuntimeError("fixture failure")
        def close(self):closed.append(True)
    monkeypatch.setattr(jobs,"WorkerClient",Broken)
    out=tmp_path/"failed"
    with pytest.raises(RuntimeError,match="fixture failure"):
        jobs.build_game_score({"result_path":str(evidence[1]),"output_dir":str(out)},
                              threading.Event(),lambda _:None)
    assert calls==closed==[True]
    assert read(out/"EXIT.json")["status"]=="failed"
    assert read(out/"EXIT.json")["retried"] is False
    assert not (out/"DELIVERY.json").exists() and not (out/"transcription.vsqx").exists()


def test_real_json_without_raw_evidence_cannot_claim_real(evidence):
    payload=read(evidence[0]);payload["mode"]="real"
    write(evidence[0],payload,replace=True)
    with pytest.raises(ValueError,match="require raw evidence"):
        load_prediction(evidence[0])


def test_pip_resource_does_not_contain_historical_notes():
    from any2xsampa.local.generated_vsqx import TEMPLATE
    from xml.etree import ElementTree as ET
    root=ET.parse(TEMPLATE).getroot(); ns=root.tag.split("}")[0]+"}"
    assert not root.findall(".//"+ns+"note")
    assert not root.findall(".//"+ns+"wavPart")
    assert root.findtext(".//"+ns+"vsPart/"+ns+"t")=="0"
    assert root.findtext(".//"+ns+"preMeasure")=="0"


def test_half_tick_rounding_is_not_confused_by_timeline_display_precision(evidence,tmp_path):
    prediction=read(evidence[0])
    prediction["notes"]=[note(5+0.5/960,5.25+0.5/960,60)]
    score=score_from_notes(prediction,None)
    file=tmp_path/"half.json";write(file,score)
    result=export_generated_score(file,tmp_path/"half.vsqx")
    assert result["maximum_onset_roundtrip_error_sec"] <= 0.5/960+1e-9
    assert result["maximum_offset_roundtrip_error_sec"] <= 0.5/960+1e-9
