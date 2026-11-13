"""Candidate-4 task contracts with tiny local assets and no model/network work."""
from __future__ import annotations

import builtins
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import urllib.request
import wave

import pytest

from any2xsampa.local import cantonese4 as tasks
from any2xsampa.local import cli, process
from any2xsampa.local.common import Cancelled, read, sha, write
from any2xsampa.local.engine import LocalEngine


COMPLETE = {"status": "ok", "eos_observed": True,
            "truncated_or_incomplete": False, "truncation_status": "complete_with_eos"}
CODE_PATHS = (
    "separator/code/configs/config_vocals_mel_band_roformer.yaml",
    "separator/code/models/mel_band_roformer/__init__.py",
    "separator/code/models/mel_band_roformer/attend.py",
    "separator/code/models/mel_band_roformer/mel_band_roformer.py",
)


def forbidden(*args, **kwargs):
    pytest.fail("Task attempted real networking or model worker startup")


@pytest.fixture(autouse=True)
def isolated_tasks(tmp_path, monkeypatch):
    root = tmp_path / "product"
    root.mkdir()
    monkeypatch.setattr(tasks, "ROOT", root)
    monkeypatch.setattr(tasks, "ASSET_LOCK", tmp_path / "assets.json")
    monkeypatch.setenv("ANY2XSAMPA_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(process, "WorkerClient", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(urllib.request, "build_opener", forbidden)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", forbidden)
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers"} or name.endswith("cantonese4_runtime"):
            pytest.fail(f"Task loaded model implementation: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)


@pytest.fixture
def assets(tmp_path):
    bundle = tmp_path / "selected bundle 中文"
    payloads = {path: f"tiny source {index}\n".encode()
                for index, path in enumerate(CODE_PATHS)}
    payloads.update({"small/pytorch_model.bin": b"small weight fixture",
                     "base/pytorch_model.bin": b"base weight fixture",
                     "separator/MelBandRoformer.ckpt": b"separator weight fixture"})
    rows = [{"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
             "url": "https://assets.example.invalid/pinned/" + path}
            for path, data in payloads.items()]
    write(tasks.ASSET_LOCK, {"schema_version": "cantonese4_assets_v1", "files": rows})
    for path, data in payloads.items():
        if path not in CODE_PATHS:
            file = bundle / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(data)
    python = tasks.ROOT / "runtime/phonetic/python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"not executable; workers must be mocked")
    return SimpleNamespace(bundle=bundle, params={"bundle_dir": str(bundle)},
                           payloads=payloads, rows=rows, python=python)


def fill_code(assets):
    for path in CODE_PATHS:
        file = assets.bundle / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(assets.payloads[path])


def make_wav(path, *, frames=32001, rate=16000, channels=1):
    with wave.open(str(path), "wb") as stream:
        stream.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * frames * channels)
    return path


@pytest.fixture
def audio(tmp_path):
    return make_wav(tmp_path / "输入 audio.wav")


def prediction(text="nei", **extra):
    return {**COMPLETE, "decoded": text, "token_ids": [11, 22, 33],
            "diagnostics": {"verbatim": "  中文\n", "score": -3.0}, **extra}


def window(*, start=0.0, end=1.0, small=None, base=None):
    return {"start_sec": start, "end_sec": end, "small": small, "base": base}


def raw_fixture(path, windows):
    write(path, {"schema_version": "cantonese4_raw_v1", "windows": windows})
    return path


class Response(io.BytesIO):
    def __init__(self, data, *, status=200, url="https://cdn.example.invalid/source", headers=None):
        super().__init__(data)
        self.status = status
        self.url = url
        self.headers = {"Content-Length": str(len(data))} if headers is None else headers

    def geturl(self):
        return self.url


class FakeOpener:
    def __init__(self, assets, response=None):
        self.payloads = {row["url"]: assets.payloads[row["path"]]
                         for row in assets.rows if row["path"] in CODE_PATHS}
        self.calls = []
        self.response = response

    def open(self, request, timeout):
        # Weight URLs deliberately have no entry: even one weight request fails.
        data = self.payloads[request.full_url]
        self.calls.append((request, timeout))
        return self.response if self.response is not None else Response(data)


@pytest.mark.parametrize("channels", [1, 2])
def test_validate_whole_uses_exact_wav_frames_and_preserves_input(audio, channels, tmp_path):
    make_wav(audio, frames=32001, channels=channels)
    before = audio.read_bytes()
    params = {"audio": str(audio), "scope": "whole_file", "start": 1.0, "duration": .1,
              "output_dir": str(tmp_path / "unused")}
    original = deepcopy(params)
    result = tasks.validate(params)
    assert result["scope"] == "whole_file"
    assert result["start"] == 0 and result["duration"] == 32001 / 16000
    assert result["audio"] == str(audio.resolve())
    assert params == original and audio.read_bytes() == before
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("start,frames", [(0, 1600), (1600, 3200), (30401, 1600)])
def test_validate_clip_accepts_exact_frame_intervals(audio, start, frames):
    result = tasks.validate({"audio": str(audio), "scope": "clip",
                             "start": start / 16000, "duration": frames / 16000})
    assert result["start"] == start / 16000
    assert result["duration"] == frames / 16000


@pytest.mark.parametrize("changes", [
    {"start": -1 / 16000}, {"start": 30402 / 16000, "duration": .1},
    {"duration": 0}, {"duration": 1599 / 16000}, {"duration": 20.0001},
    {"start": True}, {"duration": False}, {"start": float("nan")},
    {"duration": float("inf")}, {"scope": "unknown"}, {"lyrics": "forbidden"},
])
def test_validate_rejects_invalid_clip_and_extra_conditions(audio, changes):
    with pytest.raises(ValueError):
        tasks.validate({"audio": str(audio), "scope": "clip", "start": 0, "duration": .1, **changes})


@pytest.mark.parametrize("frames,channels,valid", [(42000, 1, True), (42001, 1, False),
                                                  (0, 1, False), (100, 3, False)])
def test_validate_audio_limits_use_frame_count(tmp_path, frames, channels, valid):
    audio = make_wav(tmp_path / "limits.wav", frames=frames, rate=100, channels=channels)
    if valid:
        assert tasks.validate({"audio": str(audio)})["duration"] == 420
    else:
        with pytest.raises(ValueError, match="mono/stereo PCM WAV"):
            tasks.validate({"audio": str(audio)})


@pytest.mark.parametrize("state", ["asset_lock_missing", "models_missing", "separator_code_missing",
                                  "runtime_missing", "ready"])
def test_status_is_read_only_and_never_loads_models(assets, state):
    if state in {"runtime_missing", "ready"}:
        fill_code(assets)
    if state == "asset_lock_missing":
        tasks.ASSET_LOCK.unlink()
    elif state == "models_missing":
        (assets.bundle / "small/pytorch_model.bin").unlink()
    elif state == "runtime_missing":
        assets.python.unlink()
    before = {p: p.read_bytes() for p in assets.bundle.rglob("*") if p.is_file()}
    engine = LocalEngine()
    # Status must leave an existing worker alive instead of loading/unloading it.
    existing = SimpleNamespace(close=forbidden)
    engine.worker = existing
    result = engine.execute("cantonese4_status", assets.params, threading.Event(), forbidden)
    assert result["reason"] == state
    assert result["model_loaded"] is False
    assert result["available"] is (state == "ready")
    assert engine.worker is existing
    assert before == {p: p.read_bytes() for p in assets.bundle.rglob("*") if p.is_file()}
    assert not (tasks.ROOT / ".state").exists()


def test_replay_three_selection_paths_are_lossless_and_untimed(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "status", forbidden)
    small = prediction("La la HnG，", score=-100)
    base = prediction("nei hou", score=100)
    failed = prediction("do not repair", status="failed", error="raw failure")
    incomplete = prediction("ignored", eos_observed=False)
    windows = [window(small=small, base=base),
               window(start=1, end=2, small=failed, base=prediction("nei")),
               window(start=2, end=3, small=incomplete, base=None)]
    original = deepcopy(windows)
    fixture = raw_fixture(tmp_path / "raw.json", windows)
    raw_bytes = fixture.read_bytes()
    output = tmp_path / "result"
    engine = LocalEngine()
    closed = []
    engine.worker = SimpleNamespace(close=lambda: closed.append(True))
    updates = []
    summary = engine.execute("cantonese4_replay", {"fixture": str(fixture), "output_dir": str(output)},
                             threading.Event(), updates.append)
    result = read(Path(summary["result_path"]))
    assert closed == [True] and engine.worker is None
    assert result["mode"] == "replay"
    assert result["source"] == {"fixture_sha256": hashlib.sha256(raw_bytes).hexdigest(), "model_forwards": 0}
    assert [row["selection"]["selected_model"] for row in result["windows"]] == ["small", "base", None]
    assert [row["selection"]["reason"] for row in result["windows"]] == [
        "primary_complete", "fallback_base_complete", "no_complete_prediction"]
    assert [row["raw"] for row in result["windows"]] == [
        {"small": row["small"], "base": row["base"]} for row in original]
    assert result["canonical_phones"] == ["l", "ɐ", "l", "ɐ", "<UNK:2>",
                                           "n", "e", "i", "<UNK:completion>"]
    assert result["unknown_count"] == summary["unknown_count"] == 2
    assert result["jyutping"] == "La la HnG，\nnei\n<UNK:completion>"
    assert result["phone_times"] is None and result["tone"] == "unknown"
    assert result["timings_are_phone_durations"] is False
    assert result["repetition_cleanup"] is result["reference_used"] is result["default_model_changed"] is False
    for row in result["windows"]:
        assert row["time_scope"] == "model_window_only"
        assert all("duration" not in syllable and "start_sec" not in syllable
                   for syllable in row["projection"]["syllables"])
    assert result["xsampa_export"]["unmapped_label_counts"] == {"<UNK:2>": 1, "<UNK:completion>": 1}
    assert "[UNMAPPED:<UNK:2>]" in (output / "phones.xsampa.txt").read_text(encoding="utf-8")
    assert (output / "jyutping.txt").read_text(encoding="utf-8") == result["jyutping"] + "\n"
    assert (output / "phones.ipa.txt").read_text(encoding="utf-8") == result["ipa"] + "\n"
    assert summary["window_count"] == 3 and updates == [{"stage": "saving"}]
    assert fixture.read_bytes() == raw_bytes and windows == original


@pytest.mark.parametrize("windows", [
    [], [window(start=-.1, small=prediction())], [window(start=1, end=1)],
    [window(end=18.01)], [window(start=float("nan"))], [window(end=True)],
    [window(end=2), window(start=1, end=3)],
])
def test_save_predictions_rejects_bad_window_clocks_without_results(tmp_path, windows):
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError):
        tasks.save_predictions(output, windows, source={}, mode="replay", cancel=threading.Event())
    assert list(output.iterdir()) == []


def test_save_predictions_keeps_request_objects_and_touching_window_boundaries(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    windows = [window(start=8, end=26, small=prediction("la la")),
               window(start=26, end=26.1, small=prediction("nei"))]
    before = deepcopy(windows)
    result = tasks.save_predictions(output, windows, source={"offset": 8}, mode="replay",
                                     cancel=threading.Event())
    assert windows == before
    assert [(row["start_sec"], row["end_sec"]) for row in result["windows"]] == [(8, 26), (26, 26.1)]
    assert read(output / "result.json") == result


@pytest.mark.parametrize("decoded", [None, 42, ["nei"]])
def test_save_predictions_rejects_non_text_selected_decode(tmp_path, decoded):
    with pytest.raises(ValueError, match="decoded text"):
        tasks.save_predictions(tmp_path, [window(small=prediction(decoded))], source={},
                               mode="replay", cancel=threading.Event())
    assert not (tmp_path / "result.json").exists()


def test_replay_refuses_text_fixture_before_creating_output(tmp_path):
    fixture = tmp_path / "text.json"
    write(fixture, {"segments": [], "text": "nei"})
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="cantonese4_raw_v1"):
        tasks.replay({"fixture": str(fixture), "output_dir": str(output)}, threading.Event(), forbidden)
    assert not output.exists()


@pytest.mark.parametrize("action", ["cantonese4", "cantonese4_replay"])
def test_engine_precancel_has_no_output(audio, tmp_path, action):
    output = tmp_path / "out"
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        LocalEngine().execute(action, {"audio": str(audio), "output_dir": str(output)}, cancel, forbidden)
    assert not output.exists()


def test_replay_cancel_at_saving_writes_no_prediction_files(tmp_path):
    fixture = raw_fixture(tmp_path / "raw.json", [window(small=prediction())])
    cancel = threading.Event()
    output = tmp_path / "out"
    with pytest.raises(Cancelled):
        tasks.replay({"fixture": str(fixture), "output_dir": str(output)}, cancel, lambda _: cancel.set())
    assert not list(output.glob("*.json")) and not list(output.glob("*.txt"))


@pytest.mark.parametrize("action", ["replay", "run", "save_predictions"])
def test_existing_output_is_never_overwritten(assets, audio, tmp_path, action):
    fill_code(assets)
    output = tmp_path / "out"
    output.mkdir()
    sentinel = b"keep original result exactly\x00\n"
    (output / "result.json").write_bytes(sentinel)
    windows = [window(small=prediction())]
    fixture = raw_fixture(tmp_path / "raw.json", windows)
    with pytest.raises(FileExistsError):
        if action == "save_predictions":
            tasks.save_predictions(output, windows, source={}, mode="replay", cancel=threading.Event())
        else:
            params = {"output_dir": str(output)}
            params.update({"fixture": str(fixture)} if action == "replay" else
                          {**assets.params, "audio": str(audio)})
            getattr(tasks, action)(params, threading.Event(), forbidden)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == {"result.json": sentinel}


@pytest.mark.parametrize("outcome", ["ok", "bad_sha", "bad_schema", "cancel", "error"])
def test_run_checks_worker_result_identity_closes_and_never_retries(assets, audio, tmp_path, monkeypatch, outcome):
    fill_code(assets)
    output = tmp_path / "out"
    calls, closed, created = [], [], []

    class Worker:
        def __init__(self, python, log, **kwargs):
            created.append((python, log, kwargs))

        def call(self, action, params, cancel, progress, *, timeout):
            calls.append((action, params, timeout))
            if outcome == "error":
                raise RuntimeError("worker failed without retry")
            tasks.save_predictions(Path(params["_output"]), [window(small=prediction())],
                                   source={"sha256": sha(audio)}, mode="real", cancel=cancel)
            if outcome == "bad_schema":
                result = read(output / "result.json")
                result["schema_version"] = "wrong_schema"
                write(output / "result.json", result, replace=True)
            if outcome == "cancel":
                cancel.set()
            return {"result_sha256": "0" * 64 if outcome == "bad_sha" else sha(output / "result.json")}

        def close(self):
            closed.append(True)

    monkeypatch.setattr(process, "WorkerClient", Worker)
    params = {**assets.params, "audio": str(audio), "scope": "clip", "start": .5,
              "duration": 1, "output_dir": str(output)}
    if outcome == "ok":
        summary = LocalEngine().execute("cantonese4", params, threading.Event(), lambda _: None)
        assert summary["result_path"] == str(output / "result.json")
        assert summary["jyutping"] == "nei" and summary["window_count"] == 1
    else:
        error = Cancelled if outcome == "cancel" else RuntimeError if outcome == "error" else ValueError
        match = "cancelled" if outcome == "cancel" else "worker failed" if outcome == "error" else "identity mismatch"
        with pytest.raises(error, match=match):
            LocalEngine().execute("cantonese4", params, threading.Event(), lambda _: None)
    assert len(created) == len(calls) == 1 and closed == [True]
    assert created[0] == (str(assets.python), output / "worker.log",
                           {"runtime": False, "module": "any2xsampa.local.cantonese4_worker"})
    action, submitted, timeout = calls[0]
    assert action == "cantonese4" and timeout == 1800
    assert submitted["start"] == .5 and submitted["duration"] == 1
    assert submitted["_output"] == str(output)
    assert read(output / "request.json")["retry"] is False
    exit_data = read(output / "EXIT.json")
    expected = "completed" if outcome == "ok" else "cancelled" if outcome == "cancel" else "failed"
    assert exit_data["status"] == expected and exit_data["retried"] is False


def test_run_missing_bundle_never_starts_worker_or_prepares(assets, audio, tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "prepare", forbidden)
    output = tmp_path / "out"
    with pytest.raises(FileNotFoundError, match="not ready"):
        tasks.run({**assets.params, "audio": str(audio), "output_dir": str(output)}, threading.Event(), forbidden)
    assert not output.exists()


def test_prepare_explicit_engine_action_downloads_only_four_code_files(assets, monkeypatch):
    opener = FakeOpener(assets)
    built = []

    def build_opener(handler):
        built.append(handler)
        return opener

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    before = {path: (assets.bundle / path).read_bytes()
              for path in assets.payloads if path not in CODE_PATHS}
    assert tasks.status(assets.params)["reason"] == "separator_code_missing"
    assert opener.calls == built == []
    result = LocalEngine().execute("cantonese4_prepare", {**assets.params, "proxy": "http://127.0.0.1:10808"},
                                   threading.Event(), lambda _: None)
    assert result["available"] is True and result["model_loaded"] is False
    assert len(built) == 1 and built[0].proxies == {"http": "http://127.0.0.1:10808", "https": "http://127.0.0.1:10808"}
    assert [request.full_url for request, _ in opener.calls] == [
        row["url"] for row in assets.rows if row["path"] in CODE_PATHS]
    for request, timeout in opener.calls:
        assert timeout == 30
        assert request.get_header("Accept-encoding") == "identity"
        assert request.get_header("Range") is None
    for path in CODE_PATHS:
        assert (assets.bundle / path).read_bytes() == assets.payloads[path]
    assert before == {path: (assets.bundle / path).read_bytes() for path in before}


def test_prepare_correct_files_use_zero_network_even_without_opener(assets):
    fill_code(assets)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in assets.bundle.rglob("*") if p.is_file()}
    assert tasks.prepare(assets.params, threading.Event(), forbidden)["available"] is True
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


@pytest.mark.parametrize("bad", ["sha", "short", "long", "status", "redirect"])
def test_prepare_rejects_bad_download_without_replacing_target_or_retrying(assets, bad):
    path = CODE_PATHS[0]
    target = assets.bundle / path
    target.parent.mkdir(parents=True)
    old = b"old source must survive"
    target.write_bytes(old)
    data = assets.payloads[path]
    response = {
        "sha": lambda: Response(b"X" * len(data)),
        "short": lambda: Response(data[:-1]),
        "long": lambda: Response(data + b"X", headers={}),
        "status": lambda: Response(data, status=503),
        "redirect": lambda: Response(data, url="http://cdn.example.invalid/source"),
    }[bad]()
    opener = FakeOpener(assets, response)
    with pytest.raises(ValueError, match="SHA256|length|size|status|HTTPS"):
        tasks.prepare(assets.params, threading.Event(), lambda _: None, opener=opener)
    assert len(opener.calls) == 1 and opener.calls[0][1] == 30
    assert target.read_bytes() == old
    assert not list(assets.bundle.rglob("*.installing"))
    assert not any((assets.bundle / name).exists() for name in CODE_PATHS[1:])


def test_prepare_rechecks_sha_for_existing_same_size_code(assets):
    fill_code(assets)
    target = assets.bundle / CODE_PATHS[2]
    target.write_bytes(b"X" * len(assets.payloads[CODE_PATHS[2]]))
    opener = FakeOpener(assets)
    assert tasks.prepare(assets.params, threading.Event(), lambda _: None, opener=opener)["available"]
    assert len(opener.calls) == 1
    assert target.read_bytes() == assets.payloads[CODE_PATHS[2]]


def test_prepare_missing_weights_refuses_without_network(assets):
    (assets.bundle / "separator/MelBandRoformer.ckpt").unlink()
    opener = FakeOpener(assets)
    with pytest.raises(FileNotFoundError, match="extracted Cantonese model bundle"):
        tasks.prepare(assets.params, threading.Event(), forbidden, opener=opener)
    assert opener.calls == []
    assert not any((assets.bundle / name).exists() for name in CODE_PATHS)


@pytest.mark.parametrize("proxy", ["socks5://127.0.0.1:1080", "http://", "not a URL"])
def test_prepare_bad_proxy_never_contacts_upstream(assets, proxy):
    opener = FakeOpener(assets)
    with pytest.raises(ValueError, match="Proxy"):
        tasks.prepare({**assets.params, "proxy": proxy}, threading.Event(), forbidden, opener=opener)
    assert opener.calls == []


@pytest.mark.parametrize("during_download", [False, True])
def test_prepare_cancellation_stops_before_install_and_next_request(assets, during_download):
    cancel = threading.Event()
    if not during_download:
        cancel.set()
    opener = FakeOpener(assets)
    with pytest.raises(Cancelled):
        tasks.prepare(assets.params, cancel, lambda _: cancel.set(), opener=opener)
    assert len(opener.calls) == int(during_download)
    assert not any((assets.bundle / path).exists() for path in CODE_PATHS)


@pytest.mark.parametrize("action,extra", [
    ("cantonese4_status", {"load_model": True}),
    ("cantonese4_prepare", {"download_weights": True}),
    ("cantonese4_replay", {"lyrics": "not evidence"}),
])
def test_candidate_actions_reject_extra_conditions_before_side_effects(tmp_path, action, extra):
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="accepts"):
        LocalEngine().execute(action, extra, threading.Event(), forbidden)
    assert not output.exists()


@pytest.mark.parametrize("argv,action,expected", [
    (["cantonese4-status", "--bundle-dir", "bundle"], "cantonese4_status", {"bundle_dir": "bundle"}),
    (["cantonese4-prepare", "--bundle-dir", "bundle", "--proxy", "http://proxy:80"], "cantonese4_prepare",
     {"bundle_dir": "bundle", "proxy": "http://proxy:80"}),
    (["cantonese4-replay", "raw.json", "--output-dir", "out"], "cantonese4_replay",
     {"fixture": "raw.json", "output_dir": "out"}),
    (["cantonese4", "a.wav", "--scope", "clip", "--start", "1.25", "--duration", "0.5"], "cantonese4",
     {"audio": "a.wav", "scope": "clip", "start": 1.25, "duration": .5}),
    (["cantonese4", "a.wav"], "cantonese4",
     {"audio": "a.wav", "scope": "whole_file", "start": 0, "duration": 8}),
])
def test_cli_maps_candidate_commands_without_inference(argv, action, expected, monkeypatch, capsys):
    from any2xsampa import cli as entrypoint

    calls, closed = [], []

    class Engine:
        def execute(self, received, params, cancel, progress):
            calls.append((received, params))
            assert not cancel.is_set()
            return {"mock": True}

        def close(self):
            closed.append(True)

    monkeypatch.setattr(cli, "LocalEngine", Engine)
    assert entrypoint.main(argv) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"mock": True} and captured.err == ""
    assert calls == [(action, expected)] and closed == [True]
