from __future__ import annotations
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
import wave
import pytest
from any2xsampa.local.common import VERSION, Cancelled, crop, request, sha, write
from any2xsampa.local.engine import LocalEngine
from any2xsampa.local.process import WorkerClient
from any2xsampa.local.server import Server

@pytest.fixture
def wav(tmp_path):
    p = tmp_path / "中文 空格.wav"
    with wave.open(str(p), "wb") as f:
        f.setparams((1,2,16000,0,"NONE","not compressed"))
        f.writeframes(b"\0\0"*32000)
    return p

@pytest.fixture
def fixture(tmp_path):
    p = tmp_path / "phones.json"
    write(p, {"segments": [{"start_sec": 0.0, "end_sec": .1, "confidence": .9,
         "candidate_phones": [{"symbol": "i", "confidence": .9}]}]})
    return p

def test_no_conditions_and_no_implicit_fixtures():
    for forbidden in ("lyrics", "vsqx", "pitch", "fixture", "recognizer_mode"):
        with pytest.raises(ValueError):
            request({"audio":"x.wav", "model":"a", forbidden: "not permitted"})
    with pytest.raises(ValueError): request({"audio":"x.wav", "model":"a", "device":"auto"})
    for value in (float("nan"), float("inf"), True, -1, 21):
        with pytest.raises(ValueError): request({"audio":"x", "model":"a", "duration":value})

def test_crop_exact_and_source_unchanged(wav,tmp_path):
    before = sha(wav)
    info = crop(wav,.5,1,tmp_path/"crop.wav")
    assert info["source_start_sample"] == 8000
    assert info["source_end_sample_exclusive"] == 24000
    assert sha(wav)==before
    with pytest.raises(ValueError): crop(wav,1.9,1,tmp_path/"bad.wav")

def test_replay_draft_export_and_replay_equality(wav,fixture,tmp_path):
    engine=LocalEngine()
    r=engine.execute("replay", {"fixture":str(fixture),"audio":str(wav),"start":.5,"duration":1,
                              "output_dir":str(tmp_path/"result")}, threading.Event(), lambda _:None)
    file=Path(r["result_path"]); p=json.loads(file.read_text(encoding="utf8"))
    assert p["mode"]=="replay" and p["segments"][0]["start_sec"]==.5
    assert p["segments"][0]["normalized_ipa"]=="i"
    original=file.read_bytes()
    with pytest.raises(ValueError):
        engine.draft({"result_path":str(file),"result_sha256":"wrong","edits":[]})
    engine.draft({"result_path":str(file),"result_sha256":sha(file),"edits":[{"index":0,"ipa":"UNKNOWN"}]})
    engine.export({"result_path":str(file),"format":"xsampa","output":str(tmp_path/"export.txt"),"use_draft":True})
    assert "UNMAPPED:UNKNOWN" in (tmp_path/"export.txt").read_text()
    assert file.read_bytes()==original
    second=engine.execute("replay",{"fixture":str(file),"output_dir":str(tmp_path/"second")},threading.Event(),lambda _:None)
    assert json.loads(Path(second["result_path"]).read_text(encoding="utf8"))["segments"]==p["segments"]
    with pytest.raises(FileExistsError):
        engine.execute("replay",{"fixture":str(file),"output_dir":str(tmp_path/"second")},threading.Event(),lambda _:None)

def test_out_of_crop_fixture_rejected(wav,fixture,tmp_path):
    p=json.loads(fixture.read_text());p["segments"][0]["end_sec"]=2
    fixture.write_text(json.dumps(p))
    with pytest.raises(ValueError,match="outside"):
        LocalEngine().execute("replay",{"fixture":str(fixture),"audio":str(wav),"duration":1,"output_dir":str(tmp_path/"out")},threading.Event(),lambda _:None)
    assert json.loads((tmp_path/"out/EXIT.json").read_text())["status"]=="failed"

def test_server_version_and_duplicate():
    stream=io.StringIO();s=Server(LocalEngine(),stream)
    with pytest.raises(ValueError):s.receive({"protocol":"bad"})
    cmd={"protocol":VERSION,"id":"x","command":"capabilities"}
    s.receive(cmd)
    with pytest.raises(ValueError,match="Duplicate"):s.receive(cmd)
    s.close()
    assert json.loads(stream.getvalue())["data"]["requires_server_port"] is False

def test_running_and_queued_cancel():
    entered=threading.Event()
    class E:
        def execute(self, action, p, cancel, progress):
            entered.set()
            cancel.wait(3)
            if cancel.is_set():raise Cancelled("cancelled")
            return {}
        def close(self):pass
    stream=io.StringIO();s=Server(E(),stream)
    def send(id,command,**kw):s.receive({"protocol":VERSION,"id":id,"command":command,**kw})
    send("one","job");assert entered.wait(1)
    send("two","job");send("cancel2","cancel",target="two");send("cancel1","cancel",target="one")
    s.close();events=[json.loads(x) for x in stream.getvalue().splitlines()]
    assert {x["id"] for x in events if x["event"]=="cancelled"}=={"one","two"}

def test_stdio_owned_process_roundtrip(wav,fixture,tmp_path):
    worker=WorkerClient(sys.executable,tmp_path/"worker.log",runtime=False)
    pid=worker.process.pid
    try:
        assert worker.call("capabilities",{},threading.Event(),lambda _:None)["transport"]=="stdio"
        result=worker.call("replay",{"fixture":str(fixture),"audio":str(wav),"duration":1,"output_dir":str(tmp_path/"out")},threading.Event(),lambda _:None)
        assert result["mode"]=="replay"
        assert worker.process.pid==pid
        assert worker.call("unload",{},threading.Event(),lambda _:None)["unloaded"]
    finally:worker.close()
    assert worker.process.poll() is not None

def test_model_missing_cannot_return_fixture(wav,tmp_path):
    with pytest.raises((FileNotFoundError,ValueError)):
        LocalEngine().execute("recognize",{"audio":str(wav),"model":"definitely-not-configured","duration":1,"output_dir":str(tmp_path/"out")},threading.Event(),lambda _:None)
    assert not (tmp_path/"out/result.json").exists()

def test_precancel_has_no_artifacts(wav,tmp_path):
    cancelled=threading.Event();cancelled.set()
    with pytest.raises(Cancelled):LocalEngine().execute("recognize",{},cancelled,lambda _:None)

def test_registry_hash_drift_rejected(tmp_path,monkeypatch):
    from any2xsampa.local import registry
    file=tmp_path/"model.json";write(file,{"schema_version":"local_model_package_v1"})
    r=tmp_path/"registry.json";write(r,{"schema_version":"local_model_registry_v1","models":[{"id":"model","package":"model.json","sha256":sha(file)}]})
    monkeypatch.setenv("ANY2XSAMPA_MODELS",str(r))
    file.write_text("{}")
    assert registry.models()[0]["available"] is False
    with pytest.raises(ValueError,match="hash"):registry.package("model")


def test_runtime_reuses_model_and_releases_lock(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from any2xsampa.local import runtime
    from any2xsampa.adapters import candidate_inference, phoneticxeus_runtime
    counts={"loads":0,"acquired":0,"released":0}
    @contextmanager
    def lock():
        counts["acquired"]+=1
        try:yield
        finally:counts["released"]+=1
    monkeypatch.setattr(candidate_inference,"serial_candidate_job",lock)
    monkeypatch.setattr(phoneticxeus_runtime,"enforce_offline",lambda _:None)
    monkeypatch.setattr(runtime,"package",lambda _:dict(python=sys.executable,devices=["cpu"],package_sha256="x",backend="onnx-espeak",id="test",quality="fixture-for-test-only",paths={}))
    worker=runtime.RuntimeEngine()
    def load(*_):counts["loads"]+=1
    monkeypatch.setattr(worker,"_load",load)
    monkeypatch.setattr(worker,"_onnx",lambda _:{"segments":[],"time_basis":"crop"})
    for i in range(2):
        p=tmp_path/str(i);p.mkdir()
        r=worker.execute("recognize",{"model":"test","audio":"not-read-by-mock.wav","device":"cpu","_output":str(p)},threading.Event(),lambda _:None)
        assert set(r)=={"evidence_path","evidence_sha256"}
        assert json.loads(Path(r["evidence_path"]).read_text())["load_count"]==1
    assert counts=={"loads":1,"acquired":1,"released":0}
    worker.close()
    assert counts["released"]==1

def test_real_path_requires_hash_bound_small_message(wav,tmp_path,monkeypatch):
    from any2xsampa.local import engine as mod
    class W:
        def __init__(self,*_):self.process=type("P",(),{"poll":lambda self:None})()
        def call(self,*_):return {"evidence_path":str(tmp_path/"foreign.json"),"evidence_sha256":"bad"}
        def close(self):pass
    monkeypatch.setattr(mod,"package",lambda _:dict(package_sha256="x",python=sys.executable))
    monkeypatch.setattr(mod,"WorkerClient",W)
    with pytest.raises(ValueError,match="evidence"):
        mod.LocalEngine().execute("recognize",{"audio":str(wav),"model":"x","duration":1,"output_dir":str(tmp_path/"out")},threading.Event(),lambda _:None)

def test_replay_tampered_source_rejected(wav,fixture,tmp_path):
    engine=LocalEngine()
    r=engine.execute("replay",{"audio":str(wav),"fixture":str(fixture),"duration":1,"output_dir":str(tmp_path/"first")},threading.Event(),lambda _:None)
    b=bytearray(wav.read_bytes());b[-1]=1;wav.write_bytes(b)
    with pytest.raises(ValueError,match="hash"):
        engine.execute("replay",{"fixture":r["result_path"],"output_dir":str(tmp_path/"second")},threading.Event(),lambda _:None)


def test_pipe_reader_allows_native_import_while_stdin_is_open():
    # Regression for the Windows CRT stdin-lock/native DLL import deadlock.
    script = "import threading,sys,os;from any2xsampa.local.server import input_lines\ndef native():\n import numpy\n os._exit(0)\nthreading.Thread(target=native).start()\nfor line in input_lines(sys.stdin): pass"
    p=subprocess.Popen([sys.executable,"-X","utf8","-c",script],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        assert p.wait(timeout=20)==0
    finally:
        if p.poll() is None:p.kill();p.wait()
        p.stdin.close();p.stdout.close();p.stderr.close()

def test_fragmented_unicode_protocol_bytes():
    p=subprocess.Popen([sys.executable,"-X","utf8","-m","any2xsampa.local.server"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        assert json.loads(p.stdout.readline())["event"]=="ready"
        data=(json.dumps({"protocol":VERSION,"id":"中文-id","command":"capabilities"},ensure_ascii=False)+"\n").encode()
        for b in data:
            p.stdin.write(bytes([b]));p.stdin.flush()
        message=json.loads(p.stdout.readline())
        assert message["id"]=="中文-id" and message["event"]=="result"
        p.stdin.close();assert p.wait(timeout=10)==0
    finally:
        if p.poll() is None:p.kill();p.wait()
        if not p.stdin.closed:p.stdin.close()
        p.stdout.close();p.stderr.close()
