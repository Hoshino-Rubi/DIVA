import hashlib
import io
import json
import threading
import zipfile
from pathlib import Path

import pytest

from any2xsampa.local.common import Cancelled
from any2xsampa.local.model_install import catalog, install, inside, status


def entry(data, **extra):
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), **extra)


def setup(tmp_path, data=b"model weights"):
    target = "models/test/weights"
    download = entry(data, url="https://example.invalid/model", target=target, offline_name="weights")
    model = dict(id="test", name="test", license="test", files=[entry(data, path=target)], downloads=[download])
    spec = dict(schema_version="local_model_downloads_v1", auto_download=True, bundles=[model])
    (tmp_path/"model-downloads.json").write_text(json.dumps(spec))
    return spec, download


class Response(io.BytesIO):
    def __init__(self, data, code=200, **headers):
        super().__init__(data)
        self.status = code
        self.headers = headers
    def geturl(self):
        return "https://cdn.example.invalid/blob"


class Opener:
    def __init__(self, response):
        self.response = response
        self.calls = []
    def open(self, request, timeout):
        self.calls.append(request)
        return self.response


def test_download_and_no_duplicate_network(tmp_path):
    _, row = setup(tmp_path)
    op = Opener(Response(b"model weights"))
    updates = []
    result = install({}, threading.Event(), updates.append, tmp_path, opener=op)
    assert result["ready"] and len(op.calls) == 1
    install({}, threading.Event(), updates.append, tmp_path, opener=op)
    assert len(op.calls) == 1
    assert updates[-1]["stage"] == "model_ready"


def test_resume_requires_exact_range(tmp_path):
    _, row = setup(tmp_path)
    cache = tmp_path/".state/model-downloads"
    cache.mkdir(parents=True)
    (cache/(row["sha256"]+".part")).write_bytes(b"model ")
    op = Opener(Response(b"weights", 206, **{"Content-Range":"bytes 6-12/13", "Content-Length":"7"}))
    assert install({}, threading.Event(), lambda _:None, tmp_path, opener=op)["ready"]
    assert op.calls[0].get_header("Range") == "bytes=6-"


def test_range_refusal_does_not_discard_partial_or_retry(tmp_path):
    _, row = setup(tmp_path)
    cache = tmp_path/".state/model-downloads"
    cache.mkdir(parents=True)
    part=cache/(row["sha256"]+".part")
    part.write_bytes(b"model ")
    op=Opener(Response(b"model weights"))
    with pytest.raises(ValueError, match="resume"):
        install({}, threading.Event(), lambda _:None, tmp_path, opener=op)
    assert part.read_bytes()==b"model " and len(op.calls)==1


def test_corrupt_download_does_not_replace_existing(tmp_path):
    _, row = setup(tmp_path)
    old=tmp_path/row["target"]
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old")
    with pytest.raises(ValueError, match="mismatch"):
        install({}, threading.Event(), lambda _:None, tmp_path, opener=Opener(Response(b"x"*13)))
    assert old.read_bytes()==b"old"


def test_cancellation_keeps_partial(tmp_path):
    setup(tmp_path)
    cancel=threading.Event()
    cancel.set()
    op=Opener(Response(b"model weights"))
    with pytest.raises(Cancelled):
        install({}, cancel, lambda _:None, tmp_path, opener=op)
    assert not op.calls


def test_offline_parts_and_validation(tmp_path):
    spec, row=setup(tmp_path)
    row["offline_parts"]=[entry(b"model ", name="part1"), entry(b"weights", name="part2")]
    (tmp_path/"model-downloads.json").write_text(json.dumps(spec))
    src=tmp_path/"offline"
    src.mkdir()
    (src/"part1").write_bytes(b"model ")
    (src/"part2").write_bytes(b"weights")
    op=Opener(None)
    assert install({"offline_dir":str(src)}, threading.Event(), lambda _:None, tmp_path, opener=op)["ready"]
    assert not op.calls


def test_corrupt_offline_part_no_install(tmp_path):
    spec,row=setup(tmp_path)
    row["offline_parts"]=[entry(b"model weights", name="part1")]
    (tmp_path/"model-downloads.json").write_text(json.dumps(spec))
    src=tmp_path/"offline";src.mkdir();(src/"part1").write_bytes(b"x"*13)
    with pytest.raises(ValueError,match="hash mismatch"):
        install({"offline_dir":str(src)}, threading.Event(), lambda _:None,tmp_path)
    assert not (tmp_path/row["target"]).exists()


@pytest.mark.parametrize("name", ["../other", "models/../../other", "C:/outside", "/outside", "models\\x"])
def test_escape_rejected(tmp_path,name):
    with pytest.raises(ValueError):
        inside(tmp_path,name)


def test_bad_catalog_url_and_target(tmp_path):
    spec,row=setup(tmp_path)
    row["url"]="http://example.invalid/insecure"
    (tmp_path/"model-downloads.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError,match="HTTPS"):
        catalog(tmp_path)
    row["url"]="https://example.invalid/model"
    spec["bundles"][0]["files"][0]["path"]="src/evil.py"
    (tmp_path/"model-downloads.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError,match="non-model"):
        catalog(tmp_path)


@pytest.mark.parametrize("extra", [None, "../escape"])
def test_archive_extract_and_escape(tmp_path,extra):
    contents=io.BytesIO()
    with zipfile.ZipFile(contents,"w") as z:
        z.writestr("model/encoder.onnx",b"graph")
        if extra:z.writestr(extra,b"bad")
    data=contents.getvalue()
    spec,row=setup(tmp_path,b"graph")
    row.clear()
    row.update(entry(data,url="https://example.invalid/model.zip",offline_name="game.zip",
                     members=[dict(archive_path="model/encoder.onnx",target="models/test/weights")]))
    (tmp_path/"model-downloads.json").write_text(json.dumps(spec))
    op=Opener(Response(data))
    if extra:
        with pytest.raises(ValueError):
            install({},threading.Event(),lambda _:None,tmp_path,opener=op)
        assert not (tmp_path/"models/test/weights").exists()
    else:
        assert install({},threading.Event(),lambda _:None,tmp_path,opener=op)["ready"]


def test_no_catalog_does_not_enable_downloads(tmp_path):
    assert status(tmp_path)==dict(enabled=False,ready=False,auto_download=False,bundles=[])


def test_missing_bundled_metadata_is_not_forged(tmp_path):
    spec,row=setup(tmp_path)
    spec["bundles"][0]["files"].append(entry(b"lock",path="models/test/lock.json"))
    (tmp_path/"model-downloads.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError,match="metadata damaged"):
        install({},threading.Event(),lambda _:None,tmp_path,opener=Opener(Response(b"model weights")))
    assert not status(tmp_path)["ready"]

def test_midstream_cancel_resumes_on_next_explicit_call(tmp_path):
    import any2xsampa.local.model_install as module
    _, row=setup(tmp_path)
    cancel=threading.Event()
    class SmallResponse(Response):
        def read(self,size=-1):
            return super().read(min(size,6))
    def stop(e):
        if e.get("downloaded_bytes",0)>=6:cancel.set()
    op=Opener(SmallResponse(b"model weights"))
    with pytest.raises(Cancelled):
        install({},cancel,stop,tmp_path,opener=op)
    part=tmp_path/".state/model-downloads"/(row["sha256"]+".part")
    assert part.read_bytes()==b"model "
    assert not (tmp_path/row["target"]).exists()
    cancel.clear()
    op=Opener(Response(b"weights",206,**{"Content-Range":"bytes 6-12/13"}))
    assert install({},cancel,lambda _:None,tmp_path,opener=op)["ready"]


def test_bad_download_length_stops_before_replace(tmp_path):
    setup(tmp_path)
    with pytest.raises(ValueError,match="length"):
        install({},threading.Event(),lambda _:None,tmp_path,opener=Opener(Response(b"abc",200,**{"Content-Length":"3"})))


def test_failed_process_lock_is_released(tmp_path):
    from any2xsampa.local.model_install import install_lock
    with pytest.raises(RuntimeError):
        with install_lock(tmp_path):
            raise RuntimeError("failure")
    with install_lock(tmp_path):
        pass
