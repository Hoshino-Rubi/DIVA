import hashlib
import importlib.util
import io
from pathlib import Path
import zipfile

import pytest


def tool():
    path = Path(__file__).parents[1] / "tools/stream_portable_release.py"
    spec = importlib.util.spec_from_file_location("stream_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload(tmp_path):
    source, output = tmp_path/"Application", tmp_path/"assets"
    source.mkdir(); output.mkdir()
    (source/"runtime.bin").write_bytes(bytes(range(256))*10)
    (source/"说明.txt").write_text("relocated input", encoding="utf8")
    (source/".state").mkdir(); (source/".state/private.json").write_text("excluded")
    return source, output


def test_streamed_volumes_form_valid_archive_without_full_copy(tmp_path):
    module = tool()
    source, output = payload(tmp_path)
    result = module.stream_release(source, output, "v1.0.1", part_bytes=256,
                                   max_archive_bytes=4096, reserve_bytes=0)
    parts = result["parts"]
    assert len(parts) > 1 and all(p["bytes"] <= 256 for p in parts)
    joined = b"".join((output/p["name"]).read_bytes() for p in parts)
    assert hashlib.sha256(joined).hexdigest() == result["archive_sha256"]
    assert not (output/"Application.zip").exists()
    with zipfile.ZipFile(io.BytesIO(joined)) as archive:
        assert archive.testzip() is None and len(archive.infolist()) == 2
        assert archive.read("Application/runtime.bin") == (source/"runtime.bin").read_bytes()
    with module.VolumeReader(output, parts) as reader:
        reader.seek(253)
        assert reader.read(600) == joined[253:853]
        reader.seek(-20, 2)
        assert reader.read() == joined[-20:]
    with pytest.raises(FileExistsError):
        module.stream_release(source, output, "v1.0.1", reserve_bytes=0)


def test_disk_cap_leaves_no_success_manifest(tmp_path):
    source, output = payload(tmp_path)
    with pytest.raises(OSError, match="disk budget"):
        tool().stream_release(source, output, "v1.0.1", part_bytes=100,
                              max_archive_bytes=150, reserve_bytes=0)
    assert not (output/"release-manifest.json").exists()


def test_reserved_space_fails_before_writing(tmp_path, monkeypatch):
    module = tool()
    source, output = payload(tmp_path)
    class Disk:
        free = 200
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: Disk())
    with pytest.raises(OSError, match="Insufficient space"):
        module.stream_release(source, output, "v1.0.1", max_archive_bytes=100, reserve_bytes=101)
    assert not list(output.iterdir())


def test_truncated_volume_is_detected(tmp_path):
    module = tool()
    (tmp_path/"part").write_bytes(b"a")
    with module.VolumeReader(tmp_path, [dict(name="part", bytes=4)]) as reader:
        with pytest.raises(OSError, match="truncated"):
            reader.read()


def test_output_cannot_become_part_of_payload(tmp_path):
    source, _ = payload(tmp_path)
    inside = source/"assets"; inside.mkdir()
    with pytest.raises(ValueError, match="outside"):
        tool().stream_release(source, inside, "v1.0.1", reserve_bytes=0)


def test_invalid_tag_writes_nothing(tmp_path):
    source, output = payload(tmp_path)
    with pytest.raises(ValueError, match="stable release tag"):
        tool().stream_release(source, output, "../latest", reserve_bytes=0)
    assert not list(output.iterdir())
