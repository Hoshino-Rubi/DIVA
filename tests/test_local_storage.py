import hashlib
import json
import threading

import pytest

from any2xsampa.local.model_install import install
from any2xsampa.local.storage import data_home, model_cache, state_dir


def test_legacy_and_explicit_portable_state(tmp_path, monkeypatch):
    monkeypatch.delenv("ANY2XSAMPA_DATA_HOME", raising=False)
    assert state_dir(tmp_path) == tmp_path / ".state"
    assert model_cache(tmp_path) == tmp_path / ".state/model-downloads"
    with pytest.raises(ValueError, match="absolute"):
        monkeypatch.setenv("ANY2XSAMPA_DATA_HOME", "relative")
        data_home(tmp_path)


def test_versions_share_state_not_program_directories(tmp_path, monkeypatch):
    monkeypatch.delenv("ANY2XSAMPA_DATA_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "profile"))
    for version in ("v1.0.1", "v1.0.2"):
        root = tmp_path / version
        root.mkdir()
        (root / "local-app.json").write_text(json.dumps({"storage_mode": "per_user_v1"}))
    assert state_dir(tmp_path / "v1.0.1") == state_dir(tmp_path / "v1.0.2")
    assert not state_dir(tmp_path / "v1.0.1").is_relative_to(tmp_path / "v1.0.1")


def test_shared_verified_blob_prevents_repeat_download(tmp_path, monkeypatch):
    from test_model_install import Opener, Response, setup
    monkeypatch.setenv("ANY2XSAMPA_DATA_HOME", str(tmp_path / "user-data"))
    first, second = tmp_path / "v1", tmp_path / "v2"
    first.mkdir(); second.mkdir()
    setup(first); setup(second)
    opener = Opener(Response(b"model weights"))
    install({}, threading.Event(), lambda _: None, first, opener=opener)
    install({}, threading.Event(), lambda _: None, second, opener=opener)
    assert len(opener.calls) == 1
    assert (first / "models/test/weights").read_bytes() == b"model weights"
    assert (second / "models/test/weights").read_bytes() == b"model weights"
    blob = model_cache(first) / (hashlib.sha256(b"model weights").hexdigest() + ".blob")
    assert blob.is_file()
    assert not (first / ".state").exists()


def test_invalid_storage_config_fails_closed(tmp_path, monkeypatch):
    monkeypatch.delenv("ANY2XSAMPA_DATA_HOME", raising=False)
    (tmp_path / "local-app.json").write_text('{"storage_mode":"guess"}')
    with pytest.raises(ValueError, match="Unsupported"):
        data_home(tmp_path)
