from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys

import pytest

from any2xsampa.training.limited_data_artifacts import sha256_object
from any2xsampa.training.limited_data_reference import SPLITS
from tools.run_phoneticxeus_worker import validate_requests


ROOT = Path(__file__).resolve().parents[1]


def request() -> dict:
    return {
        "schema_version": "phoneticxeus_audio_requests_v1", "purpose": "smoke8",
        "source_dataset_hash": "a" * 64, "source_manifest_file_sha256": "b" * 64,
        "split_sha256": sha256_object(SPLITS), "contains_labels_or_score_context": False,
        "rows": [{
            "example_id": "freedom-flower-smoke8", "song_id": "freedom-flower",
            "split": "train", "audio_role": "original",
            "audio_path": "source.wav", "audio_sha256": "c" * 64, "start_sec": 10.0, "end_sec": 18.0,
        }],
    }


@pytest.mark.parametrize("field", ["lyrics", "phones", "pitch", "score_path"])
def test_worker_rejects_side_context_at_either_level(field: str) -> None:
    value = request()
    validate_requests(value)
    value[field] = "forbidden"
    with pytest.raises(ValueError, match="unexpected request"):
        validate_requests(value)
    value = request()
    value["rows"][0][field] = "forbidden"
    with pytest.raises(ValueError, match="unexpected acoustic"):
        validate_requests(value)


@pytest.mark.parametrize("change", [
    {"start_sec": float("nan")}, {"end_sec": 50},
    {"start_sec": True}, {"audio_sha256": "wrong"}, {"example_id": "../escape"},
    {"song_id": "king-of-karaoke", "split": "train"},
])
def test_worker_rejects_invalid_audio_requests_before_model_load(change: dict) -> None:
    value = request()
    value["rows"][0].update(change)
    with pytest.raises(ValueError):
        validate_requests(value)


def test_test_song_cannot_be_disguised_as_training_features() -> None:
    value = request()
    value["purpose"] = "rendered-features"
    value["rows"][0].update(song_id="king-of-karaoke", split="test", audio_role="rendered")
    with pytest.raises(ValueError, match="test/original"):
        validate_requests(value)


def run_without_site(code: str, *args: str, utf8: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-X", f"utf8={int(utf8)}", "-c", code, str(ROOT), *args],
        capture_output=True, text=True, timeout=30, check=False,
    )


def test_default_fixture_survives_without_optional_model_dependencies(tmp_path: Path) -> None:
    code = (
        "import sys,wave; from pathlib import Path; root=Path(sys.argv[1]); sys.path.insert(0,str(root/'src'));"
        "from any2xsampa.adapters.recognizer import RecognizerConfig,FixtureUniversalPhoneRecognizer;"
        "from any2xsampa.adapters.phoneticxeus import PhoneticXeusPosteriorRecognizer;"
        "p=Path(sys.argv[2]); w=wave.open(str(p),'wb');w.setparams((1,2,16000,0,'NONE','none'));w.writeframes(bytes(64000));w.close();"
        "r=FixtureUniversalPhoneRecognizer(root/'src/any2xsampa/recognizer_fixtures/default_fixture.json');"
        "assert RecognizerConfig().mode=='fixture';assert len(r.recognize(p))==4;"
        "assert not {'torch','transformers','numpy','onnxruntime'} & sys.modules.keys();print('stdlib_fixture_ok')"
    )
    result = run_without_site(code, str(tmp_path / "fixture.wav"))
    assert result.returncode == 0, result.stderr
    assert "stdlib_fixture_ok" in result.stdout


def test_offline_worker_blocks_dns_tcp_and_udp_without_network(tmp_path: Path) -> None:
    code = """
import sys,socket
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/'src'))
from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline
enforce_offline(Path(sys.argv[2]))
attempts=[lambda: socket.getaddrinfo('example.invalid',443),
          lambda: socket.create_connection(('example.invalid',443)),
          lambda: socket.socket().connect(('127.0.0.1',1)),
          lambda: socket.socket(type=socket.SOCK_DGRAM).sendto(b'x',('127.0.0.1',1))]
for attempt in attempts:
    try:
        attempt()
    except RuntimeError as exc:
        assert 'network access forbidden' in str(exc)
    else:
        raise AssertionError('network operation was not blocked')
print('offline_guard_ok')
"""
    result = run_without_site(code, str(tmp_path / "cache"))
    assert result.returncode == 0, result.stderr
    assert "offline_guard_ok" in result.stdout


def test_utf8_guard_precedes_snapshot_or_model_access() -> None:
    code = """
import sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/'src'))
from any2xsampa.adapters.phoneticxeus_runtime import PhoneticXeusRuntime
try:
    PhoneticXeusRuntime(Path('nonexistent-snapshot'))
except RuntimeError as exc:
    assert '-X utf8' in str(exc)
else:
    raise AssertionError('missing UTF-8 guard')
assert 'torch' not in sys.modules
print('utf8_guard_ok')
"""
    result = run_without_site(code, utf8=False)
    assert result.returncode == 0, result.stderr
    assert "utf8_guard_ok" in result.stdout
