"""Download transport regressions using fake openers only."""
import hashlib
import io
import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error

from any2xsampa.local.model_install import _download, status


DATA = b"model weights"


class Response(io.BytesIO):
    status = 200
    headers = {}

    def __init__(self, error=None):
        super().__init__(DATA)
        self.error = error

    def geturl(self):
        return "https://example.invalid/model"

    def read(self, size=-1):
        if self.error is not None and self.tell() >= 6:
            raise self.error
        return super().read(min(size, 6))


class Opener:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        if self.error is not None:
            raise self.error
        return self.response


class DownloadTimeoutTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(directory.cleanup)
        self.cache = Path(directory.name)
        self.item = {
            "bytes": len(DATA), "sha256": hashlib.sha256(DATA).hexdigest(),
            "url": "https://example.invalid/model", "offline_name": "weights",
        }
        self.part = self.cache / (self.item["sha256"] + ".part")

    def download(self, opener):
        return _download(self.item, self.cache, threading.Event(), lambda _: None, opener)

    def test_network_failure_is_bounded_preserves_partial_and_does_not_retry(self):
        errors = [
            (socket.timeout("timed out"), "timed out.*30-second"),
            (urllib.error.URLError(socket.timeout("timed out")), "timed out.*30-second"),
            (socket.gaierror(-2, "Name resolution failed"), "network connection error"),
            (ConnectionResetError("Connection reset"), "network connection error"),
            (urllib.error.URLError("untrusted details " * 1000), "network connection error"),
            (urllib.error.HTTPError("https://example.invalid", 503, "Unavailable", {}, None), "HTTP 503"),
        ]
        for phase in ("open", "read"):
            for error, message in errors:
                with self.subTest(phase=phase, error=type(error).__name__):
                    self.part.unlink(missing_ok=True)
                    if phase == "open":
                        self.part.write_bytes(DATA[:6])
                        opener = Opener(error=error)
                    else:
                        opener = Opener(response=Response(error))
                    with self.assertRaisesRegex(RuntimeError, message) as caught:
                        self.download(opener)
                    self.assertIs(caught.exception.__cause__, error)
                    self.assertLess(len(str(caught.exception)), 240)
                    self.assertIn("connection or proxy", str(caught.exception))
                    self.assertIn("Partial download retained", str(caught.exception))
                    self.assertEqual(self.part.read_bytes(), DATA[:6])
                    self.assertEqual(len(opener.calls), 1)
                    self.assertEqual(opener.calls[0][1], 30)
                    self.assertFalse((self.cache / (self.item["sha256"] + ".blob")).exists())

    def test_timeout_can_resume_only_on_next_explicit_call(self):
        first = Opener(response=Response(socket.timeout("timed out")))
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            self.download(first)
        response = Response()
        response.seek(6)
        response.status = 206
        response.headers = {"Content-Range": "bytes 6-12/13", "Content-Length": "7"}
        second = Opener(response=response)
        self.assertEqual(self.download(second).read_bytes(), DATA)
        self.assertEqual(len(first.calls), 1)
        self.assertEqual(len(second.calls), 1)
        self.assertEqual(second.calls[0][0].get_header("Range"), "bytes=6-")
        self.assertEqual(second.calls[0][1], 30)

    def test_success_uses_30_second_timeout(self):
        opener = Opener(response=Response())
        self.assertEqual(self.download(opener).read_bytes(), DATA)
        self.assertEqual(len(opener.calls), 1)
        self.assertEqual(opener.calls[0][1], 30)

    def test_response_validation_error_is_not_a_network_error(self):
        error = ValueError("invalid response")
        with self.assertRaisesRegex(ValueError, "invalid response") as caught:
            self.download(Opener(response=Response(error)))
        self.assertIs(caught.exception, error)

    def test_local_write_failure_is_not_a_network_error(self):
        original = Path.open
        error = PermissionError("local cache is read-only")

        def open_path(path, *args, **kwargs):
            if path.suffix == ".part":
                raise error
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", open_path):
            with self.assertRaisesRegex(PermissionError, "local cache") as caught:
                self.download(Opener(response=Response()))
        self.assertIs(caught.exception, error)

    def test_inexact_range_retains_partial(self):
        self.part.write_bytes(DATA[:6])
        for code, headers in [(200, {}), (206, {"Content-Range": "bytes 5-12/13"})]:
            with self.subTest(code=code):
                response = Response()
                response.status, response.headers = code, headers
                opener = Opener(response=response)
                with self.assertRaisesRegex(ValueError, "exact resume range"):
                    self.download(opener)
                self.assertEqual(self.part.read_bytes(), DATA[:6])
                self.assertEqual(len(opener.calls), 1)

    def test_hash_failure_stays_a_validation_error(self):
        response = Response()
        response.getbuffer()[:] = b"x" * len(DATA)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.download(Opener(response=response))
        self.assertEqual((self.cache / (self.item["sha256"] + ".bad")).read_bytes(), b"x" * len(DATA))
        self.assertFalse((self.cache / (self.item["sha256"] + ".blob")).exists())

    def test_catalog_without_auto_download_defaults_to_false(self):
        (self.cache / "model-downloads.json").write_text(json.dumps({
            "schema_version": "local_model_downloads_v1", "bundles": [],
        }), encoding="utf-8")
        self.assertIs(status(self.cache)["auto_download"], False)
