from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from tempfile import mkdtemp
import threading
import unittest
from unittest.mock import patch

from any2xsampa.local import whole_audio
from any2xsampa.local.common import sha


RELEASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RELEASE_ROOT.parents[1]
TRACE_FIXTURES = REPO_ROOT / ".state/runs/render-audit-entry-20261116/trace-fixtures"
FIXTURE_LIMIT = 100 * 1024
TRACE_KEYS = {"raw_label_trace", "raw_trace_scope"}
ABSENT = object()
PHONE_MAPPING = {"path": "fixture-mapping.json", "sha256": "f" * 64,
                 "profile": "fixture", "schema_version": "fixture_v1"}


def captured_frame(start: int, end: int, symbol: str | None = "aa") -> dict:
    """The serialized _frame_prediction shape; no inference or PCM input."""
    raw = [
        {"label": "iɑ5", "posterior": .5, "logit": math.log(.5), "mapped_symbol": "aa"},
        {"label": "aː", "posterior": .3, "logit": math.log(.3), "mapped_symbol": "aa"},
        {"label": "<pad>", "posterior": .1, "logit": math.log(.1), "mapped_symbol": None},
    ]
    if symbol is None:
        raw = [
            {"label": "<pad>", "posterior": .9, "logit": math.log(.9), "mapped_symbol": None},
            {"label": "iɑ5", "posterior": .05, "logit": math.log(.05), "mapped_symbol": "aa"},
        ]
    candidates = [{"symbol": "aa", "confidence": .8}, {"symbol": "b", "confidence": .1}]
    return {
        "start_index": start, "end_index": end, "symbol": symbol,
        "raw_top_label": raw[0]["label"], "candidates": candidates,
        "raw_top_labels": raw,
        "mapped_frame_candidates": [
            {"symbol": c["symbol"], "posterior": c["confidence"], "logit": math.log(c["confidence"])}
            for c in candidates
        ],
    }


def memory_stitch(chunks: list[dict], frame_groups: list[list[dict]]) -> tuple[list, dict]:
    results = [
        {"frame_evidence_path": f"memory-{i}.json", "frame_evidence_sha256": "memory-only"}
        for i in range(len(chunks))
    ]
    payloads = [{"schema_version": "local_onnx_frames_v1", "frames": frames} for frames in frame_groups]
    with patch.object(whole_audio, "sha", return_value="memory-only"), patch.object(
        whole_audio, "read", side_effect=payloads
    ):
        return whole_audio.stitch_onnx(chunks, results, {}, threading.Event())


def memory_mapping_merge(manifest_mapping: object, result_mappings: list, frame_mappings: list) -> dict:
    info = {"sample_rate": 16000, "frames": 21 * 16000}
    chunks = whole_audio.plan_chunks(info)
    manifest = {
        "schema_version": "local_whole_evidence_v1", "policy": whole_audio.POLICY,
        "backend": "onnx-espeak", "input": info, "chunks": chunks, "completed": [{}, {}],
        "model_id": "fixture-onnx", "package_sha256": "fixture-package", "device": "cpu",
    }
    if manifest_mapping is not ABSENT:
        manifest["phone_mapping"] = manifest_mapping
    results, payloads = [], []
    for index in range(2):
        result = {
            "model_id": "fixture-onnx", "package_sha256": "fixture-package", "device": "cpu",
            "frame_evidence_path": f"memory-{index}.json", "frame_evidence_sha256": "memory-only",
        }
        if result_mappings[index] is not ABSENT:
            result["phone_mapping"] = result_mappings[index]
        start = 0 if index == 0 else 16000
        payload = {"schema_version": "local_onnx_frames_v1", "frames": [captured_frame(start, start + 320)]}
        if frame_mappings[index] is not ABSENT:
            payload["phone_mapping"] = frame_mappings[index]
        results.append(result)
        payloads.append(payload)
    with patch.object(whole_audio, "verify_chunk_result", side_effect=results), patch.object(
        whole_audio, "sha", return_value="memory-only"
    ), patch.object(whole_audio, "read", side_effect=payloads):
        return whole_audio.merge_results(manifest, threading.Event())


def core_segment(start: float, end: float) -> dict:
    """Fixed pre-change output, including existing normalization metadata."""
    return {
        "start_sec": start, "end_sec": end,
        "candidate_phones": [{"symbol": "aa", "confidence": .8}, {"symbol": "b", "confidence": .1}],
        "confidence": .8, "normalized_ipa": "a", "xsampa": "",
        "engine_export": None, "dialect_result": None, "syllable_group": None,
        "metadata": {
            "source_mode": "real", "backend_kind": "onnx-espeak",
            "whole_file_policy": whole_audio.POLICY,
            "normalization_notes": ["alias normalized: aa -> a"], "normalized_from": "aa",
        },
    }


def without_trace(segment: dict) -> dict:
    return {**segment, "metadata": {k: v for k, v in segment["metadata"].items() if k not in TRACE_KEYS}}


def fixture_bytes() -> int:
    return sum(path.stat().st_size for path in TRACE_FIXTURES.rglob("*") if path.is_file())


def write_small_json(path: Path, payload: dict) -> None:
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if fixture_bytes() + len(data) > FIXTURE_LIMIT:
        raise AssertionError("Retained trace fixtures would exceed 100 KiB; no files were deleted")
    with path.open("xb") as stream:
        stream.write(data)


class WholeOnnxTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        expected = RELEASE_ROOT / "src/any2xsampa/local/whole_audio.py"
        if Path(whole_audio.__file__).resolve() != expected.resolve():
            raise AssertionError("Set PYTHONPATH to the release source before running these tests")

    def test_unicode_raw_top_k_and_fixed_core(self) -> None:
        frames = [captured_frame(0, 320), captured_frame(320, 640)]
        frames[1]["raw_top_labels"][0]["label"] = "aː"
        frames[1]["raw_top_labels"][1]["label"] = "iɑ5"
        frames[1]["raw_top_label"] = "aː"
        before = copy.deepcopy(frames)
        chunks = [{"index": 0, "start_sec": 0., "keep_start_sec": 0., "keep_end_sec": .04}]
        segments, summary = memory_stitch(chunks, [frames])
        self.assertEqual([without_trace(s) for s in segments], [core_segment(0., .04)])
        self.assertEqual(summary, {
            "algorithm": "owned_mapped_frames_then_ctc", "frame_count": 2, "blank_frames": 0,
        })
        metadata = segments[0]["metadata"]
        self.assertEqual(metadata["raw_trace_scope"], "partial_top_k_not_full_posterior")
        trace = metadata["raw_label_trace"]
        self.assertEqual(trace["source"], "onnxruntime:raw_ctc_logits")
        self.assertEqual(trace["time_basis"], "original_audio")
        self.assertFalse(trace["trace_missing"])
        self.assertNotIn("label", trace)
        self.assertEqual(len(trace["frames"]), 2)
        for index, row in enumerate(trace["frames"]):
            self.assertEqual(row["candidates"], frames[index]["raw_top_labels"])
            self.assertEqual(row["raw_top_label"], frames[index]["raw_top_label"])
            self.assertEqual((row["chunk_index"], row["frame_index"]), (0, index))
            self.assertFalse(row["trace_missing"])
        self.assertEqual(frames, before)

    def test_owned_frames_absolute_clock_and_blank_separation(self) -> None:
        chunks = [
            {"index": 3, "start_sec": 18., "keep_start_sec": 18.9, "keep_end_sec": 19.},
            {"index": 4, "start_sec": 18.5, "keep_start_sec": 19., "keep_end_sec": 19.2},
        ]
        groups = [
            [captured_frame(12800, 13120), captured_frame(15680, 16000), captured_frame(16000, 16320)],
            [captured_frame(7360, 7680), captured_frame(7680, 8000), captured_frame(8000, 8320),
             captured_frame(8320, 8640, None), captured_frame(8640, 8960)],
        ]
        segments, summary = memory_stitch(chunks, groups)
        self.assertEqual([without_trace(s) for s in segments], [
            core_segment(18.98, 19.02), core_segment(19.04, 19.06),
        ])
        self.assertEqual(summary, {
            "algorithm": "owned_mapped_frames_then_ctc", "frame_count": 4, "blank_frames": 1,
        })
        rows = [row for s in segments for row in s["metadata"]["raw_label_trace"]["frames"]]
        self.assertEqual([(r["chunk_index"], r["frame_index"]) for r in rows], [(3, 1), (4, 2), (4, 4)])
        for row, frame, chunk in zip(rows, [groups[0][1], groups[1][2], groups[1][4]],
                                     [chunks[0], chunks[1], chunks[1]], strict=True):
            self.assertEqual(row["start_index"], frame["start_index"])
            self.assertEqual(row["end_index"], frame["end_index"])
            self.assertEqual(row["start_sec"], chunk["start_sec"] + frame["start_index"] / 16000)
            self.assertEqual(row["end_sec"], chunk["start_sec"] + frame["end_index"] / 16000)
            self.assertEqual(row["candidates"], frame["raw_top_labels"])

    def test_legacy_frames_are_unknown_without_inventing_raw_labels(self) -> None:
        chunks = [{"index": 0, "start_sec": 0., "keep_start_sec": 0., "keep_end_sec": .02}]
        for raw_state in ("absent", "empty", "null"):
            with self.subTest(raw_state=raw_state):
                frame = captured_frame(0, 320)
                frame.pop("raw_top_label")
                if raw_state == "absent":
                    frame.pop("raw_top_labels")
                else:
                    frame["raw_top_labels"] = [] if raw_state == "empty" else None
                segments, _ = memory_stitch(chunks, [[frame]])
                self.assertEqual(without_trace(segments[0]), core_segment(0., .02))
                trace = segments[0]["metadata"]["raw_label_trace"]
                self.assertEqual(trace["source"], "unknown")
                self.assertTrue(trace["trace_missing"])
                self.assertEqual(len(trace["frames"]), 1)
                self.assertTrue(trace["frames"][0]["trace_missing"])
                self.assertEqual(trace["frames"][0]["candidates"], [])
                self.assertNotIn("raw_top_label", trace["frames"][0])

    def test_mixed_missing_trace_keeps_frame_positions(self) -> None:
        frames = [captured_frame(0, 320), captured_frame(320, 640)]
        frames[1].pop("raw_top_labels")
        chunks = [{"index": 0, "start_sec": 0., "keep_start_sec": 0., "keep_end_sec": .04}]
        segments, _ = memory_stitch(chunks, [frames])
        self.assertEqual(without_trace(segments[0]), core_segment(0., .04))
        trace = segments[0]["metadata"]["raw_label_trace"]
        self.assertTrue(trace["trace_missing"])
        self.assertEqual([r["frame_index"] for r in trace["frames"]], [0, 1])
        self.assertEqual([r["trace_missing"] for r in trace["frames"]], [False, True])
        self.assertEqual(trace["frames"][0]["candidates"], frames[0]["raw_top_labels"])
        self.assertEqual(trace["frames"][1]["candidates"], [])
        self.assertEqual(trace["frames"][1]["raw_top_label"], frames[1]["raw_top_label"])

    def test_metadata_is_independent_between_frames_segments_and_input(self) -> None:
        frames = [captured_frame(0, 320), captured_frame(320, 640, None), captured_frame(640, 960)]
        frames[2]["raw_top_labels"] = frames[0]["raw_top_labels"]
        chunks = [{"index": 0, "start_sec": 0., "keep_start_sec": 0., "keep_end_sec": .06}]
        segments, _ = memory_stitch(chunks, [frames])
        first = segments[0]["metadata"]["raw_label_trace"]["frames"][0]
        second = segments[1]["metadata"]["raw_label_trace"]["frames"][0]
        self.assertIsNot(first["candidates"], frames[0]["raw_top_labels"])
        self.assertIsNot(first["candidates"][0], second["candidates"][0])
        first["candidates"][0]["label"] = "changed-output"
        self.assertEqual(frames[0]["raw_top_labels"][0]["label"], "iɑ5")
        self.assertEqual(second["candidates"][0]["label"], "iɑ5")
        frames[2]["raw_top_labels"][0]["label"] = "changed-input"
        self.assertEqual(second["candidates"][0]["label"], "iɑ5")
        self.assertEqual(segments[0]["normalized_ipa"], "a")
        self.assertEqual(segments[1]["normalized_ipa"], "a")

    def test_small_json_merge_uses_real_capture_shape(self) -> None:
        TRACE_FIXTURES.mkdir(parents=True, exist_ok=True)
        self.assertLessEqual(fixture_bytes(), FIXTURE_LIMIT)
        directory = Path(mkdtemp(prefix="whole-onnx-trace-", dir=TRACE_FIXTURES))
        info = {"sample_rate": 48000, "frames": 40 * 48000, "duration": 40., "sha256": "no-audio-read"}
        chunks = whole_audio.plan_chunks(info)
        groups = [
            [captured_frame(303680, 304000), captured_frame(304000, 304320)],
            [captured_frame(15680, 16000), captured_frame(16000, 16320),
             captured_frame(16320, 16640, None), captured_frame(16640, 16960)],
            [captured_frame(16000, 16320, None)],
        ]
        manifest = {
            "schema_version": "local_whole_evidence_v1", "policy": whole_audio.POLICY,
            "backend": "onnx-espeak", "input": info, "chunks": chunks, "completed": [],
            "model_id": "fixture-onnx", "package_sha256": "fixture-package", "device": "cpu",
        }
        for chunk, frames in zip(chunks, groups, strict=True):
            frame_path = directory / f"chunk-{chunk['index']}-frames.json"
            write_small_json(frame_path, {"schema_version": "local_onnx_frames_v1", "frames": frames})
            result = {
                "mode": "real", "input": {**info, **{k: chunk[k] for k in (
                    "source_start_sample", "source_end_sample_exclusive"
                )}},
                "time_basis": "original_audio", "package_sha256": "fixture-package",
                "model_id": "fixture-onnx", "device": "cpu", "segments": [],
                "frame_evidence_path": str(frame_path), "frame_evidence_sha256": sha(frame_path),
            }
            result_path = directory / f"chunk-{chunk['index']}-result.json"
            write_small_json(result_path, result)
            manifest["completed"].append({"result_path": str(result_path), "result_sha256": sha(result_path)})
        merged = whole_audio.merge_results(manifest, threading.Event())
        self.assertEqual([without_trace(s) for s in merged["segments"]], [
            core_segment(18.98, 19.02), core_segment(19.04, 19.06),
        ])
        self.assertEqual(merged["merge_summary"], {
            "algorithm": "owned_mapped_frames_then_ctc", "frame_count": 5, "blank_frames": 2,
        })
        self.assertEqual(merged["time_basis"], "original_audio")
        self.assertEqual(merged["chunk_count"], 3)
        self.assertNotIn("phone_mapping", merged)
        trace = merged["segments"][0]["metadata"]["raw_label_trace"]["frames"]
        self.assertEqual([(r["chunk_index"], r["frame_index"]) for r in trace], [(0, 0), (1, 1)])
        self.assertEqual(trace[0]["candidates"], groups[0][0]["raw_top_labels"])
        self.assertEqual(trace[1]["candidates"], groups[1][1]["raw_top_labels"])
        self.assertEqual((trace[0]["start_sec"], trace[1]["end_sec"]), (18.98, 19.02))
        self.assertLessEqual(fixture_bytes(), FIXTURE_LIMIT)

    def test_phone_mapping_round_trip_and_legacy_none(self) -> None:
        legacy = memory_mapping_merge(ABSENT, [ABSENT, None], [None, ABSENT])
        self.assertNotIn("phone_mapping", legacy)
        for identity in (None, PHONE_MAPPING):
            with self.subTest(identity=identity):
                merged = memory_mapping_merge(identity, [identity, identity], [identity, identity])
                self.assertEqual(merged["segments"], legacy["segments"])
                self.assertEqual(merged["merge_summary"], legacy["merge_summary"])
                if identity is None:
                    self.assertNotIn("phone_mapping", merged)
                else:
                    self.assertEqual(merged["phone_mapping"], PHONE_MAPPING)
                    self.assertIsNot(merged["phone_mapping"], identity)
                    merged["phone_mapping"]["profile"] = "changed-output"
                    self.assertEqual(identity["profile"], "fixture")

    def test_rejects_mixed_or_missing_chunk_mapping(self) -> None:
        cases = [(PHONE_MAPPING, [PHONE_MAPPING, missing]) for missing in (ABSENT, None)]
        cases.extend((PHONE_MAPPING, [PHONE_MAPPING, {**PHONE_MAPPING, key: "different"}])
                     for key in PHONE_MAPPING)
        cases.append((ABSENT, [ABSENT, PHONE_MAPPING]))
        for index, (identity, result_identities) in enumerate(cases):
            with self.subTest(case=index), self.assertRaisesRegex(ValueError, "chunk phone_mapping identity"):
                memory_mapping_merge(identity, result_identities, result_identities)

    def test_rejects_frame_evidence_mapping_mismatch(self) -> None:
        cases = [(PHONE_MAPPING, [PHONE_MAPPING, missing]) for missing in (ABSENT, None)]
        cases.extend((PHONE_MAPPING, [PHONE_MAPPING, {**PHONE_MAPPING, key: "different"}])
                     for key in PHONE_MAPPING)
        cases.append((ABSENT, [ABSENT, PHONE_MAPPING]))
        for index, (identity, frame_identities) in enumerate(cases):
            with self.subTest(case=index), self.assertRaisesRegex(ValueError, "frame evidence phone_mapping identity"):
                memory_mapping_merge(identity, [identity, identity], frame_identities)


if __name__ == "__main__":
    unittest.main()
