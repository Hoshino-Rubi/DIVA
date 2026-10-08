from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from any2xsampa.adapters.recognizer import OnnxEspeakCTCUniversalPhoneRecognizer
from any2xsampa.config import __path__ as config_paths
from any2xsampa.core.models import PhoneSegment
from any2xsampa.dialects.cantonese import CantoneseDialectPlugin
from any2xsampa.exporters.xsampa import JsonXSampaExporter
from any2xsampa.pipeline.normalizer import ConfigurablePhoneNormalizer
from any2xsampa.vocaloid.vsqx_alignment import parse_vsqx_timeline


SCHEMA_VERSION = "note_sliced_onnx_espeak_phone_segments_v1"


@dataclass(frozen=True)
class NoteWindow:
    track_index: int
    part_index: int
    note_index: int
    start_sec: float
    end_sec: float


def extract_note_timing_windows(vsqx_path: str | Path) -> tuple[dict[str, object], list[NoteWindow]]:
    timeline = parse_vsqx_timeline(vsqx_path)
    raw_notes = timeline.get("notes")
    if not isinstance(raw_notes, list) or not raw_notes:
        raise ValueError("VSQX reference does not contain notes")
    windows = [
        NoteWindow(
            track_index=_optional_int(note, "track_index", 0),
            part_index=_optional_int(note, "part_index", 0),
            note_index=_optional_int(note, "note_index", note_index),
            start_sec=_number(note, "start_sec"),
            end_sec=_number(note, "end_sec"),
        )
        for note_index, note in enumerate(raw_notes)
        if isinstance(note, dict)
    ]
    timing_source = {
        "path": timeline.get("path"),
        "sha256": timeline.get("sha256"),
        "note_count": len(windows),
        "fields_used_for_decoding": [
            "track_index",
            "part_index",
            "note_index",
            "start_sec",
            "end_sec",
        ],
        "fields_explicitly_excluded_from_decoding": ["lyric", "phoneme", "note_number"],
    }
    return timing_source, windows


def recognize_note_slices(
    audio_path: str | Path,
    vsqx_path: str | Path,
    *,
    backend: Any | None = None,
) -> dict[str, object]:
    audio = Path(audio_path)
    timing_source, windows = extract_note_timing_windows(vsqx_path)
    recognizer = backend or OnnxEspeakCTCUniversalPhoneRecognizer()
    sample_rate, samples = recognizer._read_pcm16_mono(audio)
    frame_size = max(160, int(sample_rate * 0.04))
    hop_size = max(80, int(sample_rate * 0.02))
    segments: list[PhoneSegment] = []

    for window in windows:
        start_index = max(0, min(len(samples), int(round(window.start_sec * sample_rate))))
        end_index = max(start_index + 1, min(len(samples), int(round(window.end_sec * sample_rate))))
        note_samples = samples[start_index:end_index]
        predictions = recognizer._predict_frames(note_samples, sample_rate, frame_size, hop_size)
        note_segments = _collapse_note_predictions(
            recognizer,
            predictions,
            sample_rate=sample_rate,
            frame_size=frame_size,
            hop_size=hop_size,
        )
        absolute_offset = start_index / sample_rate
        note_end = end_index / sample_rate
        for segment in note_segments:
            segment.time_range.start_sec = max(window.start_sec, segment.time_range.start_sec + absolute_offset)
            segment.time_range.end_sec = min(note_end, segment.time_range.end_sec + absolute_offset)
            if segment.time_range.end_sec <= segment.time_range.start_sec:
                continue
            segment.metadata.update(
                {
                    "note_sliced_second_pass": True,
                    "vsqx_timing_used": True,
                    "vsqx_reference_content_used": False,
                    "reference_lyric_excluded": True,
                    "reference_phoneme_excluded": True,
                    "note_slice_track_index": window.track_index,
                    "note_slice_part_index": window.part_index,
                    "note_slice_note_index": window.note_index,
                    "note_slice_start_sec": window.start_sec,
                    "note_slice_end_sec": window.end_sec,
                    "producer_backend_kind": "onnx-espeak",
                    "producer_raw_model_output": {
                        "raw_model_output_kind": "trained_onnx_ctc_model",
                        **recognizer.model_metadata(),
                    },
                }
            )
            segments.append(segment)

    config_root = Path(next(iter(config_paths)))
    ConfigurablePhoneNormalizer(config_root / "ipa_aliases.json").normalize(segments)
    CantoneseDialectPlugin().apply(segments)
    JsonXSampaExporter(config_root / "ipa_to_xsampa.json").export(segments)
    segments.sort(key=lambda segment: (segment.time_range.start_sec, segment.time_range.end_sec))
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "infer-open-note-sliced-second-pass",
        "audio_path": str(audio.resolve(strict=False)),
        "dialect": "cantonese",
        "lyrics": None,
        "model_output": True,
        "real_model_output": True,
        "backend": "onnx-espeak",
        "raw_model_output_kind": "trained_onnx_ctc_model",
        "note_timing_guided": True,
        "recognizer_first_pass": False,
        "reference_lyrics_used": False,
        "reference_phonemes_used": False,
        "timing_source": timing_source,
        "model": recognizer.model_metadata(),
        "trace": {
            "recognizer_adapter": type(recognizer).__name__,
            "normalizer": "ConfigurablePhoneNormalizer",
            "dialect_plugin": "CantoneseDialectPlugin",
            "exporter": "JsonXSampaExporter",
        },
        "warnings": [
            "Second-pass experiment: VSQX note timing was used to slice audio.",
            "VSQX lyric and phoneme content were excluded from decoding and are reserved for posthoc evaluation.",
        ],
        "segments": [segment.to_dict() for segment in segments],
    }


def _collapse_note_predictions(
    recognizer: Any,
    predictions: list[dict[str, Any]],
    *,
    sample_rate: int,
    frame_size: int,
    hop_size: int,
) -> list[PhoneSegment]:
    if not predictions:
        return []
    collapsed: list[PhoneSegment] = []
    current_symbol: str | None = None
    current_windows: list[dict[str, Any]] = []
    start_index = 0
    for prediction in predictions:
        candidates = prediction.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            continue
        symbol = str(candidates[0].symbol)
        prediction_start = int(prediction.get("start_index", 0))
        if current_symbol is None:
            current_symbol = symbol
            start_index = prediction_start
        if symbol != current_symbol:
            collapsed.append(
                recognizer._build_real_segment(
                    current_symbol,
                    current_windows,
                    start_index,
                    int(current_windows[-1].get("end_index", prediction_start)),
                    sample_rate,
                    frame_size,
                    hop_size,
                    None,
                )
            )
            current_symbol = symbol
            current_windows = []
            start_index = prediction_start
        current_windows.append(prediction)
    if current_symbol is not None and current_windows:
        collapsed.append(
            recognizer._build_real_segment(
                current_symbol,
                current_windows,
                start_index,
                int(current_windows[-1].get("end_index", start_index + frame_size)),
                sample_rate,
                frame_size,
                hop_size,
                None,
            )
        )
    return collapsed


def _number(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _optional_int(mapping: dict[str, Any], key: str, default: int) -> int:
    value = mapping.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value
