from __future__ import annotations

import importlib
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "pitch_note_analysis_v1"
ROUGH_EVIDENCE_NOTE = (
    "First-pass rough pitch/note evidence for ASR->VSQX preparation. "
    "Full-mix or polyphonic audio can confuse f0 and tempo extraction; "
    "do not treat this as a perfect transcription or final score."
)
_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


class PitchNoteDependencyError(RuntimeError):
    def __init__(self, message: str, missing_packages: Iterable[str] = ()) -> None:
        super().__init__(message)
        self.missing_packages = tuple(missing_packages)


@dataclass(frozen=True, slots=True)
class PitchNoteAnalysisConfig:
    sample_rate: int = 22_050
    hop_length: int = 512
    frame_length: int = 2048
    fmin_hz: float = 65.40639132514966  # C2
    fmax_hz: float = 2093.004522404789  # C7
    max_unvoiced_gap_sec: float = 0.06
    min_voiced_segment_duration_sec: float = 0.05
    min_note_duration_sec: float = 0.08
    min_note_frames: int = 2


@dataclass(frozen=True, slots=True)
class PitchFrame:
    time_sec: float
    hz: float | None
    voiced: bool
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class _VoicedFrame:
    time_sec: float
    hz: float
    confidence: float | None
    midi: float
    rounded_midi: int


@dataclass(frozen=True, slots=True)
class _BackendDependencies:
    librosa: Any
    numpy: Any
    soundfile: Any
    versions: dict[str, str | None]


class LibrosaPitchNoteAnalyzer:
    """Optional librosa backend for rough monophonic f0 / note evidence.

    The backend is intentionally isolated so environments without librosa,
    numpy, or soundfile can still import the package and run unrelated tests.
    """

    name = "librosa_pyin_v1"

    def analyze(
        self,
        audio_path: str | Path,
        *,
        config: PitchNoteAnalysisConfig | None = None,
    ) -> dict[str, Any]:
        cfg = config or PitchNoteAnalysisConfig()
        source_path = Path(audio_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Audio file not found: {source_path}")

        deps = self._load_dependencies()
        librosa = deps.librosa
        np = deps.numpy
        sf = deps.soundfile

        try:
            soundfile_info = sf.info(str(source_path))
        except Exception:
            soundfile_info = None

        y, sr = librosa.load(str(source_path), sr=cfg.sample_rate, mono=True)
        if len(y) == 0:
            raise ValueError(f"Audio file contains no samples: {source_path}")

        try:
            f0, voiced_flags, voiced_probs = librosa.pyin(
                y,
                fmin=cfg.fmin_hz,
                fmax=cfg.fmax_hz,
                sr=sr,
                hop_length=cfg.hop_length,
                frame_length=cfg.frame_length,
            )
        except Exception as exc:
            raise RuntimeError(f"librosa pYIN pitch extraction failed for {source_path}: {exc}") from exc

        frame_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=cfg.hop_length)
        tempo_bpm = self._estimate_tempo(librosa, np, y, sr, cfg)

        provenance_extra: dict[str, Any] = {
            "analyzer": self.name,
            "backend": "librosa.pyin",
            "dependency_versions": deps.versions,
            "requested_sample_rate": cfg.sample_rate,
            "actual_sample_rate": int(sr),
            "hop_length": cfg.hop_length,
            "frame_length": cfg.frame_length,
        }
        if soundfile_info is not None:
            provenance_extra["audio_info"] = {
                "samplerate": getattr(soundfile_info, "samplerate", None),
                "frames": getattr(soundfile_info, "frames", None),
                "channels": getattr(soundfile_info, "channels", None),
                "format": getattr(soundfile_info, "format", None),
                "subtype": getattr(soundfile_info, "subtype", None),
            }

        return build_pitch_note_analysis(
            audio_path=source_path,
            f0_hz=f0,
            frame_times_sec=frame_times,
            voiced_probabilities=voiced_probs,
            voiced_flags=voiced_flags,
            tempo_bpm=tempo_bpm,
            frame_hz=float(sr) / float(cfg.hop_length),
            config=cfg,
            provenance_extra=provenance_extra,
        )

    def _load_dependencies(self) -> _BackendDependencies:
        modules: dict[str, Any] = {}
        missing: list[str] = []
        for package_name in ("librosa", "numpy", "soundfile"):
            try:
                modules[package_name] = importlib.import_module(package_name)
            except ImportError:
                missing.append(package_name)
        if missing:
            package_text = ", ".join(missing)
            raise PitchNoteDependencyError(
                "Pitch/note audio analysis requires optional packages "
                f"{package_text}. Install librosa, numpy, and soundfile to use the audio backend.",
                missing,
            )
        versions = {
            package_name: str(getattr(module, "__version__", "")) or None
            for package_name, module in modules.items()
        }
        return _BackendDependencies(
            librosa=modules["librosa"],
            numpy=modules["numpy"],
            soundfile=modules["soundfile"],
            versions=versions,
        )

    def _estimate_tempo(
        self,
        librosa: Any,
        np: Any,
        y: Any,
        sr: int,
        config: PitchNoteAnalysisConfig,
    ) -> float | None:
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=config.hop_length)
            if len(onset_env) == 0:
                return None
            if hasattr(librosa.beat, "tempo"):
                tempo_values = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
            else:
                tempo_values = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)
            values = np.asarray(tempo_values, dtype=float).reshape(-1)
            if len(values) == 0 or not math.isfinite(float(values[0])):
                return None
            return float(values[0])
        except Exception:
            return None


def analyze_audio_pitch_notes(
    audio_path: str | Path,
    *,
    config: PitchNoteAnalysisConfig | None = None,
    analyzer: LibrosaPitchNoteAnalyzer | None = None,
) -> dict[str, Any]:
    selected_analyzer = analyzer or LibrosaPitchNoteAnalyzer()
    return selected_analyzer.analyze(audio_path, config=config)


def build_pitch_note_analysis(
    *,
    audio_path: str | Path,
    f0_hz: Sequence[Any],
    frame_times_sec: Sequence[Any],
    voiced_probabilities: Sequence[Any] | None = None,
    voiced_flags: Sequence[Any] | None = None,
    tempo_bpm: float | None = None,
    frame_hz: float | None = None,
    config: PitchNoteAnalysisConfig | None = None,
    provenance_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or PitchNoteAnalysisConfig()
    source_path = Path(audio_path)
    resolved_frame_hz = float(frame_hz) if _is_finite_number(frame_hz) and float(frame_hz) > 0 else (
        float(cfg.sample_rate) / float(cfg.hop_length)
    )
    frame_interval = 1.0 / resolved_frame_hz if resolved_frame_hz > 0 else 0.0
    frames = _build_frames(
        f0_hz=f0_hz,
        frame_times_sec=frame_times_sec,
        voiced_probabilities=voiced_probabilities,
        voiced_flags=voiced_flags,
    )
    voiced_frames = [_to_voiced_frame(frame) for frame in frames]
    voiced_frames = [frame for frame in voiced_frames if frame is not None]
    voiced_groups = _group_voiced_frames(voiced_frames, cfg, frame_interval)
    voiced_segments = [
        _voiced_segment_payload(group, index, frame_interval)
        for index, group in enumerate(voiced_groups)
        if _duration(group, frame_interval) >= cfg.min_voiced_segment_duration_sec
    ]
    note_candidates = _note_candidate_payloads(voiced_groups, cfg, frame_interval)
    safe_tempo = _rounded_or_none(tempo_bpm, digits=3)
    provenance = _build_provenance(
        source_path=source_path,
        frame_count=len(frames),
        voiced_frame_count=len(voiced_frames),
        config=cfg,
        extra=provenance_extra,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "audio_path": str(source_path),
        "tempo_bpm": safe_tempo,
        "frame_hz": round(resolved_frame_hz, 6),
        "f0": [_frame_payload(frame) for frame in frames],
        "voiced_segments": voiced_segments,
        "note_candidates": note_candidates,
        "provenance": provenance,
        "model_output": False,
        "analysis_output": True,
    }


def _build_frames(
    *,
    f0_hz: Sequence[Any],
    frame_times_sec: Sequence[Any],
    voiced_probabilities: Sequence[Any] | None,
    voiced_flags: Sequence[Any] | None,
) -> list[PitchFrame]:
    frames: list[PitchFrame] = []
    count = min(len(f0_hz), len(frame_times_sec))
    for index in range(count):
        time_sec = _float_or_none(frame_times_sec[index])
        if time_sec is None:
            continue
        hz = _float_or_none(f0_hz[index])
        probability = _float_or_none(voiced_probabilities[index]) if voiced_probabilities is not None and index < len(voiced_probabilities) else None
        explicit_voiced = None
        if voiced_flags is not None and index < len(voiced_flags):
            explicit_voiced = bool(voiced_flags[index])
        voiced = bool(explicit_voiced) if explicit_voiced is not None else hz is not None and hz > 0
        if hz is None or hz <= 0:
            hz = None
            voiced = False
        frames.append(
            PitchFrame(
                time_sec=float(time_sec),
                hz=hz,
                voiced=voiced,
                confidence=_clamp01(probability) if probability is not None else None,
            )
        )
    return frames


def _to_voiced_frame(frame: PitchFrame) -> _VoicedFrame | None:
    if not frame.voiced or frame.hz is None or frame.hz <= 0:
        return None
    midi = hz_to_midi(frame.hz)
    if not math.isfinite(midi):
        return None
    return _VoicedFrame(
        time_sec=frame.time_sec,
        hz=frame.hz,
        confidence=frame.confidence,
        midi=midi,
        rounded_midi=int(round(midi)),
    )


def _group_voiced_frames(
    frames: list[_VoicedFrame],
    config: PitchNoteAnalysisConfig,
    frame_interval: float,
) -> list[list[_VoicedFrame]]:
    if not frames:
        return []
    max_gap = config.max_unvoiced_gap_sec + (frame_interval * 1.5)
    groups: list[list[_VoicedFrame]] = [[frames[0]]]
    for frame in frames[1:]:
        if frame.time_sec - groups[-1][-1].time_sec <= max_gap:
            groups[-1].append(frame)
        else:
            groups.append([frame])
    return groups


def _note_candidate_payloads(
    voiced_groups: list[list[_VoicedFrame]],
    config: PitchNoteAnalysisConfig,
    frame_interval: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for segment_index, group in enumerate(voiced_groups):
        current: list[_VoicedFrame] = []
        current_midi: int | None = None
        for frame in group:
            if current_midi is None or frame.rounded_midi == current_midi:
                current.append(frame)
                current_midi = frame.rounded_midi
                continue
            _append_note_candidate(candidates, current, segment_index, frame_interval, config)
            current = [frame]
            current_midi = frame.rounded_midi
        _append_note_candidate(candidates, current, segment_index, frame_interval, config)
    return candidates


def _append_note_candidate(
    candidates: list[dict[str, Any]],
    frames: list[_VoicedFrame],
    voiced_segment_index: int,
    frame_interval: float,
    config: PitchNoteAnalysisConfig,
) -> None:
    if not frames:
        return
    duration = _duration(frames, frame_interval)
    if duration < config.min_note_duration_sec or len(frames) < config.min_note_frames:
        return
    rounded_midi = int(round(_median([frame.midi for frame in frames])))
    mean_hz = statistics.fmean(frame.hz for frame in frames)
    median_hz = _median([frame.hz for frame in frames])
    midi_values = [frame.midi for frame in frames]
    pitch_std = statistics.pstdev(midi_values) if len(midi_values) > 1 else 0.0
    pitch_stability = _clamp01(1.0 - (pitch_std / 0.75))
    confidences = [frame.confidence for frame in frames if frame.confidence is not None]
    voiced_confidence = statistics.fmean(confidences) if confidences else 0.75
    confidence = _clamp01(0.65 * voiced_confidence + 0.35 * pitch_stability)
    candidates.append(
        {
            "note_id": f"note-{len(candidates)}",
            "start_sec": round(frames[0].time_sec, 6),
            "end_sec": round(frames[-1].time_sec + frame_interval, 6),
            "duration_sec": round(duration, 6),
            "midi": rounded_midi,
            "note_name": midi_to_note_name(rounded_midi),
            "mean_hz": round(mean_hz, 6),
            "median_hz": round(median_hz, 6),
            "confidence": round(confidence, 6),
            "frame_count": len(frames),
            "voiced_segment_index": voiced_segment_index,
            "source": "rough_f0_quantized",
            "analysis_output": True,
            "model_output": False,
        }
    )


def _voiced_segment_payload(
    frames: list[_VoicedFrame],
    index: int,
    frame_interval: float,
) -> dict[str, Any]:
    mean_hz = statistics.fmean(frame.hz for frame in frames)
    median_hz = _median([frame.hz for frame in frames])
    mean_midi = statistics.fmean(frame.midi for frame in frames)
    return {
        "segment_id": f"voiced-{index}",
        "start_sec": round(frames[0].time_sec, 6),
        "end_sec": round(frames[-1].time_sec + frame_interval, 6),
        "duration_sec": round(_duration(frames, frame_interval), 6),
        "mean_hz": round(mean_hz, 6),
        "median_hz": round(median_hz, 6),
        "mean_midi": round(mean_midi, 6),
        "nearest_midi": int(round(_median([frame.midi for frame in frames]))),
        "nearest_note_name": midi_to_note_name(int(round(_median([frame.midi for frame in frames])))),
        "frame_count": len(frames),
        "analysis_output": True,
        "model_output": False,
    }


def _frame_payload(frame: PitchFrame) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "time_sec": round(frame.time_sec, 6),
        "hz": _rounded_or_none(frame.hz, digits=6),
        "voiced": frame.voiced,
    }
    if frame.confidence is not None:
        payload["confidence"] = round(frame.confidence, 6)
    return payload


def _build_provenance(
    *,
    source_path: Path,
    frame_count: int,
    voiced_frame_count: int,
    config: PitchNoteAnalysisConfig,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_path": str(source_path),
        "truth_layer": "pitch_note_evidence",
        "asr_to_vsqx_role": "note_evidence",
        "full_mix_rough_evidence": True,
        "rough_evidence_note": ROUGH_EVIDENCE_NOTE,
        "analysis_backend_optional": True,
        "frame_count": frame_count,
        "voiced_frame_count": voiced_frame_count,
        "config": {
            "sample_rate": config.sample_rate,
            "hop_length": config.hop_length,
            "frame_length": config.frame_length,
            "fmin_hz": config.fmin_hz,
            "fmax_hz": config.fmax_hz,
            "max_unvoiced_gap_sec": config.max_unvoiced_gap_sec,
            "min_voiced_segment_duration_sec": config.min_voiced_segment_duration_sec,
            "min_note_duration_sec": config.min_note_duration_sec,
            "min_note_frames": config.min_note_frames,
        },
        "model_output": False,
        "analysis_output": True,
    }
    if extra:
        provenance.update(extra)
    return provenance


def hz_to_midi(hz: float) -> float:
    if hz <= 0:
        return float("nan")
    return 69.0 + (12.0 * math.log2(hz / 440.0))


def midi_to_hz(midi: int | float) -> float:
    return 440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0))


def midi_to_note_name(midi: int) -> str:
    octave = (midi // 12) - 1
    return f"{_NOTE_NAMES[midi % 12]}{octave}"


def _duration(frames: Sequence[_VoicedFrame], frame_interval: float) -> float:
    if not frames:
        return 0.0
    return max(0.0, frames[-1].time_sec + frame_interval - frames[0].time_sec)


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _is_finite_number(value: Any) -> bool:
    return _float_or_none(value) is not None


def _rounded_or_none(value: Any, *, digits: int) -> float | None:
    number = _float_or_none(value)
    return round(number, digits) if number is not None else None


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
