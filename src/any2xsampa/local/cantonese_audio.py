"""Whole-song Cantonese audio to cleaned GAME notes and phrase-conditioned Miku V4C syllables."""
from __future__ import annotations

from math import gcd
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
from time import perf_counter

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from any2xsampa.adapters.game_onnx import GameOnnxRuntime
from any2xsampa.music.game_notes import decode_regions
from any2xsampa.music.melody_cleanup import game_rows as note_rows
from .common import read, write
from .note_windows import stitch_note_windows
from .whole_audio import plan_chunks

CONFIG = Path(__file__).resolve().parents[1] / "config"
POLICY = CONFIG / "local_score_v1.json"


def _resample(samples: np.ndarray, source: int, target: int) -> np.ndarray:
    divisor = gcd(source, target)
    return resample_poly(samples, target // divisor, source // divisor, axis=0)


def melody(samples: np.ndarray, snapshot: Path, device: str = "cpu") -> dict:
    started = perf_counter()
    engine = GameOnnxRuntime(snapshot, device=device)
    windows = plan_chunks({"sample_rate": 44100, "frames": len(samples)})
    predictions = []
    for window in windows:
        a, b = window["source_start_sample"], window["source_end_sample_exclusive"]
        raw = engine.infer(np.ascontiguousarray(samples[None, a:b]))
        predictions.append(decode_regions(raw, start_sec=a / 44100, duration_sec=(b-a) / 44100))
    notes = stitch_note_windows(predictions, windows)
    return {"game_seconds": perf_counter() - started,
            "game_providers": {k: v.get_providers()[0] for k, v in engine.sessions.items()},
            "note_candidates": [{"start_sec": r["start_sec"], "duration_sec": r["duration_sec"],
                                 "midi": r["pitch_midi"]} for r in notes]}


def build_playable_score(audio: Path, pcm: Path, pitch: dict, bundle: Path | None,
                         acoustic_python: Path | None, fixture: Path | None, separated: bool,
                         checkpoint: Path | None = None, ctc: Path | None = None, scoop_merge: bool = True, device: str = "auto") -> dict:
    duration = sf.info(audio).duration
    info = {"path": str(audio.resolve()), "start_sec": 0., "end_sec": duration}
    policy = read(POLICY)
    if fixture is not None:
        from .score_export import make_fixture_score as make_score
        from any2xsampa.pipeline.normalizer import ConfigurablePhoneNormalizer
        from any2xsampa.music.melody_cleanup import clean_melody
        pitch = clean_melody(pitch,policy["tempo_default"]*policy["ticks_per_quarter"]/60,v6=True)
        from any2xsampa.adapters.recognizer import FixtureUniversalPhoneRecognizer
        phones = FixtureUniversalPhoneRecognizer(fixture).recognize(pcm)
        ConfigurablePhoneNormalizer(CONFIG / "ipa_aliases.json").normalize(phones)
        segments = [dict(start_sec=p.time_range.start_sec, end_sec=p.time_range.end_sec,
                         normalized_ipa=p.normalized_ipa) for p in phones]
        score = make_score({"input": info, "segments": segments}, pitch, policy["tempo_default"])
        score.update(source="audio_game", recognition_mode="fixture", separated=separated)
        return score
    if bundle is None or checkpoint is None or ctc is None:
        raise ValueError("Supply Cantonese4, a local eSpeak ONNX model and a local singing adapter")
    from any2xsampa.adapters.singing_frames import SingingEncoders, audio16
    from ..adapters.singing_v9_asr import LocalWhisper
    from .singing_v9_score import score as decode
    vocal = audio16(pcm)
    encoder = SingingEncoders(bundle, ctc, device)
    frames = encoder.encode(vocal)
    ctc_provider = encoder.ctc.get_providers()[0]
    del encoder
    if scoop_merge:
        from .singing_v13_score import score as supervised
        phrases = LocalWhisper(bundle).decode(vocal, note_rows(pitch, duration))
        result = supervised(audio, vocal, frames, pitch, checkpoint,
                            read(CONFIG / "supervised_seam.json"), phrases,
                            read(ctc.parent / "tokenizer/vocab.json"))
    else:
        result = decode(audio, vocal, frames, pitch, bundle, checkpoint, separated,
                        checked_confidence=.70)
    result["inference"] = dict(game_seconds=pitch["game_seconds"],
                               game_providers=pitch["game_providers"], ctc_provider=ctc_provider)
    return result


def save_playable_score(score: dict, output: Path) -> dict:
    from .generated_vsqx import export_generated_score
    score_file = output.with_suffix(".score.json")
    if any(p.exists() for p in (output, score_file, Path(str(output) + ".provenance.json"))):
        raise FileExistsError(output)
    write(score_file, score)
    try:
        return export_generated_score(score_file, output)
    except BaseException:
        score_file.unlink(missing_ok=True)
        raise


def transcribe(audio: Path, output: Path, snapshot: Path | None, model: Path | None,
               separator_bundle: Path | None = None, fixture: Path | None = None,
               syllable_bundle: Path | None = None,
               acoustic_python: Path | None = None, singing_adapter: Path | None = None, scoop_merge: bool = True, device: str = "auto") -> dict:
    """All optional resources are local. Temporary audio is removed on success or failure."""
    if output.suffix.lower() != ".vsqx":
        raise ValueError("Expected a .vsqx destination")
    score_file = output.with_suffix(".score.json")
    if any(p.exists() for p in (output, score_file, Path(str(output)+".provenance.json"))):
        raise FileExistsError(output)
    from .common import ROOT
    local_models = ROOT / "models"
    if fixture is None:
        separator_bundle = separator_bundle or local_models / "cantonese4"
    syllable_bundle = syllable_bundle or separator_bundle
    if fixture is None and (syllable_bundle is None or not (syllable_bundle / "small" / "config.json").is_file()):
        raise ValueError("Supply a local cantonese4-local-models folder with --syllable-bundle")
    if fixture is None:
        singing_adapter = singing_adapter or local_models / "cantonese-singing-v9-candidate/final_rank8.pt"
        model = model or local_models / "onnx-espeak" / "model"
        if not singing_adapter.is_file() or not model.exists():
            raise FileNotFoundError("Supply --singing-adapter and --model for the singing backend")
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = sf.info(audio).duration
    if duration * 125000 > 5_000_000_000 or shutil.disk_usage(output.parent).free < 150 * 1024**3 + duration * 125000:
        raise OSError("Insufficient temporary audio budget")
    from any2xsampa.adapters.cantonese_audio_worker import run, invoke
    def acoustic(action: str, source: Path, bundle: Path, destination: Path) -> None:
        if acoustic_python:
            invoke(acoustic_python, action, source, bundle, destination)
        else:
            run(action, source, bundle, destination)

    with TemporaryDirectory(prefix="diva-v4c-", dir=output.parent) as tmp:
        source = audio
        if separator_bundle:
            source = Path(tmp) / "separated.wav"
            acoustic("separate", audio, separator_bundle, source)
        values, rate = sf.read(source, dtype="float32", always_2d=True)
        mono = values.mean(axis=1, dtype=np.float32)
        if rate != 44100:
            mono = _resample(mono, rate, 44100)
        pcm = source
        if fixture is not None:
            pcm = Path(tmp) / "vocal.wav"
            sf.write(pcm, _resample(mono, 44100, 16000), 16000, subtype="PCM_16")
        # CUDA changes GAME boundaries even with the same ORT seed. Auto keeps
        # the qualified CPU grid; CTC still prefers CUDA.
        pitch = melody(mono, snapshot or local_models / "game", "cpu" if device == "auto" else device)
        score = build_playable_score(audio, pcm, pitch, syllable_bundle, acoustic_python,
                                     fixture, separator_bundle is not None, singing_adapter,
                                     model, scoop_merge, device)
    return save_playable_score(score, output)
