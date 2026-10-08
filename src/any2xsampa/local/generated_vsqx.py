"""VSQ4 serialization of machine notes; never estimates or changes their pitch."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from .common import finite, read, sha, write
from .score_export import note_number, positive
from any2xsampa.vocaloid.draft import VocalNoteDraft, VocalTrackDraft, VocalTrackDraftTrack
from any2xsampa.vocaloid.phoneme_policy import validate_miku_v4c_note_phonemes
from any2xsampa.vocaloid.vsqx import build_vsqx_tree
from any2xsampa.vocaloid.vsqx_alignment import parse_vsqx_timeline

TEMPLATE = Path(__file__).resolve().parents[1] / "vocaloid/resources/vsq4_empty.vsqx"


def validate_grid(tempo: float, resolution: int) -> float:
    tempo = finite(tempo, "tempo")
    if not 20 <= tempo <= 400 or abs(tempo * 100 - round(tempo * 100)) > 1e-6:
        raise ValueError("Tempo must be 20–400 BPM with at most two decimal places")
    if type(resolution) is not int or not 1 <= resolution <= 9600:
        raise ValueError("Invalid ticks_per_quarter")
    return tempo * resolution / 60


def _draft_notes(score: dict[str, Any], scale: float) -> list[VocalNoteDraft]:
    notes = []
    previous_end = -1.0
    previous_tick = -1
    for i, row in enumerate(score["notes"]):
        start = finite(row["start_sec"], "note start")
        duration = positive(row["duration_sec"], "note duration")
        end = start + duration
        if start < 0 or start < previous_end - 1e-8:
            raise ValueError("Overlapping or unordered monophonic score")
        if "end_sec" in row and abs(finite(row["end_sec"], "note end") - end) > 1e-8:
            raise ValueError("Score onset, offset and duration disagree")
        pitch = note_number(row["pitch"])
        first, last = round(start * scale), round(end * scale)
        if last <= first or first < previous_tick:
            raise ValueError("Notes cannot be represented without loss at this tick resolution")
        phones = row["v4c_phonemes"]
        if (not isinstance(phones, list) or not phones
                or not all(isinstance(p, str) and p.strip() for p in phones)
                or validate_miku_v4c_note_phonemes(phones)):
            raise ValueError("Missing or invalid Vocaloid phoneme sequence")
        if not isinstance(row["lyric"], str) or not row["lyric"]:
            raise ValueError("A VSQX note needs a displayed lyric or explicit placeholder")
        notes.append(VocalNoteDraft(id=str(i), start_tick=first, duration_ticks=last-first,
                                    pitch=pitch, lyric=row["lyric"], phonemes=phones))
        previous_end, previous_tick = end, last
    if not notes:
        raise ValueError("No notes to export; no fixed-pitch substitute was created")
    return notes


def _verify_written(file: Path, notes: list[VocalNoteDraft], score: dict, scale: float) -> dict:
    restored = parse_vsqx_timeline(file)["notes"]
    if len(restored) != len(notes):
        raise ValueError("VSQX note count failed independent readback")
    max_start = max_end = 0.0
    for original, draft, actual in zip(score["notes"], notes, restored, strict=True):
        if (actual["pitch"] != draft.pitch or actual["start_tick"] != draft.start_tick
                or actual["duration_ticks"] != draft.duration_ticks):
            raise ValueError("VSQX pitch/onset/duration changed during serialization")
        a = abs(actual["start_tick"] / scale - original["start_sec"])
        b = abs((actual["start_tick"] + actual["duration_ticks"]) / scale
                - original["start_sec"] - original["duration_sec"])
        if max(a, b) > .5 / scale + 1e-8:
            raise ValueError("VSQX timing exceeds one-half-tick rounding tolerance")
        max_start, max_end = max(max_start, a), max(max_end, b)
    return {"pitch_and_timing_readback_passed": True,
            "maximum_onset_roundtrip_error_sec": max_start,
            "maximum_offset_roundtrip_error_sec": max_end,
            "maximum_time_rounding_sec": .5 / scale,
            "maximum_duration_rounding_sec": 1 / scale}


def export_generated_score(score_path: Path, output: Path) -> dict:
    score_path, output = Path(score_path).resolve(), Path(output).resolve()
    proof_path = output.with_name(output.name + ".provenance.json")
    if output.suffix.lower() != ".vsqx":
        raise ValueError("VSQX output must use the .vsqx extension")
    if output.exists() or proof_path.exists():
        raise FileExistsError("Refusing to overwrite VSQX or provenance")
    digest = sha(score_path)
    score = read(score_path)
    if score.get("schema_version") != "local_note_score_v1":
        raise ValueError("Expected local_note_score_v1")
    tempo, resolution = score["tempo"], score["ticks_per_quarter"]
    scale = validate_grid(tempo, resolution)
    notes = _draft_notes(score, scale)
    draft = VocalTrackDraft(tempo=tempo, ticks_per_quarter=resolution, tracks=[
        VocalTrackDraftTrack(id="audio", name="Audio transcription", singer_target="miku_v4c", notes=notes)
    ])
    tree = build_vsqx_tree(draft, template_path=TEMPLATE)
    root = tree.getroot()
    ns = root.tag.split("}")[0] + "}"
    master = root.find(ns + "masterTrack")
    master.find(ns + "resolution").text = str(resolution)
    master.find(ns + "preMeasure").text = "0"
    part = root.find(".//" + ns + "vsPart")
    part.find(ns + "t").text = "0"
    part.find(ns + "name").text = "Audio transcription"
    # Preserve leading and trailing rest as empty project time.
    part.find(ns + "playTime").text = str(max(
        notes[-1].start_tick + notes[-1].duration_ticks,
        round(score.get("input", {}).get("end_sec", 0) * scale)))
    for node in root.findall(".//" + ns + "note"):
        node.find(ns + "p").set("lock", "1")
    if sha(score_path) != digest:
        raise ValueError("Score changed during export")
    output.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with output.open("xb") as handle:
            created.append(output)
            tree.write(handle, encoding="utf-8", xml_declaration=True)
        check = _verify_written(output, notes, score, scale)
        proof = {"schema_version": "local_generated_vsqx_export_v2",
                 "score_path": str(score_path), "score_sha256": digest,
                 "output_sha256": sha(output), "template_sha256": sha(TEMPLATE),
                 "note_count": len(notes), "pitch_preserved_from_analysis": True,
                 "transpose_semitones": 0, "tempo_grid": tempo, "ticks_per_quarter": resolution,
                 "time_basis": "source_audio_seconds", "ctc_times_used_as_note_durations": False,
                 "placeholder_note_count": score["placeholder_note_count"],
                 "unassigned_phone_count": score["unassigned_phone_count"],
                 "note_source": score["source"], "manual_review_required": False,
                 "manual_vocaloid_import_verified": False, "quality_validated": False,
                 "quality": "automatic_transcription", "model_forwards": 0, **check}
        write(proof_path, proof)
        created.append(proof_path)
    except BaseException:
        for file in created:
            file.unlink(missing_ok=True)
        raise
    return {"output": str(output), "provenance": str(proof_path), "note_count": len(notes),
            "placeholder_note_count": score["placeholder_note_count"], "model_forwards": 0, **check}
