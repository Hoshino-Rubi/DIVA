"""Assemble a V4 score from source-clock GAME candidates and continuous vocal evidence."""
from __future__ import annotations

from pathlib import Path
import soundfile as sf

from .common import read
from any2xsampa.music.melody_cleanup import clean_melody


def clean_rows(pitch: dict, duration: float, v5: bool = False, *, v6: bool = False) -> tuple[list[dict], dict, dict]:
    from any2xsampa.music.melody_cleanup import game_rows
    policy = read(Path(__file__).resolve().parents[1] / "config/local_score_v1.json")
    clean = clean_melody(pitch, policy["tempo_default"] * policy["ticks_per_quarter"] / 60,v5=v5,v6=v6)
    return game_rows(clean, duration), clean["melody_cleanup"], policy


def singing_score(audio: Path, vocal, frames, pitch: dict, checkpoint: Path, separated: bool, *, prepared=None) -> dict:
    import torch
    state = torch.load(checkpoint,weights_only=True,map_location="cpu")
    if state["format"] != "diva_singing_v5":
        raise ValueError("Expected the frozen V5 base checkpoint")
    duration = sf.info(audio).duration
    rows,audit,policy = prepared if prepared is not None else clean_rows(pitch, duration, v5=True)
    from ..adapters.singing_v5 import predict_v5
    predictions = predict_v5(rows,frames,vocal,state)
    for row, evidence in zip(rows,predictions,strict=True):
        row.update(v4c_phonemes=evidence["v4c_phonemes"], lyric=evidence["lyric"],
                   acoustic_prediction=evidence, recognition_source="joint_mask_v5",placeholder=False)
    return dict(schema_version="local_note_score_v1",source="audio_game",recognition_mode="singing_v5",
                input={"path":str(audio.resolve()),"start_sec":0.,"end_sec":duration},
                separated=separated,tempo=policy["tempo_default"],ticks_per_quarter=policy["ticks_per_quarter"],
                notes=rows,melody_cleanup=audit,adapter=str(checkpoint.resolve()),placeholder_note_count=0,
                unassigned_phone_count=0,acoustic_note_count=len(rows))
