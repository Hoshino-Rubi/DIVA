from __future__ import annotations

from pathlib import Path

from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange
from any2xsampa.vocaloid.note_sliced_recognizer import extract_note_timing_windows, recognize_note_slices


class FakeBackend:
    def __init__(self) -> None:
        self.slice_lengths: list[int] = []

    def _read_pcm16_mono(self, audio_path: Path) -> tuple[int, list[int]]:
        return 1000, [100] * 2000

    def _predict_frames(
        self,
        samples: list[int],
        sample_rate: int,
        frame_size: int,
        hop_size: int,
    ) -> list[dict[str, object]]:
        self.slice_lengths.append(len(samples))
        return [
            {
                "start_index": 0,
                "end_index": len(samples),
                "candidates": [PhoneCandidate("a", 0.8), PhoneCandidate("i", 0.2)],
            }
        ]

    def _build_real_segment(self, symbol: str, windows: list[dict[str, object]], *args: object) -> PhoneSegment:
        end_index = int(windows[-1]["end_index"])
        return PhoneSegment(
            time_range=TimeRange(0.0, end_index / 1000.0),
            candidate_phones=[PhoneCandidate("a", 0.8), PhoneCandidate("i", 0.2)],
            confidence=0.8,
            metadata={"source_mode": "real", "real_backend": "onnx-espeak"},
        )

    def model_metadata(self) -> dict[str, str]:
        return {
            "model_backend": "fake_onnxruntime",
            "phone_mapping_profile": "test",
        }


def test_note_windows_exclude_lyrics_and_phonemes_from_decoder_contract(tmp_path: Path) -> None:
    vsqx_path = _write_vsqx(tmp_path / "reference.vsqx")

    timing_source, windows = extract_note_timing_windows(vsqx_path)

    assert len(windows) == 2
    assert not hasattr(windows[0], "lyric")
    assert not hasattr(windows[0], "phoneme")
    assert timing_source["fields_explicitly_excluded_from_decoding"] == ["lyric", "phoneme", "note_number"]


def test_note_sliced_recognizer_uses_only_timing_windows(tmp_path: Path) -> None:
    backend = FakeBackend()
    vsqx_path = _write_vsqx(tmp_path / "reference.vsqx")

    payload = recognize_note_slices(tmp_path / "audio.wav", vsqx_path, backend=backend)

    assert backend.slice_lengths == [500, 500]
    assert payload["reference_lyrics_used"] is False
    assert payload["reference_phonemes_used"] is False
    assert len(payload["segments"]) == 2
    assert payload["segments"][0]["start_sec"] == 0.5
    assert payload["segments"][0]["end_sec"] == 1.0
    assert payload["segments"][0]["metadata"]["reference_phoneme_excluded"] is True


def _write_vsqx(path: Path) -> Path:
    path.write_text(
        """<vsq4 xmlns="http://www.yamaha.co.jp/vocaloid/schema/vsq4/"><masterTrack><resolution>480</resolution><tempo><t>0</t><v>12000</v></tempo></masterTrack><vsTrack><tNo>0</tNo><vsPart><t>480</t><note><t>0</t><dur>480</dur><n>60</n><y>secret lyric</y><p>m a</p></note><note><t>480</t><dur>480</dur><n>62</n><y>another secret</y><p>t i</p></note></vsPart></vsTrack></vsq4>""",
        encoding="utf-8",
    )
    return path
