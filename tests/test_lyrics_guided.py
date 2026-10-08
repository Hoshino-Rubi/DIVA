from pathlib import Path

from any2xsampa.adapters.recognizer import RecognizerConfig
from any2xsampa.lyrics.lexicon import DialectLexiconRepository
from any2xsampa.lyrics.parser import LyricToken
from any2xsampa.pipeline.runner import SingingToPhonesPipeline


def _pipeline_with_fixture(fixture_name: str) -> SingingToPhonesPipeline:
    fixture_path = Path(__file__).resolve().parent / "fixtures" / fixture_name
    return SingingToPhonesPipeline(
        recognizer_config=RecognizerConfig(
            mode="fixture",
            fixture_path=fixture_path,
            allow_stub_fallback=False,
        )
    )


def test_infer_with_lyrics_preserves_recognizer_truth_and_adds_candidate_metadata(tmp_path: Path) -> None:
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"")
    pipeline = _pipeline_with_fixture("lyrics_constrained_fixture.json")

    open_result = pipeline.run(str(audio_path), dialect="taigi", mode="infer-open")
    guided_result = pipeline.run(str(audio_path), dialect="taigi", mode="infer-with-lyrics", lyrics="li ng")

    assert [segment.normalized_ipa for segment in open_result.segments] == ["l", "a", "N"]
    assert [segment.normalized_ipa for segment in guided_result.segments] == ["l", "a", "N"]
    assert [segment.xsampa for segment in guided_result.segments] == ["l", "a", "N"]
    assert guided_result.segments[1].metadata["lyric_candidate_phone"] == "i"
    assert guided_result.segments[1].metadata["lyric_candidate_replaces_recognizer_truth"] is False


def test_cantonese_lexicon_hit(tmp_path: Path) -> None:
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"")
    pipeline = _pipeline_with_fixture("lyrics_constrained_fixture.json")

    result = pipeline.run(str(audio_path), dialect="cantonese", mode="infer-with-lyrics", lyrics="laa ng")

    assert result.segments[0].metadata["lyric_token"] == "laa"
    assert result.segments[0].metadata["dialect_lexicon_hit"] is True
    assert result.segments[2].metadata["lyric_token"] == "ng"


def test_taigi_lexicon_hit(tmp_path: Path) -> None:
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"")
    pipeline = _pipeline_with_fixture("lyrics_constrained_fixture.json")

    result = pipeline.run(str(audio_path), dialect="taigi", mode="infer-with-lyrics", lyrics="li ng")

    assert result.segments[1].metadata["lyric_token"] == "li"
    assert result.segments[1].metadata["dialect_lexicon_hit"] is True
    assert result.segments[1].normalized_ipa == "a"
    assert result.segments[1].metadata["lyric_candidate_phone"] == "i"


def test_oov_fallback_is_predictable(tmp_path: Path) -> None:
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"")
    pipeline = _pipeline_with_fixture("lyrics_constrained_fixture.json")

    open_result = pipeline.run(str(audio_path), dialect="cantonese", mode="infer-open")
    guided_result = pipeline.run(str(audio_path), dialect="cantonese", mode="infer-with-lyrics", lyrics="zzz")

    assert [segment.normalized_ipa for segment in guided_result.segments] == [segment.normalized_ipa for segment in open_result.segments]
    assert guided_result.segments[0].metadata["dialect_lexicon_hit"] is False
    assert guided_result.segments[0].metadata["lyric_constraint_applied"] is False


def test_oov_guesser_is_dialect_specific() -> None:
    config_root = Path(__file__).resolve().parent.parent / "src" / "any2xsampa" / "config"
    repository = DialectLexiconRepository(config_root)

    cantonese_entry = repository.lookup_tokens("cantonese", [LyricToken(index=0, text="tsi")])[0]
    taigi_entry = repository.lookup_tokens("taigi", [LyricToken(index=0, text="tsi")])[0]

    assert cantonese_entry.hit is False
    assert taigi_entry.hit is False
    assert cantonese_entry.source == "dialect_oov_guesser"
    assert taigi_entry.source == "dialect_oov_guesser"
    assert cantonese_entry.phones == ["ts", "i"]
    assert taigi_entry.phones == ["ts_h", "i"]


def test_oov_guesser_can_soft_constrain_pipeline(tmp_path: Path) -> None:
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"")
    pipeline = _pipeline_with_fixture("lyrics_constrained_fixture.json")

    open_result = pipeline.run(str(audio_path), dialect="cantonese", mode="infer-open")
    guided_result = pipeline.run(str(audio_path), dialect="cantonese", mode="infer-with-lyrics", lyrics="li")

    assert [segment.normalized_ipa for segment in open_result.segments] == ["l", "a", "N"]
    assert [segment.normalized_ipa for segment in guided_result.segments] == ["l", "a", "N"]
    assert guided_result.segments[1].metadata["dialect_lexicon_hit"] is False
    assert guided_result.segments[1].metadata["lyric_constraint_applied"] is True
    assert guided_result.segments[1].metadata["oov_fallback_used"] is True
    assert guided_result.segments[1].metadata["oov_fallback_source"] == "dialect_oov_guesser"


def test_output_metadata_contains_lyric_alignment_info(tmp_path: Path) -> None:
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"")
    pipeline = _pipeline_with_fixture("lyrics_constrained_fixture.json")

    result = pipeline.run(str(audio_path), dialect="taigi", mode="infer-with-lyrics", lyrics="li ng")
    metadata = result.segments[1].metadata

    assert "lyric_token" in metadata
    assert "lyric_constraint_applied" in metadata
    assert "dialect_lexicon_hit" in metadata
    assert "alignment_notes" in metadata
    assert "alignment_score" in metadata
    assert "alignment_operation" in metadata
    assert "target_phone" in metadata
    assert "oov_fallback_used" in metadata
    assert "oov_fallback_source" in metadata
    assert "lyric_candidate_phone" in metadata
    assert metadata["lyric_candidate_replaces_recognizer_truth"] is False


def test_dp_alignment_handles_length_mismatch_with_merge(tmp_path: Path) -> None:
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"")
    pipeline = _pipeline_with_fixture("lyrics_dp_mismatch_fixture.json")

    result = pipeline.run(str(audio_path), dialect="cantonese", mode="infer-with-lyrics", lyrics="laa ng")

    assert [segment.normalized_ipa for segment in result.segments] == ["l", "a", "N", "N"]
    assert result.segments[2].metadata["lyric_token"] == "ng"
    assert result.segments[2].metadata["alignment_operation"] == "merge_segments"
    assert result.segments[3].metadata["alignment_operation"] == "merge_segments"


def test_dp_alignment_supports_split_target(tmp_path: Path) -> None:
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"")
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "lyrics_dp_split_fixture.json"
    pipeline = SingingToPhonesPipeline(
        recognizer_config=RecognizerConfig(
            mode="fixture",
            fixture_path=fixture_path,
            allow_stub_fallback=False,
        )
    )

    result = pipeline.run(str(audio_path), dialect="taigi", mode="infer-with-lyrics", lyrics="li")

    assert len(result.segments) == 1
    assert result.segments[0].metadata["alignment_operation"] == "split_segment"
    assert result.segments[0].metadata["lyric_token"] == "li"
