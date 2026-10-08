"""Typed singing-to-phones orchestration contract.

This is a functional reconstruction.  Adapter implementations remain
replaceable; the runner only owns stage ordering and PipelineResult shaping.
"""

from __future__ import annotations

from pathlib import Path

from any2xsampa.adapters.aligner import SegmentScorePhoneAligner
from any2xsampa.adapters.recognizer import ConfigurableUniversalPhoneRecognizer, RecognizerConfig
from any2xsampa.adapters.separator import PassthroughVocalSeparator
from any2xsampa.core.models import PipelineResult, PipelineTrace
from any2xsampa.dialects.registry import DialectRegistry
from any2xsampa.exporters.registry import ExporterRegistry
from any2xsampa.exporters.xsampa import JsonXSampaExporter
from any2xsampa.lyrics.decoder import LyricsConstrainedDecoder
from any2xsampa.pipeline.future_aligner import ScoreHintAcousticAligner
from any2xsampa.pipeline.future_aligner_sequence_hint import FutureAlignerSequenceHintBuilder
from any2xsampa.pipeline.normalizer import ConfigurablePhoneNormalizer
from any2xsampa.pipeline.note_context import NoteContextApplier
from any2xsampa.pipeline.score_hint import ScoreHintLayerBuilder
from any2xsampa.pipeline.score_identity import ScoreBridgeIdentityProducer, SegmentScoreIdentityNormalizer


class SingingToPhonesPipeline:
    """Run separation, recognition, phone normalization, and export stages."""

    def __init__(self, recognizer_config: RecognizerConfig | None = None) -> None:
        package_root = Path(__file__).resolve().parent.parent
        config_root = package_root / "config"
        fixture_root = package_root / "recognizer_fixtures"
        self.separator = PassthroughVocalSeparator()
        self.recognizer_config = recognizer_config or RecognizerConfig(
            mode="fixture",
            fixture_path=fixture_root / "default_fixture.json",
            real_output_path=Path("real_recognizer_output.json"),
            allow_stub_fallback=False,
        )
        self.recognizer = ConfigurableUniversalPhoneRecognizer(self.recognizer_config)
        self.normalizer = ConfigurablePhoneNormalizer(config_root / "ipa_aliases.json")
        self.lyrics_decoder = LyricsConstrainedDecoder(config_root)
        self.note_context_applier = NoteContextApplier()
        self.score_bridge_identity_producer = ScoreBridgeIdentityProducer()
        self.score_identity_normalizer = SegmentScoreIdentityNormalizer()
        self.score_hint_builder = ScoreHintLayerBuilder()
        self.future_aligner = ScoreHintAcousticAligner()
        self.future_aligner_sequence_hint_builder = FutureAlignerSequenceHintBuilder()
        self.registry = DialectRegistry()
        self.xsampa_exporter = JsonXSampaExporter(config_root / "ipa_to_xsampa.json")
        self.exporter_registry = ExporterRegistry(config_root)
        self.aligner = SegmentScorePhoneAligner(self.future_aligner)

    def run(
        self,
        audio_path: str,
        dialect: str,
        mode: str = "infer-open",
        lyrics: str | None = None,
        export_format: str = "xsampa",
        engine_profile: str | None = None,
        engine_inventory: str | None = None,
        engine_metadata_preferences: dict[str, list[str]] | None = None,
        note_context: str | None = None,
        note_context_format: str | None = None,
    ) -> PipelineResult:
        audio = Path(audio_path)
        vocal_audio = self.separator.separate(audio)
        # Lyrics are evidence for the constrained decoder, not recognizer truth.
        segments = self.recognizer.recognize(vocal_audio)
        segments = self.normalizer.normalize(segments)
        if mode == "infer-with-lyrics" and lyrics:
            segments = self.lyrics_decoder.apply(segments, lyrics=lyrics, dialect=dialect)
        segments = self.registry.get(dialect).apply(segments)
        segments = self.score_identity_normalizer.apply(segments)

        note_context_path = Path(note_context) if note_context else None
        note_context_warnings = self.note_context_applier.apply(
            segments,
            note_context_path,
            note_context_format=note_context_format,
        )
        if note_context_path is not None:
            segments = self.score_bridge_identity_producer.apply(segments)
            segments = self.score_identity_normalizer.apply(segments)
            note_context_warnings = self.note_context_applier.apply(
                segments,
                note_context_path,
                note_context_format=note_context_format,
            )

        segments = self.score_hint_builder.apply(segments)
        segments = self.aligner.align_segments(segments)
        segments = self.future_aligner.apply(segments)
        segments = self.future_aligner_sequence_hint_builder.apply(segments)
        segments = self.xsampa_exporter.export(segments)
        engine_exporter = self.exporter_registry.get(
            export_format,
            engine_profile=engine_profile,
            engine_inventory=Path(engine_inventory) if engine_inventory else None,
            engine_metadata_preferences=engine_metadata_preferences,
        )
        if engine_exporter is not None:
            segments = engine_exporter.export(segments)

        result = PipelineResult(
            mode=mode,
            audio_path=str(audio),
            dialect=dialect,
            lyrics=lyrics,
            trace=PipelineTrace(
                separation_adapter=self.separator.__class__.__name__,
                recognizer_adapter=self.recognizer._build_primary().__class__.__name__,
                normalizer=self.normalizer.__class__.__name__,
                dialect_plugin=self.registry.get(dialect).__class__.__name__,
                exporter=self.xsampa_exporter.__class__.__name__
                if engine_exporter is None
                else f"{self.xsampa_exporter.__class__.__name__}+{engine_exporter.__class__.__name__}",
                aligner=self.aligner.__class__.__name__,
            ),
            segments=segments,
            warnings=note_context_warnings,
        )
        return result


__all__ = ["SingingToPhonesPipeline"]
