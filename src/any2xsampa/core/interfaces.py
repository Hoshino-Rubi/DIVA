from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from any2xsampa.core.models import DialectConstraintResult, PhoneSegment, PipelineResult


class VocalSeparator(ABC):
    @abstractmethod
    def separate(self, audio_path: Path) -> Path:
        """Return a path to the isolated vocal track."""


class UniversalPhoneRecognizer(ABC):
    @abstractmethod
    def recognize(self, vocal_audio_path: Path, lyrics: str | None = None) -> list[PhoneSegment]:
        """Return time-aligned phone candidates."""


class PhoneNormalizer(ABC):
    @abstractmethod
    def normalize(self, segments: list[PhoneSegment]) -> list[PhoneSegment]:
        """Normalize recognizer output into canonical IPA-like symbols."""


class DialectPlugin(ABC):
    name: str

    @abstractmethod
    def apply(self, segments: list[PhoneSegment]) -> list[PhoneSegment]:
        """Constrain normalized phones with dialect-specific rules."""

    @abstractmethod
    def describe(self) -> dict[str, str]:
        """Return a small metadata payload for registry or CLI display."""


class XSampaExporter(ABC):
    @abstractmethod
    def export(self, segments: list[PhoneSegment]) -> list[PhoneSegment]:
        """Fill X-SAMPA values for segments."""


class EnginePhoneExporter(ABC):
    @abstractmethod
    def export(self, segments: list[PhoneSegment]) -> list[PhoneSegment]:
        """Fill engine-target phone exports for segments."""


class PhoneAligner(ABC):
    @abstractmethod
    def align(self, result: PipelineResult) -> PipelineResult:
        """Hook for future MFA / SOFA / singing alignment support."""
