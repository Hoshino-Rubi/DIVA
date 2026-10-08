from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from any2xsampa.core.interfaces import VocalSeparator


@dataclass(frozen=True)
class SeparationTrace:
    """Machine-readable trace for separator decisions."""

    adapter: str
    adapter_kind: str
    input_audio: str
    output_audio: str
    event: str
    reason: str
    separation_requested: bool
    separated: bool

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "adapter": self.adapter,
            "adapter_kind": self.adapter_kind,
            "input_audio": self.input_audio,
            "output_audio": self.output_audio,
            "event": self.event,
            "reason": self.reason,
            "separation_requested": self.separation_requested,
            "separated": self.separated,
        }


class PassthroughVocalSeparator(VocalSeparator):
    """Real adapter for the explicit no-separation path.

    This adapter is not a stub and does not simulate separator output. It is the
    production path used when upstream configuration requests recognition on the
    original input audio without vocal isolation.
    """

    adapter_kind: ClassVar[str] = "real"
    trace_event: ClassVar[str] = "input_audio_passthrough"
    trace_reason: ClassVar[str] = "separation_not_requested"

    def separate(self, audio_path: Path) -> Path:
        self._validate_audio_path(audio_path)
        return audio_path

    def trace(self, audio_path: Path) -> dict[str, str | bool]:
        """Return trace metadata for the no-separation decision."""

        self._validate_audio_path(audio_path)
        return self._build_trace(audio_path).to_dict()

    def metadata(self, audio_path: Path) -> dict[str, str | bool]:
        """Alias for callers that attach separator decisions to metadata."""

        return self.trace(audio_path)

    @classmethod
    def _build_trace(cls, audio_path: Path) -> SeparationTrace:
        audio = str(audio_path)
        return SeparationTrace(
            adapter=cls.__name__,
            adapter_kind=cls.adapter_kind,
            input_audio=audio,
            output_audio=audio,
            event=cls.trace_event,
            reason=cls.trace_reason,
            separation_requested=False,
            separated=False,
        )

    @staticmethod
    def _validate_audio_path(audio_path: Path) -> None:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
