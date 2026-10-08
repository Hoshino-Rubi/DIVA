from __future__ import annotations

from any2xsampa.cantonese.resolver import CantonesePhonemeResolver, load_lexicon_file, load_phoneme_evidence_file
from any2xsampa.cantonese.g2p import (
    CantoneseG2PAdapter,
    CantoneseG2PCandidate,
    CantoneseG2PError,
    ExternalCommandCantoneseG2PAdapter,
    PyCantoneseG2PAdapter,
    ResolvedLyricsItem,
    ReviewedLexiconCantoneseG2PAdapter,
    resolve_lyrics_timeline,
)
from any2xsampa.cantonese.types import CantoneseCandidate, CantoneseSyllableResolution

__all__ = [
    "CantoneseCandidate",
    "CantonesePhonemeResolver",
    "CantoneseSyllableResolution",
    "load_lexicon_file",
    "load_phoneme_evidence_file",
    "CantoneseG2PAdapter",
    "CantoneseG2PCandidate",
    "CantoneseG2PError",
    "ExternalCommandCantoneseG2PAdapter",
    "PyCantoneseG2PAdapter",
    "ResolvedLyricsItem",
    "ReviewedLexiconCantoneseG2PAdapter",
    "resolve_lyrics_timeline",
]
