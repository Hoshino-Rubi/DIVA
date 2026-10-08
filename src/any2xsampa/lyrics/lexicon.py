from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from any2xsampa.core.io import load_json
from any2xsampa.lyrics.parser import LyricToken
from any2xsampa.lyrics.oov import DialectOOVGuesser


@dataclass(slots=True)
class LexiconEntry:
    token: LyricToken
    phones: list[str]
    hit: bool
    source: str = "lexicon"
    notes: list[str] | None = None


class DialectLexiconRepository:
    def __init__(self, config_root: Path) -> None:
        self.config_root = config_root
        self._cache: dict[str, dict[str, list[str]]] = {}
        self._oov_guesser = DialectOOVGuesser(config_root)

    def lookup_tokens(self, dialect: str, tokens: list[LyricToken]) -> list[LexiconEntry]:
        lexicon = self._load_lexicon(dialect)
        entries: list[LexiconEntry] = []
        for token in tokens:
            phones = list(lexicon.get(token.text, []))
            if phones:
                entries.append(LexiconEntry(token=token, phones=phones, hit=True, source="lexicon", notes=[]))
                continue

            guess = self._oov_guesser.guess(dialect, token.text)
            entries.append(
                LexiconEntry(
                    token=token,
                    phones=guess.phones,
                    hit=False,
                    source=guess.source,
                    notes=guess.notes,
                )
            )
        return entries

    def _load_lexicon(self, dialect: str) -> dict[str, list[str]]:
        key = dialect.lower()
        if key not in self._cache:
            path = self.config_root / f"lexicon_{key}.json"
            if not path.exists():
                self._cache[key] = {}
            else:
                payload = load_json(path)
                self._cache[key] = {str(token).lower(): list(phones) for token, phones in payload.get("entries", {}).items()}
        return self._cache[key]
