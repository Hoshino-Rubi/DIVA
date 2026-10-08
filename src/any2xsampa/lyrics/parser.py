from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class LyricToken:
    index: int
    text: str


class LyricsParser:
    _pattern = re.compile(r"[A-Za-z0-9']+|[\u3400-\u9fff]")

    def parse(self, lyrics: str) -> list[LyricToken]:
        matches = self._pattern.findall(lyrics.lower())
        return [LyricToken(index=index, text=token) for index, token in enumerate(matches)]
