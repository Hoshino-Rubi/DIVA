from __future__ import annotations

"""Small, evidence-oriented phrase-source loader for resolver coverage tools."""

import json
from pathlib import Path
import re
from typing import Any, Iterable


_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
_TEXT_SUFFIXES = {".txt", ".lrc"}
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def _cjk_len(text: str) -> int:
    return sum(1 for char in text if _CJK_RE.match(char))


def _iter_input_files(inputs: Iterable[Path], *, include_audio_titles: bool) -> Iterable[Path]:
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            for child in sorted(item for item in path.rglob("*") if item.is_file()):
                if _supported(child, include_audio_titles=include_audio_titles):
                    yield child
        elif path.is_file() and _supported(path, include_audio_titles=include_audio_titles):
            yield path


def _supported(path: Path, *, include_audio_titles: bool) -> bool:
    suffix = path.suffix.lower()
    return suffix == ".json" or suffix in _TEXT_SUFFIXES or (
        include_audio_titles and suffix in _AUDIO_SUFFIXES
    )


def _rows_from_file(
    path: Path,
    *,
    min_chars: int,
    max_chars: int,
    include_audio_titles: bool,
) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if include_audio_titles and suffix in _AUDIO_SUFFIXES:
        lyrics = "".join(char for char in path.stem if _CJK_RE.match(char))
        if min_chars <= _cjk_len(lyrics) <= max_chars:
            return [{"lyrics": lyrics, "source": "audio_title_fallback", "source_path": str(path)}]
        return []
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return _rows_from_json(payload, path, min_chars=min_chars, max_chars=max_chars)
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines()):
        lyrics = line.strip()
        if min_chars <= _cjk_len(lyrics) <= max_chars:
            rows.append(
                {
                    "lyrics": lyrics,
                    "source": "text_line",
                    "source_path": str(path),
                    "source_index": index,
                }
            )
    return rows


def _rows_from_json(
    payload: Any,
    path: Path,
    *,
    min_chars: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        values = payload.get("phrases")
        if not isinstance(values, list):
            values = payload.get("segments")
        if not isinstance(values, list):
            values = [payload]
    elif isinstance(payload, list):
        values = payload
    else:
        values = []
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if isinstance(value, dict):
            lyrics = str(value.get("lyrics") or value.get("text") or "").strip()
            row = dict(value)
        else:
            lyrics = str(value).strip()
            row = {}
        if not min_chars <= _cjk_len(lyrics) <= max_chars:
            continue
        row.update(
            {
                "lyrics": lyrics,
                "source": row.get("source") or "phrase_batch_json",
                "source_path": str(path),
                "source_index": index,
            }
        )
        rows.append(row)
    return rows
