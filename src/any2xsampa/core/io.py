from __future__ import annotations

import codecs
import json
from pathlib import Path
from typing import Any
import uuid


_UTF8_BOM = codecs.BOM_UTF8
_UTF16_BOMS = (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    encodings = _preferred_text_encodings(raw)
    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        encodings[0],
        raw,
        0,
        min(len(raw), 1),
        f"Could not decode JSON text from {path} with supported encodings: {', '.join(encodings)}",
    )


def dump_json(payload: Any, *, ensure_ascii: bool = False, indent: int | None = 2) -> str:
    return json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent, allow_nan=False)


def write_text(path: Path, text: str, *, atomic: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not atomic:
        path.write_text(text, encoding="utf-8")
        return
    temp_path = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def write_json(
    path: Path,
    payload: Any,
    *,
    ensure_ascii: bool = False,
    indent: int | None = 2,
    append_newline: bool = False,
    atomic: bool = False,
) -> None:
    text = dump_json(payload, ensure_ascii=ensure_ascii, indent=indent)
    if append_newline:
        text += "\n"
    write_text(path, text, atomic=atomic)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(read_text(path))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload at {path} must contain an object at the top level.")
    return payload


def _preferred_text_encodings(raw: bytes) -> tuple[str, ...]:
    if raw.startswith(_UTF8_BOM):
        return ("utf-8-sig", "utf-8")
    if raw.startswith(_UTF16_BOMS):
        return ("utf-16", "utf-16-le", "utf-16-be")
    if _looks_utf16(raw):
        return ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be")
    return ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be")


def _looks_utf16(raw: bytes) -> bool:
    window = raw[:8]
    if len(window) < 2:
        return False
    return any(byte == 0 for byte in window)
