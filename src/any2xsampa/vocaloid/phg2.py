"""Functional recovery of the public Miku V4C PHG2 token parser.

The source and module bytecode in the original tree are not valid Python.  This
module keeps the small API exposed by ``any2xsampa.vocaloid`` while returning
only the public token table; DDI headers and trailing metadata are never
included in the result.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PHG2_SCHEMA_VERSION = "miku_v4c_phg2_tokens_v1"


class PHG2ParseError(ValueError):
    """Raised when a PHG2 section cannot be decoded safely."""


_MAX_GROUPS = 16_384
_MAX_TOKENS_PER_GROUP = 1_000_000
_MAX_TOTAL_TOKENS = 2_000_000
_MAX_FIELD_BYTES = 16 * 1024 * 1024
_MAX_SECTION_BYTES = 128 * 1024 * 1024
_JSON_INVALID = object()


def _read_u32(body: bytes, offset: int, label: str) -> tuple[int, int]:
    if offset < 0 or offset + 4 > len(body):
        raise PHG2ParseError(f"truncated PHG2 {label}")
    return int.from_bytes(body[offset : offset + 4], "little"), offset + 4


def _read_ascii(body: bytes, offset: int, size: int, label: str) -> tuple[str, int]:
    if size > _MAX_FIELD_BYTES:
        raise PHG2ParseError(f"PHG2 {label} is too large")
    end = offset + size
    if offset < 0 or end > len(body):
        raise PHG2ParseError(f"truncated PHG2 {label}")
    return body[offset:end].decode("ascii", "replace"), end


def _token_values(value: object) -> list[str]:
    """Normalize token strings and the record shape in historical dumps."""

    if isinstance(value, str):
        return [part for part in value.split() if part]
    if isinstance(value, Mapping):
        token = value.get("token")
        if token is None:
            token = value.get("phoneme")
        if token is None:
            token = value.get("name")
        return _token_values(token) if token is not None else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, memoryview)):
        result: list[str] = []
        for item in value:
            result.extend(_token_values(item))
        return result
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        text = str(value)
        return [text] if text else []
    return []


def _clean_groups(value: object) -> list[dict[str, Any]]:
    """Return the stable group shape without copying arbitrary DDI metadata."""

    if isinstance(value, Mapping):
        value = [{"name": name, "tokens": tokens} for name, tokens in value.items()]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray, memoryview)):
        return []
    if len(value) > _MAX_GROUPS:
        raise PHG2ParseError("PHG2 group count is unreasonable")

    groups: list[dict[str, Any]] = []
    total_tokens = 0
    for index, item in enumerate(value):
        if isinstance(item, Mapping):
            name_value = item.get("name")
            if name_value is None:
                name_value = item.get("group")
            name = str(name_value).strip() if name_value is not None else ""
            if not name:
                name = f"group-{index + 1}"
            raw_tokens = item.get("tokens")
            if raw_tokens is None:
                raw_tokens = item.get("phonemes")
            if raw_tokens is None:
                raw_tokens = item.get("items", [])
        else:
            name = f"group-{index + 1}"
            raw_tokens = item
        tokens = _token_values(raw_tokens)
        if len(tokens) > _MAX_TOKENS_PER_GROUP:
            raise PHG2ParseError("PHG2 token count is unreasonable")
        total_tokens += len(tokens)
        if total_tokens > _MAX_TOTAL_TOKENS:
            raise PHG2ParseError("PHG2 total token count is unreasonable")
        groups.append({"name": name, "tokens": tokens, "token_count": len(tokens)})
    return groups


def _parse_binary_table(body: bytes) -> list[dict[str, Any]]:
    """Parse the observed little-endian PHG2 group/token table.

    The table is ``count`` followed by ``name_len/name/count/unknown`` for each
    group, then ``token_len/token/token_id`` for each token.  IDs are evidence
    metadata and are intentionally discarded from the public result.
    """

    group_count, offset = _read_u32(body, 0, "group count")
    if group_count > _MAX_GROUPS:
        raise PHG2ParseError("PHG2 group count is unreasonable")

    groups: list[dict[str, Any]] = []
    for index in range(group_count):
        name_len, offset = _read_u32(body, offset, "group name length")
        name, offset = _read_ascii(body, offset, name_len, "group name")
        token_count, offset = _read_u32(body, offset, "token count")
        if token_count > _MAX_TOKENS_PER_GROUP:
            raise PHG2ParseError("PHG2 token count is unreasonable")
        _unknown, offset = _read_u32(body, offset, "group metadata")

        tokens: list[str] = []
        for _ in range(token_count):
            token_len, offset = _read_u32(body, offset, "token length")
            token, offset = _read_ascii(body, offset, token_len, "token")
            _token_id, offset = _read_u32(body, offset, "token metadata")
            if token:
                tokens.append(token)
        groups.append(
            {
                "name": name or f"group-{index + 1}",
                "tokens": tokens,
                "token_count": len(tokens),
            }
        )
    return groups


def _decode_json(body: bytes) -> object:
    try:
        return json.loads(body.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _JSON_INVALID


def _flatten_tokens(groups: Sequence[Mapping[str, Any]]) -> list[str]:
    tokens: list[str] = []
    for group in groups:
        for token in _token_values(group.get("tokens", [])):
            if token not in tokens:
                tokens.append(token)
    return tokens


def parse_phg2_bytes(data: bytes | bytearray | memoryview) -> dict[str, Any]:
    """Parse a DDI byte payload and return its public PHG2 token schema."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    raw = bytes(data)
    marker = raw.find(b"PHG2")
    if marker < 0:
        raise PHG2ParseError("PHG2 section marker not found")
    declared, body_start = _read_u32(raw, marker + 4, "section length")
    body_end = body_start + declared
    if declared > _MAX_SECTION_BYTES:
        raise PHG2ParseError("PHG2 section is too large")
    if body_end > len(raw):
        raise PHG2ParseError("PHG2 section is truncated")
    body = raw[body_start:body_end]

    payload = _decode_json(body)
    if payload is _JSON_INVALID:
        groups = _parse_binary_table(body)
    elif isinstance(payload, Mapping):
        groups = _clean_groups(payload.get("groups", []))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray, memoryview)):
        groups = _clean_groups(payload)
    else:
        groups = []

    return {
        "schema_version": PHG2_SCHEMA_VERSION,
        "section": {
            "magic": "PHG2",
            "offset": marker,
            "declared_length": declared,
            "parsed_length": len(body),
        },
        "groups": groups,
        "tokens": _flatten_tokens(groups),
    }


def parse_phg2_path(path: str | Path) -> dict[str, Any]:
    """Read and parse a PHG2-bearing DDI path."""

    return parse_phg2_bytes(Path(path).read_bytes())


def parse_phg2(source: str | Path | bytes | bytearray | memoryview) -> dict[str, Any]:
    """Dispatch to bytes or path parsing while preserving the public API."""

    if isinstance(source, (bytes, bytearray, memoryview)):
        return parse_phg2_bytes(source)
    return parse_phg2_path(source)


__all__ = [
    "PHG2_SCHEMA_VERSION",
    "PHG2ParseError",
    "parse_phg2",
    "parse_phg2_bytes",
    "parse_phg2_path",
]
