from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from any2xsampa.core.io import load_json


@dataclass(slots=True)
class OOVGuess:
    phones: list[str]
    source: str
    notes: list[str]


class DialectOOVGuesser:
    def __init__(self, config_root: Path) -> None:
        self.config_root = config_root
        self._spec_cache: dict[str, dict[str, object]] = {}
        self._inventory_cache: dict[str, set[str]] = {}

    def guess(self, dialect: str, token: str) -> OOVGuess:
        spec = self._load_spec(dialect)
        if not spec:
            return OOVGuess(phones=[], source="placeholder_oov", notes=["no oov guesser configured"])

        exact_entries = self._string_map(spec, "exact_entries")
        if token in exact_entries:
            phones = self._filter_allowed(dialect, exact_entries[token])
            if phones:
                return OOVGuess(phones=phones, source="dialect_oov_guesser", notes=["matched exact OOV entry"])

        phones, notes = self._scan_token(dialect, token, spec)
        if not phones:
            notes.append("no usable phones guessed")
            return OOVGuess(phones=[], source="placeholder_oov", notes=notes)
        return OOVGuess(phones=phones, source="dialect_oov_guesser", notes=notes)

    def _scan_token(self, dialect: str, token: str, spec: dict[str, object]) -> tuple[list[str], list[str]]:
        multigraphs = self._string_map(spec, "multigraph_map")
        single_graphs = self._string_map(spec, "single_graph_map")
        drop_chars = set(self._string_list(spec, "drop_chars"))
        ordered_multigraphs = sorted(multigraphs, key=len, reverse=True)

        phones: list[str] = []
        notes: list[str] = []
        index = 0
        while index < len(token):
            chunk = self._match_multigraph(token, index, ordered_multigraphs)
            if chunk is not None:
                phones.extend(multigraphs[chunk])
                index += len(chunk)
                continue

            char = token[index]
            if char in single_graphs:
                phones.extend(single_graphs[char])
            elif char in drop_chars:
                notes.append(f"dropped grapheme={char}")
            else:
                notes.append(f"unmapped grapheme={char}")
            index += 1

        filtered = self._filter_allowed(dialect, phones)
        if len(filtered) != len(phones):
            notes.append("removed phones outside dialect inventory")
        return filtered, notes

    def _match_multigraph(self, token: str, index: int, ordered_multigraphs: list[str]) -> str | None:
        remaining = token[index:]
        for grapheme in ordered_multigraphs:
            if remaining.startswith(grapheme):
                return grapheme
        return None

    def _filter_allowed(self, dialect: str, phones: list[str]) -> list[str]:
        allowed = self._load_inventory(dialect)
        return [phone for phone in phones if phone in allowed]

    def _load_spec(self, dialect: str) -> dict[str, object]:
        key = dialect.lower()
        if key not in self._spec_cache:
            path = self.config_root / f"oov_guesser_{key}.json"
            self._spec_cache[key] = load_json(path) if path.exists() else {}
        return self._spec_cache[key]

    def _load_inventory(self, dialect: str) -> set[str]:
        key = dialect.lower()
        if key not in self._inventory_cache:
            path = self.config_root / f"dialect_{key}.json"
            payload = load_json(path) if path.exists() else {}
            self._inventory_cache[key] = {str(phone) for phone in payload.get("allowed_ipa", [])}
        return self._inventory_cache[key]

    def _string_map(self, spec: dict[str, object], key: str) -> dict[str, list[str]]:
        raw = spec.get(key, {})
        if not isinstance(raw, dict):
            return {}
        mapping: dict[str, list[str]] = {}
        for grapheme, value in raw.items():
            if isinstance(value, str):
                mapping[str(grapheme)] = [value]
            elif isinstance(value, list):
                mapping[str(grapheme)] = [str(item) for item in value]
        return mapping

    def _string_list(self, spec: dict[str, object], key: str) -> list[str]:
        raw = spec.get(key, [])
        if not isinstance(raw, list):
            return []
        return [str(item) for item in raw]
