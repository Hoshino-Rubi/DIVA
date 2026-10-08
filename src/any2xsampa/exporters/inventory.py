from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Protocol

from any2xsampa.core.io import load_json


FeatureMap = dict[str, str]
_NOTE_OFFSETS = {
    "c": 0,
    "c#": 1,
    "db": 1,
    "d": 2,
    "d#": 3,
    "eb": 3,
    "e": 4,
    "f": 5,
    "f#": 6,
    "gb": 6,
    "g": 7,
    "g#": 8,
    "ab": 8,
    "a": 9,
    "a#": 10,
    "bb": 10,
    "b": 11,
}


def normalize_note_midi(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(round(float(text)))
    except ValueError:
        pass
    match = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", text)
    if match is None:
        return None
    note_name = f"{match.group(1).lower()}{match.group(2).lower()}"
    offset = _NOTE_OFFSETS.get(note_name)
    if offset is None:
        return None
    octave = int(match.group(3))
    return ((octave + 1) * 12) + offset


def normalize_timing_value(value: object) -> str | None:
    normalized = _normalized_metadata_value(value)
    if normalized in {"short", "fast", "quick", "tight"}:
        return "short"
    if normalized in {"medium", "mid", "balanced", "normal"}:
        return "medium"
    if normalized in {"long", "slow", "tail", "sustain", "sustained"}:
        return "long"
    return normalized


def normalize_release_value(value: object) -> str | None:
    normalized = _normalized_metadata_value(value)
    if normalized in {"short", "tight", "clipped", "staccato"}:
        return "short"
    if normalized in {"long", "tail", "linger", "sustain", "sustained", "legato"}:
        return "long"
    return normalized


def _normalized_duration_ms(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        duration_ms = float(value)
    except (TypeError, ValueError):
        return None
    if duration_ms < 0:
        return None
    return duration_ms


def _normalized_note_range_fields(raw: dict[str, object]) -> dict[str, object]:
    note_range = raw.get("note_range")
    min_note = raw.get("min_note", raw.get("note_min"))
    max_note = raw.get("max_note", raw.get("note_max"))
    min_note_midi_raw = raw.get("min_note_midi", raw.get("min_midi"))
    max_note_midi_raw = raw.get("max_note_midi", raw.get("max_midi"))
    if isinstance(note_range, dict):
        min_note = note_range.get("min_note", note_range.get("min", min_note))
        max_note = note_range.get("max_note", note_range.get("max", max_note))
        min_note_midi_raw = note_range.get("min_note_midi", note_range.get("min_midi", min_note_midi_raw))
        max_note_midi_raw = note_range.get("max_note_midi", note_range.get("max_midi", max_note_midi_raw))
    elif isinstance(note_range, list) and len(note_range) >= 2:
        min_note = min_note if min_note is not None else note_range[0]
        max_note = max_note if max_note is not None else note_range[1]
    elif isinstance(note_range, str) and ".." in note_range:
        left, right = note_range.split("..", 1)
        min_note = min_note if min_note is not None else left
        max_note = max_note if max_note is not None else right
    elif isinstance(note_range, str) and "-" in note_range:
        left, right = note_range.split("-", 1)
        min_note = min_note if min_note is not None else left
        max_note = max_note if max_note is not None else right

    normalized: dict[str, object] = {}
    normalized_min_note = _normalized_metadata_value(min_note)
    normalized_max_note = _normalized_metadata_value(max_note)
    min_note_midi = normalize_note_midi(min_note_midi_raw if min_note_midi_raw is not None else min_note)
    max_note_midi = normalize_note_midi(max_note_midi_raw if max_note_midi_raw is not None else max_note)
    if normalized_min_note:
        normalized["min_note"] = normalized_min_note
    if normalized_max_note:
        normalized["max_note"] = normalized_max_note
    if min_note_midi is not None:
        normalized["min_note_midi"] = min_note_midi
    if max_note_midi is not None:
        normalized["max_note_midi"] = max_note_midi
    return normalized


def _normalized_behavior_metadata(raw: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    normalized.update(_normalized_note_range_fields(raw))
    timing = normalize_timing_value(raw.get("timing", raw.get("timing_group", raw.get("timing_class"))))
    release = normalize_release_value(raw.get("release", raw.get("release_type", raw.get("release_behavior"))))
    min_duration_ms = _normalized_duration_ms(raw.get("min_duration_ms", raw.get("duration_min_ms")))
    max_duration_ms = _normalized_duration_ms(raw.get("max_duration_ms", raw.get("duration_max_ms")))
    if timing:
        normalized["timing"] = timing
    if release:
        normalized["release"] = release
    if min_duration_ms is not None:
        normalized["min_duration_ms"] = min_duration_ms
    if max_duration_ms is not None:
        normalized["max_duration_ms"] = max_duration_ms
    return normalized


@dataclass(slots=True)
class AliasMetadataRecord:
    values: dict[str, object]
    origins: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LoadedAliasInventory:
    aliases: set[str]
    repair_map: dict[str, list[str]]
    alias_metadata: dict[str, list[AliasMetadataRecord]] = field(default_factory=dict)
    bank_metadata: dict[str, object] = field(default_factory=dict)
    metadata_sources: list[str] = field(default_factory=list)
    metadata_fallback_used: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class InventoryValidationResult:
    checked: bool
    hits: list[str]
    misses: list[str]
    validated_aliases: list[str]
    repaired_aliases: list[str]
    unresolved_aliases: list[str]
    fallback_used: bool
    notes: list[str]
    candidate_aliases: list[str] = field(default_factory=list)
    selected_alias: str | None = None
    selection_reason: str | None = None
    ranking_features: list[str] = field(default_factory=list)
    phonetic_similarity_score: float | None = None
    bank_metadata_score: float | None = None
    selected_alias_metadata: dict[str, object] | None = None
    metadata_source: list[str] = field(default_factory=list)
    normalized_bank_metadata: dict[str, object] | None = None
    alias_metadata_origin: list[str] = field(default_factory=list)
    metadata_fallback_used: bool = False
    repair_rank_applied: bool = False
    note_context_applied: bool = False
    note_context_source: str | None = None
    note_range_match: bool | None = None
    timing_match: bool | None = None
    release_match: bool | None = None
    note_context_fallback_used: bool = False


@dataclass(slots=True)
class AliasCandidate:
    alias: str
    alias_type: str
    role: str
    is_fallback: bool = False
    target_symbols: list[str] = field(default_factory=list)
    target_features: list[FeatureMap] = field(default_factory=list)


@dataclass(slots=True)
class AliasSelectionContext:
    current_role: str
    bank_profile: str | None = None
    alias_strategy: str | None = None
    preferred_alias_types: list[str] = field(default_factory=list)
    preferred_subbanks: list[str] = field(default_factory=list)
    preferred_colors: list[str] = field(default_factory=list)
    preferred_tags: list[str] = field(default_factory=list)
    preferred_styles: list[str] = field(default_factory=list)
    lyric_token: str | None = None
    grouping_fallback_used: bool = False
    note_midi: int | None = None
    segment_duration_ms: float | None = None
    timing_bucket: str | None = None
    release_context: str | None = None
    note_context_applied: bool = False
    note_context_source: str | None = None
    note_context_fallback_used: bool = False
    normalized_track_id: str | None = None
    normalized_part_id: str | None = None
    normalized_note_id: str | None = None
    note_identity_fallback_used: bool = False
    note_identity_source: str | None = None
    future_aligner_aligned_note_id: str | None = None
    future_aligner_aligner_confidence: float | None = None
    future_aligner_same_note_support: bool = False
    future_aligner_part_continuity_support: bool = False
    future_aligner_neighbor_support_count: int = 0
    future_aligner_evidence_summary: str | None = None
    future_aligner_provenance_applied: bool = False
    future_aligner_provenance_summary: str | None = None
    future_aligner_provenance_items: list[dict[str, object]] = field(default_factory=list)
    future_aligner_fallback_used: bool = False
    future_aligner_sequence_applied: bool = False


@dataclass(slots=True)
class RankedAliasCandidate:
    alias: str
    source_alias: str
    match_kind: str
    alias_type: str
    role: str
    score: int
    features: list[str] = field(default_factory=list)
    phonetic_similarity_score: float | None = None
    bank_metadata_score: float | None = None
    selected_alias_metadata: dict[str, object] | None = None
    alias_metadata_origin: list[str] = field(default_factory=list)
    note_range_match: bool | None = None
    timing_match: bool | None = None
    release_match: bool | None = None


@dataclass(slots=True)
class MetadataScoreBreakdown:
    score: float | None = None
    bonus: int = 0
    features: list[str] = field(default_factory=list)
    selected_metadata: dict[str, object] | None = None
    alias_metadata_origin: list[str] = field(default_factory=list)
    note_range_match: bool | None = None
    timing_match: bool | None = None
    release_match: bool | None = None


class AliasInventorySource(Protocol):
    def supports(self, inventory_path: Path) -> bool: ...

    def load(self, inventory_path: Path) -> LoadedAliasInventory: ...


class JsonAliasInventorySource:
    def supports(self, inventory_path: Path) -> bool:
        return inventory_path.is_file() and inventory_path.suffix.lower() == ".json"

    def load(self, inventory_path: Path) -> LoadedAliasInventory:
        payload = load_json(inventory_path)
        aliases, embedded_metadata = self._aliases_and_metadata(payload.get("aliases", []))
        alias_metadata = self._merge_alias_metadata(
            embedded_metadata,
            self._alias_metadata(payload.get("alias_metadata", {})),
        )
        bank_metadata = self._normalize_bank_metadata(payload.get("bank_metadata", {}))
        if not aliases:
            raise ValueError(f"JSON alias inventory must contain a non-empty 'aliases' list: {inventory_path}")
        metadata_sources = []
        if alias_metadata:
            metadata_sources.append(f"inventory_json_alias_metadata:{inventory_path.name}")
        if bank_metadata:
            metadata_sources.append(f"inventory_json_bank_metadata:{inventory_path.name}")
        return LoadedAliasInventory(
            aliases=aliases,
            repair_map=self._repair_map(payload.get("repair_map", {})),
            alias_metadata=alias_metadata,
            bank_metadata=bank_metadata,
            metadata_sources=metadata_sources,
            metadata_fallback_used=False,
            notes=[f"loaded alias inventory from JSON: {inventory_path.name}"],
        )

    def _aliases_and_metadata(self, raw: object) -> tuple[set[str], dict[str, list[AliasMetadataRecord]]]:
        if not isinstance(raw, list):
            return (set(), {})
        aliases: set[str] = set()
        alias_metadata: dict[str, list[AliasMetadataRecord]] = {}
        for item in raw:
            if isinstance(item, dict):
                alias = str(item.get("alias", "")).strip()
                if not alias:
                    continue
                aliases.add(alias)
                metadata = self._normalize_alias_metadata_record(item.get("metadata", item))
                if metadata:
                    alias_metadata.setdefault(alias, []).append(
                        AliasMetadataRecord(values=metadata, origins=["inventory_json:inline_alias_metadata"])
                    )
                continue
            alias = str(item).strip()
            if alias:
                aliases.add(alias)
        return (aliases, alias_metadata)

    def _repair_map(self, raw: object) -> dict[str, list[str]]:
        if not isinstance(raw, dict):
            return {}
        repairs: dict[str, list[str]] = {}
        for key, value in raw.items():
            if not isinstance(value, list):
                continue
            candidates = [str(candidate).strip() for candidate in value if str(candidate).strip()]
            if candidates:
                repairs[str(key)] = candidates
        return repairs

    def _alias_metadata(self, raw: object) -> dict[str, list[AliasMetadataRecord]]:
        if not isinstance(raw, dict):
            return {}
        metadata: dict[str, list[AliasMetadataRecord]] = {}
        for alias, value in raw.items():
            alias_name = str(alias).strip()
            if not alias_name:
                continue
            records = value if isinstance(value, list) else [value]
            for record in records:
                normalized = self._normalize_alias_metadata_record(record)
                if normalized:
                    metadata.setdefault(alias_name, []).append(
                        AliasMetadataRecord(values=normalized, origins=["inventory_json:alias_metadata"])
                    )
        return metadata

    def _merge_alias_metadata(
        self,
        left: dict[str, list[AliasMetadataRecord]],
        right: dict[str, list[AliasMetadataRecord]],
    ) -> dict[str, list[AliasMetadataRecord]]:
        merged: dict[str, list[AliasMetadataRecord]] = {}
        for alias, records in left.items():
            merged[alias] = [
                AliasMetadataRecord(values=dict(record.values), origins=list(record.origins))
                for record in records
            ]
        for alias, records in right.items():
            merged.setdefault(alias, []).extend(
                AliasMetadataRecord(values=dict(record.values), origins=list(record.origins))
                for record in records
            )
        return merged

    def _normalize_bank_metadata(self, raw: object) -> dict[str, object]:
        if not isinstance(raw, dict):
            return {}
        normalized: dict[str, object] = {}
        singer_name = self._normalized_text(raw.get("singer_name", raw.get("name")))
        color = self._normalized_text(raw.get("color", raw.get("default_color")))
        style = self._normalized_text(raw.get("style", raw.get("default_style", raw.get("tone"))))
        tags = self._normalized_tags(raw.get("tags", raw.get("bank_tags")))
        subbanks = self._normalize_subbanks(raw.get("subbanks", {}))
        if singer_name:
            normalized["singer_name"] = singer_name
        if color:
            normalized["color"] = color
        if style:
            normalized["style"] = style
        if tags:
            normalized["tags"] = tags
        normalized.update(_normalized_behavior_metadata(raw))
        if subbanks:
            normalized["subbanks"] = subbanks
        return normalized

    def _normalize_subbanks(self, raw: object) -> dict[str, dict[str, object]]:
        entries = raw if isinstance(raw, list) else raw.items() if isinstance(raw, dict) else []
        subbanks: dict[str, dict[str, object]] = {}
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                normalized = self._normalize_alias_metadata_record(item)
                key = self._normalized_text(item.get("name", item.get("subbank", item.get("color", item.get("dir")))))
                directory = self._normalized_text(item.get("dir", item.get("directory")))
                if directory:
                    normalized["dir"] = directory
                if key and normalized:
                    subbanks[key] = normalized
            return subbanks
        if isinstance(raw, dict):
            for key, value in raw.items():
                if not isinstance(value, dict):
                    continue
                normalized = self._normalize_alias_metadata_record(value)
                directory = self._normalized_text(value.get("dir", value.get("directory")))
                if directory:
                    normalized["dir"] = directory
                alias_key = self._normalized_text(key)
                if alias_key and normalized:
                    subbanks[alias_key] = normalized
        return subbanks

    def _normalize_alias_metadata_record(self, raw: object) -> dict[str, object]:
        if not isinstance(raw, dict):
            return {}
        normalized: dict[str, object] = {}
        subbank = self._normalized_text(raw.get("subbank"))
        color = self._normalized_text(raw.get("color"))
        style = self._normalized_text(raw.get("style", raw.get("tone")))
        tags = self._normalized_tags(raw.get("tags", raw.get("alias_tags", raw.get("tag"))))
        if subbank:
            normalized["subbank"] = subbank
        if color:
            normalized["color"] = color
        if style:
            normalized["style"] = style
        if tags:
            normalized["tags"] = tags
        normalized.update(_normalized_behavior_metadata(raw))
        return normalized

    def _normalized_text(self, value: object) -> str | None:
        text = str(value).strip().lower() if value is not None else ""
        return text or None

    def _normalized_tags(self, raw: object) -> list[str]:
        if raw is None:
            return []
        values = raw if isinstance(raw, list) else [raw]
        normalized: list[str] = []
        for value in values:
            text = self._normalized_text(value)
            if text and text not in normalized:
                normalized.append(text)
        return normalized


class OtoIniAliasInventorySource:
    def supports(self, inventory_path: Path) -> bool:
        return inventory_path.is_file() and inventory_path.suffix.lower() == ".ini"

    def load(self, inventory_path: Path) -> LoadedAliasInventory:
        aliases = self._parse_oto_file(inventory_path)
        if not aliases:
            raise ValueError(f"No usable aliases found in oto inventory: {inventory_path}")
        alias_metadata, alias_sources = _load_sidecar_alias_metadata_bundle(inventory_path, inventory_path.parent)
        bank_metadata, bank_sources = _load_sidecar_bank_metadata_bundle(inventory_path, inventory_path.parent)
        return LoadedAliasInventory(
            aliases=aliases,
            repair_map=_load_sidecar_repair_map(inventory_path, inventory_path.parent),
            alias_metadata=alias_metadata,
            bank_metadata=bank_metadata,
            metadata_sources=[*alias_sources, *bank_sources],
            metadata_fallback_used=False,
            notes=[f"loaded alias inventory from oto.ini: {inventory_path.name}"],
        )

    def _parse_oto_file(self, inventory_path: Path) -> set[str]:
        aliases: set[str] = set()
        for alias, _ in self._parse_oto_entries(inventory_path):
            aliases.add(alias)
        return aliases

    def _parse_oto_entries(self, inventory_path: Path) -> list[tuple[str, Path]]:
        entries: list[tuple[str, Path]] = []
        with inventory_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or line.startswith(";") or "=" not in line:
                    continue
                wav_name, remainder = line.split("=", 1)
                alias = remainder.split(",", 1)[0].strip()
                if alias:
                    entries.append((alias, inventory_path))
                    continue
                fallback = Path(wav_name.strip()).stem
                if fallback:
                    entries.append((fallback, inventory_path))
        return entries


class OpenUtauSingerInventorySource:
    def supports(self, inventory_path: Path) -> bool:
        if inventory_path.is_dir():
            return True
        if not inventory_path.is_file():
            return False
        return inventory_path.name.lower() in {"character.yaml", "character.yml"}

    def load(self, inventory_path: Path) -> LoadedAliasInventory:
        root = inventory_path if inventory_path.is_dir() else inventory_path.parent
        oto_files = self._find_oto_files(root)
        if not oto_files:
            raise ValueError(f"No oto.ini files found for OpenUtau inventory import: {inventory_path}")

        parser = OtoIniAliasInventorySource()
        aliases: set[str] = set()
        manifest_metadata, manifest_sources = _load_singer_manifest_bank_metadata(root, inventory_path)
        derived_alias_metadata: dict[str, list[AliasMetadataRecord]] = {}
        for oto_file in oto_files:
            for alias, source_path in parser._parse_oto_entries(oto_file):
                aliases.add(alias)
                derived_record = _metadata_for_oto_alias(root, source_path, manifest_metadata, manifest_sources)
                if derived_record is not None:
                    derived_alias_metadata.setdefault(alias, []).append(derived_record)
        if not aliases:
            raise ValueError(f"OpenUtau inventory import did not yield any aliases: {inventory_path}")

        sidecar_alias_metadata, sidecar_alias_sources = _load_sidecar_alias_metadata_bundle(inventory_path, root)
        bank_sidecar_metadata, bank_sidecar_sources = _load_sidecar_bank_metadata_bundle(inventory_path, root)
        merged_bank_metadata = _merge_bank_metadata(manifest_metadata, bank_sidecar_metadata)
        merged_alias_metadata = _merge_alias_metadata_records(derived_alias_metadata, sidecar_alias_metadata)
        source_kind = "directory" if inventory_path.is_dir() else inventory_path.name
        return LoadedAliasInventory(
            aliases=aliases,
            repair_map=_load_sidecar_repair_map(inventory_path, root),
            alias_metadata=merged_alias_metadata,
            bank_metadata=merged_bank_metadata,
            metadata_sources=[*manifest_sources, *bank_sidecar_sources, *sidecar_alias_sources],
            metadata_fallback_used=bool(sidecar_alias_sources or bank_sidecar_sources),
            notes=[
                f"loaded alias inventory from OpenUtau singer source: {source_kind}",
                *(
                    ["merged real singer metadata with sidecar metadata fallback"]
                    if sidecar_alias_sources or bank_sidecar_sources
                    else ["derived alias metadata from singer manifest/config sources"]
                ),
            ],
        )

    def _find_oto_files(self, root: Path) -> list[Path]:
        return sorted(
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file() and candidate.name.lower() == "oto.ini"
        )


class PhoneticFeatureResolver:
    def __init__(self, payload: dict[str, object] | None = None) -> None:
        payload = payload or {}
        self.symbol_features = self._feature_map(payload.get("symbol_features", {}))
        self.alias_token_map = self._string_map(payload.get("alias_token_map", {}))
        self.strip_prefixes = self._string_list(payload.get("strip_prefixes", []))
        self.strip_suffixes = self._string_list(payload.get("strip_suffixes", []))

    def resolve_symbol_features(self, symbol: str) -> FeatureMap | None:
        canonical = self._canonical_symbol(symbol)
        features = self.symbol_features.get(canonical)
        if not features:
            return None
        return dict(features)

    def resolve_alias_features(self, alias: str) -> list[FeatureMap]:
        features: list[FeatureMap] = []
        for token in self.alias_tokens(alias):
            token_features = self.resolve_symbol_features(token)
            if token_features:
                features.append(token_features)
        return features

    def alias_tokens(self, alias: str) -> list[str]:
        tokens: list[str] = []
        for raw_token in str(alias).split():
            token = self._strip_affixes(raw_token)
            if not token:
                continue
            tokens.append(self._canonical_symbol(token))
        return tokens

    def _canonical_symbol(self, symbol: str) -> str:
        stripped = self._strip_affixes(symbol)
        mapped = self.alias_token_map.get(stripped, stripped)
        if mapped.startswith("UNMAPPED:"):
            return mapped.split(":", 1)[1]
        return mapped

    def _strip_affixes(self, token: str) -> str:
        stripped = token.strip()
        changed = True
        while changed:
            changed = False
            for prefix in self.strip_prefixes:
                if prefix and stripped.startswith(prefix):
                    stripped = stripped[len(prefix) :]
                    changed = True
            for suffix in self.strip_suffixes:
                if suffix and stripped.endswith(suffix):
                    stripped = stripped[: -len(suffix)]
                    changed = True
        return stripped

    def _feature_map(self, raw: object) -> dict[str, FeatureMap]:
        if not isinstance(raw, dict):
            return {}
        features: dict[str, FeatureMap] = {}
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            features[str(key)] = {str(feature_key): str(feature_value) for feature_key, feature_value in value.items()}
        return features

    def _string_map(self, raw: object) -> dict[str, str]:
        if not isinstance(raw, dict):
            return {}
        return {str(key): str(value) for key, value in raw.items()}

    def _string_list(self, raw: object) -> list[str]:
        if not isinstance(raw, list):
            return []
        return [str(item) for item in raw]


class AliasInventoryValidator:
    def __init__(
        self,
        inventory_path: Path | None,
        phonetic_features: dict[str, object] | None = None,
    ) -> None:
        self.inventory_path = inventory_path
        self._aliases: set[str] = set()
        self._repair_map: dict[str, list[str]] = {}
        self._alias_metadata: dict[str, list[AliasMetadataRecord]] = {}
        self._bank_metadata: dict[str, object] = {}
        self._metadata_sources: list[str] = []
        self._metadata_fallback_used = False
        self._notes: list[str] = []
        self._loaded = False
        self._phonetic_resolver = PhoneticFeatureResolver(phonetic_features)
        if inventory_path is not None:
            loaded = _load_alias_inventory(inventory_path)
            self._aliases = loaded.aliases
            self._repair_map = loaded.repair_map
            self._alias_metadata = loaded.alias_metadata
            self._bank_metadata = loaded.bank_metadata
            self._metadata_sources = loaded.metadata_sources
            self._metadata_fallback_used = loaded.metadata_fallback_used
            self._notes = loaded.notes
            self._loaded = True

    def validate(
        self,
        aliases: list[str],
        selection_context: AliasSelectionContext | None = None,
    ) -> InventoryValidationResult:
        candidates = [
            AliasCandidate(
                alias=alias,
                alias_type="plain",
                role=selection_context.current_role if selection_context is not None else "unknown",
            )
            for alias in aliases
        ]
        return self.validate_candidates(candidates, selection_context=selection_context)

    def validate_candidates(
        self,
        candidates: list[AliasCandidate],
        selection_context: AliasSelectionContext | None = None,
    ) -> InventoryValidationResult:
        planner_aliases = _unique_aliases(candidate.alias for candidate in candidates)
        selected_planner_alias = planner_aliases[0] if planner_aliases else None
        if not self._loaded:
            return InventoryValidationResult(
                checked=False,
                hits=[],
                misses=[],
                validated_aliases=list(planner_aliases),
                repaired_aliases=[],
                unresolved_aliases=[],
                fallback_used=False,
                notes=["no engine inventory provided; skipped alias inventory validation"],
                candidate_aliases=list(planner_aliases),
                selected_alias=selected_planner_alias,
                selection_reason="inventory ranking skipped; planner aliases kept because no engine inventory was provided",
                ranking_features=[
                    "inventory_checked=false",
                    "repair_ranking=skipped",
                    "phonetic_fallback=no_inventory",
                    "bank_metadata_fallback=no_inventory",
                    f"note_context_applied={str(selection_context.note_context_applied).lower()}"
                    if selection_context is not None
                    else "note_context_applied=false",
                ],
                phonetic_similarity_score=None,
                bank_metadata_score=None,
                selected_alias_metadata=None,
                metadata_source=[],
                normalized_bank_metadata=None,
                alias_metadata_origin=[],
                metadata_fallback_used=True,
                repair_rank_applied=False,
                note_context_applied=selection_context.note_context_applied if selection_context is not None else False,
                note_context_source=selection_context.note_context_source if selection_context is not None else None,
                note_range_match=None,
                timing_match=None,
                release_match=None,
                note_context_fallback_used=selection_context.note_context_fallback_used if selection_context is not None else True,
            )

        hits: list[str] = []
        misses: list[str] = []
        validated: list[str] = []
        repaired: list[str] = []
        unresolved: list[str] = []
        notes: list[str] = list(self._notes)
        fallback_used = False
        ranked_candidates: list[RankedAliasCandidate] = []

        for candidate in candidates:
            if candidate.alias in self._aliases:
                if candidate.alias not in hits:
                    hits.append(candidate.alias)
                if candidate.alias not in validated:
                    validated.append(candidate.alias)
                ranked_candidates.append(self._score_candidate(candidate, candidate.alias, "exact", selection_context))
                continue

            if candidate.alias not in misses:
                misses.append(candidate.alias)
            repaired_candidates = self._repairs(candidate, selection_context)
            if repaired_candidates:
                fallback_used = True
                best_repair = self._dedupe_ranked_candidates(repaired_candidates)[0]
                if best_repair.alias not in repaired:
                    repaired.append(best_repair.alias)
                if best_repair.alias not in validated:
                    validated.append(best_repair.alias)
                ranked_candidates.extend(repaired_candidates)
                notes.append(
                    "inventory repaired alias candidates: "
                    f"{candidate.alias} -> {', '.join(repaired_candidate.alias for repaired_candidate in repaired_candidates)}"
                )
                continue

            if candidate.alias not in unresolved:
                unresolved.append(candidate.alias)
            fallback_used = True
            ranked_candidates.append(self._score_candidate(candidate, candidate.alias, "unresolved", selection_context))
            notes.append(f"inventory could not resolve alias: {candidate.alias}")

        deduped_ranked = self._dedupe_ranked_candidates(ranked_candidates)
        deduped_ranked = self._apply_sequence_hint_tiebreak(deduped_ranked, selection_context)
        top_candidate = deduped_ranked[0] if deduped_ranked else None
        selection_reason = self._selection_reason(top_candidate, selection_context)
        if selection_reason is not None:
            notes.append(selection_reason)

        return InventoryValidationResult(
            checked=True,
            hits=hits,
            misses=misses,
            validated_aliases=validated,
            repaired_aliases=repaired,
            unresolved_aliases=unresolved,
            fallback_used=fallback_used,
            notes=notes or ["alias inventory validation completed"],
            candidate_aliases=[candidate.alias for candidate in deduped_ranked] or planner_aliases,
            selected_alias=top_candidate.alias if top_candidate is not None else selected_planner_alias,
            selection_reason=selection_reason,
            ranking_features=list(top_candidate.features) if top_candidate is not None else ["no_ranked_candidates"],
            phonetic_similarity_score=top_candidate.phonetic_similarity_score if top_candidate is not None else None,
            bank_metadata_score=top_candidate.bank_metadata_score if top_candidate is not None else None,
            selected_alias_metadata=dict(top_candidate.selected_alias_metadata)
            if top_candidate is not None and top_candidate.selected_alias_metadata is not None
            else None,
            metadata_source=list(self._metadata_sources),
            normalized_bank_metadata=_serializable_bank_metadata(self._bank_metadata) if self._bank_metadata else None,
            alias_metadata_origin=list(top_candidate.alias_metadata_origin) if top_candidate is not None else [],
            metadata_fallback_used=self._metadata_fallback_used,
            repair_rank_applied=any(candidate.match_kind == "repair" for candidate in deduped_ranked)
            or len(deduped_ranked) > 1,
            note_context_applied=selection_context.note_context_applied if selection_context is not None else False,
            note_context_source=selection_context.note_context_source if selection_context is not None else None,
            note_range_match=top_candidate.note_range_match if top_candidate is not None else None,
            timing_match=top_candidate.timing_match if top_candidate is not None else None,
            release_match=top_candidate.release_match if top_candidate is not None else None,
            note_context_fallback_used=selection_context.note_context_fallback_used if selection_context is not None else True,
        )

    def resolve_symbol_features(self, symbol: str) -> FeatureMap | None:
        return self._phonetic_resolver.resolve_symbol_features(symbol)

    def _repairs(
        self,
        candidate: AliasCandidate,
        selection_context: AliasSelectionContext | None,
    ) -> list[RankedAliasCandidate]:
        ranked: list[RankedAliasCandidate] = []
        for repair_alias in self._repair_map.get(candidate.alias, []):
            if repair_alias not in self._aliases:
                continue
            repair_candidate = AliasCandidate(
                alias=repair_alias,
                alias_type=self._repair_alias_type(candidate, repair_alias),
                role=candidate.role,
                is_fallback=candidate.is_fallback,
                target_symbols=list(candidate.target_symbols),
                target_features=[dict(features) for features in candidate.target_features],
            )
            ranked.append(
                self._score_candidate(
                    repair_candidate,
                    repair_alias,
                    "repair",
                    selection_context,
                    source_alias=candidate.alias,
                )
            )
        return ranked

    def _repair_alias_type(self, source_candidate: AliasCandidate, repair_alias: str) -> str:
        if source_candidate.alias_type == "pause":
            return "pause"
        if " " in repair_alias:
            return source_candidate.alias_type if source_candidate.alias_type in {"cv", "vc"} else "fallback"
        if source_candidate.alias_type == "fallback":
            return "fallback"
        return "plain"

    def _score_candidate(
        self,
        candidate: AliasCandidate,
        resolved_alias: str,
        match_kind: str,
        selection_context: AliasSelectionContext | None,
        source_alias: str | None = None,
    ) -> RankedAliasCandidate:
        role = selection_context.current_role if selection_context is not None else candidate.role
        score = self._match_kind_score(match_kind)
        features = [f"match_kind={match_kind}", f"role={role}", f"alias_type={candidate.alias_type}"]
        if selection_context is not None:
            features.append(f"note_context_applied={str(selection_context.note_context_applied).lower()}")
            if selection_context.note_context_source:
                features.append(f"note_context_source={selection_context.note_context_source}")
            features.append(f"note_context_fallback_used={str(selection_context.note_context_fallback_used).lower()}")

        role_bonus = self._role_bonus(role, candidate.alias_type)
        if role_bonus:
            score += role_bonus
            features.append(f"role_match_bonus={role_bonus}")

        preference_bonus, preference_rank = self._profile_preference_bonus(candidate.alias_type, selection_context)
        if preference_bonus:
            score += preference_bonus
        features.append(f"profile_preference_rank={preference_rank}")

        if candidate.is_fallback or candidate.alias_type == "fallback":
            score -= 40
            features.append("fallback_penalty=40")

        phonetic_similarity, phonetic_note = self._phonetic_similarity(candidate, resolved_alias, role)
        if phonetic_similarity is not None:
            phonetic_bonus = int(round(phonetic_similarity * 180))
            score += phonetic_bonus
            features.append(f"phonetic_similarity={phonetic_similarity:.3f}")
            features.append(f"phonetic_bonus={phonetic_bonus}")
        else:
            features.append(f"phonetic_fallback={phonetic_note}")

        metadata_breakdown = self._bank_metadata_score(
            resolved_alias,
            candidate,
            selection_context,
        )
        if metadata_breakdown.bonus:
            score += metadata_breakdown.bonus
        features.extend(metadata_breakdown.features)

        if selection_context is not None and selection_context.lyric_token:
            if not selection_context.grouping_fallback_used and candidate.alias_type != "fallback":
                score += 15
                features.append(f"lyric_hint={selection_context.lyric_token}")
            else:
                score += 5
                features.append("lyric_hint_with_grouping_fallback")
        if selection_context is not None and selection_context.grouping_fallback_used:
            if candidate.alias_type in {"plain", "fallback"}:
                score += 10
                features.append("grouping_fallback_prefers_plain")
            else:
                score -= 5
                features.append("grouping_fallback_penalty")
        if selection_context is not None and selection_context.alias_strategy:
            features.append(f"alias_strategy={selection_context.alias_strategy}")
        if selection_context is not None and selection_context.bank_profile:
            features.append(f"bank_profile={selection_context.bank_profile}")

        return RankedAliasCandidate(
            alias=resolved_alias,
            source_alias=source_alias or candidate.alias,
            match_kind=match_kind,
            alias_type=candidate.alias_type,
            role=role,
            score=score,
            features=features,
            phonetic_similarity_score=phonetic_similarity,
            bank_metadata_score=metadata_breakdown.score,
            selected_alias_metadata=metadata_breakdown.selected_metadata,
            alias_metadata_origin=metadata_breakdown.alias_metadata_origin,
            note_range_match=metadata_breakdown.note_range_match,
            timing_match=metadata_breakdown.timing_match,
            release_match=metadata_breakdown.release_match,
        )

    def _phonetic_similarity(
        self,
        candidate: AliasCandidate,
        resolved_alias: str,
        role: str,
    ) -> tuple[float | None, str]:
        target_features = [features for features in candidate.target_features if self._usable_features(features)]
        alias_features = self._phonetic_resolver.resolve_alias_features(resolved_alias)
        if not target_features:
            return (None, "missing_target_features")
        if not alias_features:
            return (None, "missing_alias_features")

        target_focus = self._role_focus_features(target_features, role)
        alias_focus = self._role_focus_features(alias_features, role)
        if target_focus is None or alias_focus is None:
            return (None, "missing_role_focus_features")

        sequence_similarity = self._sequence_similarity(target_features, alias_features)
        role_similarity = self._feature_similarity(target_focus, alias_focus)
        similarity = round((sequence_similarity * 0.6) + (role_similarity * 0.4), 3)
        return (similarity, "phonetic_features_used")

    def _sequence_similarity(self, target_features: list[FeatureMap], alias_features: list[FeatureMap]) -> float:
        max_length = max(len(target_features), len(alias_features))
        if max_length == 0:
            return 0.0
        total = 0.0
        compared = min(len(target_features), len(alias_features))
        for index in range(compared):
            total += self._feature_similarity(target_features[index], alias_features[index])
        if len(target_features) != len(alias_features):
            total -= 0.2 * abs(len(target_features) - len(alias_features))
        return max(0.0, min(1.0, total / max_length))

    def _role_focus_features(self, features: list[FeatureMap], role: str) -> FeatureMap | None:
        if not features:
            return None
        if role == "pause":
            return self._first_matching(features, lambda feature: feature.get("type") == "pause") or features[0]
        if role == "nucleus":
            return self._first_matching(features, lambda feature: feature.get("type") == "vowel") or features[0]
        if role == "coda":
            return self._last_matching(features, lambda feature: feature.get("type") == "consonant") or features[-1]
        return self._first_matching(features, lambda feature: feature.get("type") == "consonant") or features[0]

    def _feature_similarity(self, target: FeatureMap, candidate: FeatureMap) -> float:
        target_type = target.get("type")
        candidate_type = candidate.get("type")
        if not target_type or not candidate_type:
            return 0.0
        if target_type == "pause" or candidate_type == "pause":
            return 1.0 if target_type == candidate_type else 0.0
        if target_type != candidate_type:
            return 0.0

        score = 0.35
        if target_type == "vowel":
            score += self._feature_match(target, candidate, "height", exact=0.2, close=0.1, ordered=["open", "mid", "close"])
            score += self._feature_match(
                target,
                candidate,
                "backness",
                exact=0.2,
                close=0.1,
                ordered=["front", "central", "back"],
            )
            score += self._boolean_family_bonus(target, candidate, "rounded", 0.05)
            score += self._boolean_family_bonus(target, candidate, "nasalized", 0.05)
        else:
            score += self._feature_match(target, candidate, "manner", exact=0.18, close=0.08)
            score += self._feature_match(target, candidate, "place", exact=0.18, close=0.08)
            score += self._family_match(target, candidate, "manner_family", 0.08)
            score += self._family_match(target, candidate, "place_family", 0.08)
            score += self._boolean_family_bonus(target, candidate, "nasal", 0.06)
            score += self._boolean_family_bonus(target, candidate, "stop_like", 0.06)
            score += self._boolean_family_bonus(target, candidate, "glide_like", 0.06)
        return max(0.0, min(1.0, score))

    def _feature_match(
        self,
        target: FeatureMap,
        candidate: FeatureMap,
        key: str,
        exact: float,
        close: float,
        ordered: list[str] | None = None,
    ) -> float:
        left = target.get(key)
        right = candidate.get(key)
        if not left or not right:
            return 0.0
        if left == right:
            return exact
        if ordered is not None and left in ordered and right in ordered:
            if abs(ordered.index(left) - ordered.index(right)) == 1:
                return close
            return 0.0
        if self._related_value(left, right):
            return close
        return 0.0

    def _family_match(self, target: FeatureMap, candidate: FeatureMap, key: str, value: float) -> float:
        left = target.get(key)
        right = candidate.get(key)
        if left and right and left == right:
            return value
        return 0.0

    def _boolean_family_bonus(self, target: FeatureMap, candidate: FeatureMap, key: str, value: float) -> float:
        left = target.get(key)
        right = candidate.get(key)
        if left and right and left == right:
            return value
        return 0.0

    def _related_value(self, left: str, right: str) -> bool:
        related_groups = (
            {"stop", "stop_closure"},
            {"approximant", "lateral_approximant", "glide"},
            {"alveolar", "dental"},
            {"velar", "dorsal"},
            {"open_mid", "mid", "close_mid"},
        )
        for group in related_groups:
            if left in group and right in group:
                return True
        return False

    def _usable_features(self, features: FeatureMap) -> bool:
        return bool(features) and features.get("type") not in {None, "", "unknown"}

    def _first_matching(self, features: list[FeatureMap], predicate: object) -> FeatureMap | None:
        for feature in features:
            if predicate(feature):
                return feature
        return None

    def _last_matching(self, features: list[FeatureMap], predicate: object) -> FeatureMap | None:
        for feature in reversed(features):
            if predicate(feature):
                return feature
        return None

    def _match_kind_score(self, match_kind: str) -> int:
        if match_kind == "exact":
            return 1600
        if match_kind == "repair":
            return 900
        return 100

    def _bank_metadata_score(
        self,
        resolved_alias: str,
        candidate: AliasCandidate,
        selection_context: AliasSelectionContext | None,
    ) -> MetadataScoreBreakdown:
        metadata_records = self._alias_metadata.get(resolved_alias, [])
        if not metadata_records:
            return MetadataScoreBreakdown(features=["bank_metadata_fallback=no_alias_metadata"])

        best_score: float | None = None
        best_bonus = 0
        best_features = ["bank_metadata_fallback=no_matching_metadata"]
        best_metadata: dict[str, object] | None = None
        best_origin: list[str] = []
        best_note_range_match: bool | None = None
        best_timing_match: bool | None = None
        best_release_match: bool | None = None
        for metadata_record in metadata_records:
            score, features, note_range_match, timing_match, release_match = self._metadata_similarity(
                metadata_record.values,
                candidate,
                selection_context,
            )
            bonus = int(round(score * 120))
            feature_flags = [f"bank_metadata_score={score:.3f}", *features]
            if best_score is None or score > best_score:
                best_score = score
                best_bonus = bonus
                best_features = feature_flags
                best_metadata = dict(metadata_record.values)
                best_origin = list(metadata_record.origins)
                best_note_range_match = note_range_match
                best_timing_match = timing_match
                best_release_match = release_match
        return MetadataScoreBreakdown(
            score=best_score,
            bonus=best_bonus,
            features=best_features,
            selected_metadata=best_metadata,
            alias_metadata_origin=best_origin,
            note_range_match=best_note_range_match,
            timing_match=best_timing_match,
            release_match=best_release_match,
        )

    def _metadata_similarity(
        self,
        metadata: dict[str, object],
        candidate: AliasCandidate,
        selection_context: AliasSelectionContext | None,
    ) -> tuple[float, list[str], bool | None, bool | None, bool | None]:
        score = 0.0
        features: list[str] = []
        preferred_subbanks = self._normalized_preference_list(
            selection_context.preferred_subbanks if selection_context is not None else []
        )
        preferred_colors = self._normalized_preference_list(
            selection_context.preferred_colors if selection_context is not None else []
        )
        preferred_tags = self._normalized_preference_list(
            selection_context.preferred_tags if selection_context is not None else []
        )
        preferred_styles = self._normalized_preference_list(
            selection_context.preferred_styles if selection_context is not None else []
        )

        subbank = self._normalized_metadata_value(metadata.get("subbank"))
        color = self._normalized_metadata_value(metadata.get("color"))
        style = self._normalized_metadata_value(metadata.get("style"))
        tags = self._normalized_metadata_tags(metadata.get("tags"))

        if preferred_subbanks and subbank and subbank in preferred_subbanks:
            score += 0.28
            features.append(f"metadata_subbank_match={subbank}")
        if preferred_colors and color and color in preferred_colors:
            score += 0.32
            features.append(f"metadata_color_match={color}")
        if preferred_styles and style and style in preferred_styles:
            score += 0.22
            features.append(f"metadata_style_match={style}")
        if preferred_tags and tags:
            matched_tags = [tag for tag in tags if tag in preferred_tags]
            if matched_tags:
                tag_score = min(0.24, 0.12 * len(matched_tags))
                score += tag_score
                features.append(f"metadata_tag_match={','.join(matched_tags)}")

        note_score, note_features, note_range_match = self._note_range_similarity(metadata, selection_context)
        score += note_score
        features.extend(note_features)

        timing_score, timing_features, timing_match = self._timing_similarity(metadata, selection_context)
        score += timing_score
        features.extend(timing_features)

        release_score, release_features, release_match = self._release_similarity(metadata, selection_context)
        score += release_score
        features.extend(release_features)

        role_tags = {candidate.role, candidate.alias_type, f"{candidate.role}_{candidate.alias_type}"}
        matched_role_tags = [tag for tag in tags if tag in role_tags]
        if matched_role_tags:
            score += min(0.14, 0.07 * len(matched_role_tags))
            features.append(f"metadata_role_hint={','.join(matched_role_tags)}")

        if selection_context is not None and selection_context.bank_profile:
            profile_token = self._normalized_metadata_value(selection_context.bank_profile)
            if profile_token and (
                profile_token == subbank
                or profile_token == color
                or profile_token == style
                or profile_token in tags
            ):
                score += 0.08
                features.append(f"metadata_profile_hint={profile_token}")

        if not features:
            features.append("bank_metadata_fallback=no_preference_match")
        return (round(min(1.0, score), 3), features, note_range_match, timing_match, release_match)

    def _note_range_similarity(
        self,
        metadata: dict[str, object],
        selection_context: AliasSelectionContext | None,
    ) -> tuple[float, list[str], bool | None]:
        min_note_midi = normalize_note_midi(metadata.get("min_note_midi", metadata.get("min_note")))
        max_note_midi = normalize_note_midi(metadata.get("max_note_midi", metadata.get("max_note")))
        if min_note_midi is None and max_note_midi is None:
            return (0.0, [], None)
        if selection_context is None or selection_context.note_midi is None:
            return (0.0, ["note_range_fallback=no_note_context"], None)

        note_midi = selection_context.note_midi
        lower = min_note_midi if min_note_midi is not None else note_midi
        upper = max_note_midi if max_note_midi is not None else note_midi
        if lower <= note_midi <= upper:
            span = max(1, upper - lower)
            midpoint = lower + (span / 2.0)
            half_span = max(1.0, span / 2.0)
            centeredness = max(0.0, 1.0 - (abs(note_midi - midpoint) / half_span))
            score = 0.22 + (0.08 * centeredness)
            return (round(score, 3), [f"note_range_match={lower}-{upper}", f"note_midi={note_midi}"], True)
        return (0.0, [f"note_range_miss={lower}-{upper}", f"note_midi={note_midi}"], False)

    def _timing_similarity(
        self,
        metadata: dict[str, object],
        selection_context: AliasSelectionContext | None,
    ) -> tuple[float, list[str], bool | None]:
        metadata_timing = normalize_timing_value(metadata.get("timing"))
        min_duration_ms = _normalized_duration_ms(metadata.get("min_duration_ms"))
        max_duration_ms = _normalized_duration_ms(metadata.get("max_duration_ms"))
        has_timing_metadata = metadata_timing is not None or min_duration_ms is not None or max_duration_ms is not None
        if not has_timing_metadata:
            return (0.0, [], None)
        if selection_context is None or selection_context.segment_duration_ms is None:
            return (0.0, ["timing_fallback=no_duration_context"], None)

        score = 0.0
        features: list[str] = []
        duration_ms = selection_context.segment_duration_ms
        matched = False
        if min_duration_ms is not None or max_duration_ms is not None:
            lower = min_duration_ms if min_duration_ms is not None else duration_ms
            upper = max_duration_ms if max_duration_ms is not None else duration_ms
            if lower <= duration_ms <= upper:
                score += 0.18
                features.append(f"duration_range_match={lower:.0f}-{upper:.0f}")
                matched = True
            else:
                features.append(f"duration_range_miss={lower:.0f}-{upper:.0f}")
        if metadata_timing and selection_context.timing_bucket and metadata_timing == selection_context.timing_bucket:
            score += 0.16
            features.append(f"metadata_timing_match={metadata_timing}")
            matched = True
        elif metadata_timing:
            features.append(f"metadata_timing_miss={metadata_timing}")
        return (round(score, 3), features, matched)

    def _release_similarity(
        self,
        metadata: dict[str, object],
        selection_context: AliasSelectionContext | None,
    ) -> tuple[float, list[str], bool | None]:
        metadata_release = normalize_release_value(metadata.get("release"))
        if metadata_release is None:
            return (0.0, [], None)
        if selection_context is None or selection_context.release_context is None:
            return (0.0, ["release_fallback=no_release_context"], None)
        if metadata_release == selection_context.release_context:
            return (0.18, [f"metadata_release_match={metadata_release}"], True)
        return (0.0, [f"metadata_release_miss={metadata_release}"], False)

    def _normalized_preference_list(self, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            text = self._normalized_metadata_value(value)
            if text and text not in normalized:
                normalized.append(text)
        return normalized

    def _normalized_metadata_value(self, value: object) -> str | None:
        text = str(value).strip().lower() if value is not None else ""
        return text or None

    def _normalized_metadata_tags(self, raw: object) -> list[str]:
        if not isinstance(raw, list):
            return []
        tags: list[str] = []
        for item in raw:
            tag = self._normalized_metadata_value(item)
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    def _role_bonus(self, role: str, alias_type: str) -> int:
        role_bonus_map = {
            ("pause", "pause"): 120,
            ("onset", "cv"): 100,
            ("onset", "plain"): 40,
            ("nucleus", "vc"): 100,
            ("nucleus", "cv"): 70,
            ("nucleus", "plain"): 40,
            ("coda", "vc"): 100,
            ("coda", "plain"): 40,
        }
        return role_bonus_map.get((role, alias_type), 0)

    def _profile_preference_bonus(
        self,
        alias_type: str,
        selection_context: AliasSelectionContext | None,
    ) -> tuple[int, int]:
        if selection_context is None or not selection_context.preferred_alias_types:
            return (0, -1)
        try:
            rank = selection_context.preferred_alias_types.index(alias_type)
        except ValueError:
            return (0, len(selection_context.preferred_alias_types))
        return (120 - (rank * 20), rank)

    def _apply_sequence_hint_tiebreak(
        self,
        ranked_candidates: list[RankedAliasCandidate],
        selection_context: AliasSelectionContext | None,
    ) -> list[RankedAliasCandidate]:
        if len(ranked_candidates) < 2 or selection_context is None:
            return ranked_candidates

        hint_state = self._sequence_hint_state(selection_context)
        top_candidate = ranked_candidates[0]
        if hint_state != "strong":
            if hint_state != "missing":
                top_candidate.features.append(f"sequence_hint_tiebreak=skipped_{hint_state}")
                if selection_context.future_aligner_evidence_summary:
                    top_candidate.features.append(
                        f"sequence_hint_evidence={selection_context.future_aligner_evidence_summary}"
                    )
            return ranked_candidates

        preferred_alias_types = self._sequence_hint_preferred_alias_types(selection_context.current_role)
        if not preferred_alias_types:
            return ranked_candidates
        if top_candidate.alias_type in preferred_alias_types:
            top_candidate.features.append("sequence_hint_tiebreak=stabilized_existing_alias")
            if selection_context.future_aligner_aligned_note_id:
                top_candidate.features.append(
                    f"sequence_hint_aligned_note_id={selection_context.future_aligner_aligned_note_id}"
                )
            return ranked_candidates

        margin = 80
        near_tie_candidates = [
            candidate
            for candidate in ranked_candidates
            if (top_candidate.score - candidate.score) <= margin
        ]
        promoted_candidate = next(
            (
                candidate
                for candidate in near_tie_candidates
                if candidate is not top_candidate and candidate.alias_type in preferred_alias_types
            ),
            None,
        )
        if promoted_candidate is None:
            return ranked_candidates

        promoted_candidate.features.append(
            f"sequence_hint_tiebreak=promoted_alias_type:{promoted_candidate.alias_type}"
        )
        promoted_candidate.features.append("sequence_hint_decision_point=inventory.near_tie_alias_type")
        promoted_candidate.features.append(f"sequence_hint_promoted_over={top_candidate.alias}")
        if selection_context.future_aligner_aligned_note_id:
            promoted_candidate.features.append(
                f"sequence_hint_aligned_note_id={selection_context.future_aligner_aligned_note_id}"
            )
        if selection_context.future_aligner_aligner_confidence is not None:
            promoted_candidate.features.append(
                f"sequence_hint_confidence={selection_context.future_aligner_aligner_confidence:.3f}"
            )
        promoted_candidate.features.append(
            f"sequence_hint_neighbor_support={selection_context.future_aligner_neighbor_support_count}"
        )
        if selection_context.future_aligner_evidence_summary:
            promoted_candidate.features.append(
                f"sequence_hint_evidence={selection_context.future_aligner_evidence_summary}"
            )
        return [promoted_candidate, *[candidate for candidate in ranked_candidates if candidate is not promoted_candidate]]

    def _sequence_hint_state(self, selection_context: AliasSelectionContext) -> str:
        if not (
            selection_context.future_aligner_aligned_note_id
            or selection_context.future_aligner_sequence_applied
            or selection_context.future_aligner_fallback_used
            or selection_context.future_aligner_neighbor_support_count
        ):
            return "missing"
        if selection_context.future_aligner_fallback_used or selection_context.future_aligner_aligned_note_id is None:
            return "fallback"
        if not selection_context.future_aligner_sequence_applied:
            return "per_segment_only"
        if selection_context.future_aligner_aligner_confidence is None or selection_context.future_aligner_aligner_confidence < 0.84:
            return "low_confidence"
        if not selection_context.future_aligner_same_note_support:
            return "no_same_note_support"
        if not selection_context.future_aligner_part_continuity_support:
            return "no_part_continuity"
        if selection_context.future_aligner_neighbor_support_count <= 0:
            return "no_neighbor_support"
        return "strong"

    def _sequence_hint_preferred_alias_types(self, role: str) -> list[str]:
        if role == "onset":
            return ["cv"]
        if role in {"nucleus", "coda"}:
            return ["vc", "cv"]
        return []

    def _dedupe_ranked_candidates(self, ranked_candidates: list[RankedAliasCandidate]) -> list[RankedAliasCandidate]:
        best_by_alias: dict[str, RankedAliasCandidate] = {}
        for candidate in ranked_candidates:
            existing = best_by_alias.get(candidate.alias)
            if existing is None or candidate.score > existing.score:
                best_by_alias[candidate.alias] = candidate
                continue
            if candidate.score == existing.score and _match_kind_order(candidate.match_kind) < _match_kind_order(existing.match_kind):
                best_by_alias[candidate.alias] = candidate
        return sorted(
            best_by_alias.values(),
            key=lambda candidate: (
                -candidate.score,
                _match_kind_order(candidate.match_kind),
                candidate.alias,
            ),
        )

    def _selection_reason(
        self,
        candidate: RankedAliasCandidate | None,
        selection_context: AliasSelectionContext | None,
    ) -> str | None:
        if candidate is None:
            return "inventory ranking found no selectable alias candidates"
        profile = selection_context.bank_profile if selection_context is not None else "<none>"
        phonetic_note = (
            f" with phonetic similarity {candidate.phonetic_similarity_score:.3f}"
            if candidate.phonetic_similarity_score is not None
            else " using heuristic-only fallback"
        )
        metadata_note = (
            f" and bank metadata score {candidate.bank_metadata_score:.3f}"
            if candidate.bank_metadata_score is not None
            else ""
        )
        sequence_hint_note = ""
        if any(feature.startswith("sequence_hint_decision_point=inventory.near_tie_alias_type") for feature in candidate.features):
            sequence_hint_note = " after future_aligner_sequence_hint near-tie tie-break"
        if candidate.match_kind == "exact":
            return (
                f"selected exact inventory alias '{candidate.alias}' "
                f"for role {candidate.role} under profile {profile}{phonetic_note}{metadata_note}{sequence_hint_note}"
            )
        if candidate.match_kind == "repair":
            return (
                f"selected repaired alias '{candidate.alias}' from '{candidate.source_alias}' "
                f"for role {candidate.role} under profile {profile}{phonetic_note}{metadata_note}{sequence_hint_note}"
            )
        return (
            f"no exact or repaired inventory alias resolved; kept planner fallback '{candidate.alias}' "
            f"for role {candidate.role} under profile {profile}{phonetic_note}{metadata_note}{sequence_hint_note}"
        )


def _load_alias_inventory(inventory_path: Path) -> LoadedAliasInventory:
    sources: tuple[AliasInventorySource, ...] = (
        JsonAliasInventorySource(),
        OtoIniAliasInventorySource(),
        OpenUtauSingerInventorySource(),
    )
    for source in sources:
        if source.supports(inventory_path):
            return source.load(inventory_path)
    raise ValueError(f"Unsupported engine inventory source: {inventory_path}")


def _load_sidecar_repair_map(inventory_path: Path, root: Path) -> dict[str, list[str]]:
    for candidate in _repair_map_candidates(inventory_path, root):
        if not candidate.exists():
            continue
        payload = load_json(candidate)
        raw_repairs = payload.get("repair_map", payload)
        if isinstance(raw_repairs, dict):
            repairs: dict[str, list[str]] = {}
            for key, value in raw_repairs.items():
                if not isinstance(value, list):
                    continue
                candidates = [str(item).strip() for item in value if str(item).strip()]
                if candidates:
                    repairs[str(key)] = candidates
            if repairs:
                return repairs
    return {}


def _load_sidecar_alias_metadata_bundle(
    inventory_path: Path,
    root: Path,
) -> tuple[dict[str, list[AliasMetadataRecord]], list[str]]:
    parser = JsonAliasInventorySource()
    metadata: dict[str, list[AliasMetadataRecord]] = {}
    sources: list[str] = []
    for candidate in _alias_metadata_candidates(inventory_path, root):
        if not candidate.exists():
            continue
        payload = load_json(candidate)
        raw_metadata = payload.get("alias_metadata", payload)
        parsed = parser._alias_metadata(raw_metadata)
        if not parsed:
            continue
        sources.append(f"sidecar_alias_metadata:{candidate.name}")
        for alias, records in parsed.items():
            metadata.setdefault(alias, []).extend(
                AliasMetadataRecord(
                    values=dict(record.values),
                    origins=[f"sidecar_alias_metadata:{candidate.name}"],
                )
                for record in records
            )
    return (metadata, _dedupe_strings(sources))


def _load_sidecar_bank_metadata_bundle(
    inventory_path: Path,
    root: Path,
) -> tuple[dict[str, object], list[str]]:
    parser = JsonAliasInventorySource()
    merged: dict[str, object] = {}
    sources: list[str] = []
    for candidate in _bank_metadata_candidates(inventory_path, root):
        if not candidate.exists():
            continue
        payload = load_json(candidate)
        raw_bank_metadata = payload.get("bank_metadata", payload)
        parsed = parser._normalize_bank_metadata(raw_bank_metadata)
        if not parsed:
            continue
        merged = _merge_bank_metadata(merged, parsed)
        sources.append(f"sidecar_bank_metadata:{candidate.name}")
    return (merged, _dedupe_strings(sources))


def _repair_map_candidates(inventory_path: Path, root: Path) -> list[Path]:
    candidates: list[Path] = []
    if inventory_path.is_file():
        candidates.append(inventory_path.with_name(f"{inventory_path.stem}.repair.json"))
        candidates.append(inventory_path.with_name("repair_map.json"))
        candidates.append(inventory_path.with_name("alias_repair_map.json"))
    candidates.append(root / "repair_map.json")
    candidates.append(root / "alias_repair_map.json")
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _alias_metadata_candidates(inventory_path: Path, root: Path) -> list[Path]:
    candidates: list[Path] = []
    if inventory_path.is_file():
        candidates.append(inventory_path.with_name(f"{inventory_path.stem}.metadata.json"))
        candidates.append(inventory_path.with_name("alias_metadata.json"))
    candidates.append(root / "alias_metadata.json")
    return _unique_paths(candidates)


def _bank_metadata_candidates(inventory_path: Path, root: Path) -> list[Path]:
    candidates: list[Path] = []
    if inventory_path.is_file():
        candidates.append(inventory_path.with_name("bank_metadata.json"))
    candidates.append(root / "bank_metadata.json")
    return _unique_paths(candidates)


def _load_singer_manifest_bank_metadata(
    root: Path,
    inventory_path: Path,
) -> tuple[dict[str, object], list[str]]:
    parser = JsonAliasInventorySource()
    merged: dict[str, object] = {}
    sources: list[str] = []
    manifest_path = _character_manifest_path(root, inventory_path)
    if manifest_path is not None and manifest_path.exists():
        manifest_metadata = _parse_character_manifest(manifest_path)
        if manifest_metadata:
            merged = _merge_bank_metadata(merged, manifest_metadata)
            sources.append(f"singer_manifest:{manifest_path.name}")
    for candidate in _singer_config_candidates(root):
        if not candidate.exists():
            continue
        payload = load_json(candidate)
        raw_bank_metadata = payload.get("bank_metadata", payload)
        parsed = parser._normalize_bank_metadata(raw_bank_metadata)
        if not parsed:
            continue
        merged = _merge_bank_metadata(merged, parsed)
        sources.append(f"singer_config:{candidate.name}")
    return (_serializable_bank_metadata(merged), _dedupe_strings(sources))


def _character_manifest_path(root: Path, inventory_path: Path) -> Path | None:
    if inventory_path.is_file() and inventory_path.name.lower() in {"character.yaml", "character.yml"}:
        return inventory_path
    for candidate_name in ("character.yaml", "character.yml"):
        candidate = root / candidate_name
        if candidate.exists():
            return candidate
    return None


def _singer_config_candidates(root: Path) -> list[Path]:
    return [root / "singer_metadata.json", root / "subbanks.json"]


def _parse_character_manifest(manifest_path: Path) -> dict[str, object]:
    metadata: dict[str, object] = {}
    subbanks: dict[str, dict[str, object]] = {}
    in_subbanks = False
    current_subbank: dict[str, object] | None = None
    with manifest_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(raw_line) - len(raw_line.lstrip(" "))
            stripped = line.strip()
            if indent == 0:
                if current_subbank is not None:
                    _register_subbank(subbanks, current_subbank)
                    current_subbank = None
                if stripped.startswith("subbanks:"):
                    in_subbanks = True
                    continue
                in_subbanks = False
                key, value = _yaml_key_value(stripped)
                if key is None:
                    continue
                _assign_manifest_value(metadata, key, value)
                continue
            if not in_subbanks:
                continue
            if stripped.startswith("- "):
                if current_subbank is not None:
                    _register_subbank(subbanks, current_subbank)
                current_subbank = {}
                key, value = _yaml_key_value(stripped[2:])
                if key is not None:
                    _assign_manifest_value(current_subbank, key, value)
                continue
            if current_subbank is None:
                continue
            key, value = _yaml_key_value(stripped)
            if key is not None:
                _assign_manifest_value(current_subbank, key, value)
    if current_subbank is not None:
        _register_subbank(subbanks, current_subbank)

    normalized: dict[str, object] = {}
    singer_name = _normalized_metadata_value(metadata.get("name", metadata.get("singer_name")))
    color = _normalized_metadata_value(metadata.get("color", metadata.get("default_color")))
    style = _normalized_metadata_value(metadata.get("style", metadata.get("default_style", metadata.get("tone"))))
    tags = _normalized_metadata_tags(metadata.get("tags"))
    if singer_name:
        normalized["singer_name"] = singer_name
    if color:
        normalized["color"] = color
    if style:
        normalized["style"] = style
    if tags:
        normalized["tags"] = tags
    normalized.update(_normalized_behavior_metadata(metadata))
    if subbanks:
        normalized["subbanks"] = subbanks
    return normalized


def _assign_manifest_value(target: dict[str, object], key: str, value: object) -> None:
    normalized_key = str(key).strip().lower()
    if normalized_key in {
        "name",
        "singer_name",
        "color",
        "default_color",
        "style",
        "default_style",
        "tone",
        "subbank",
        "dir",
        "directory",
        "timing",
        "timing_group",
        "timing_class",
        "release",
        "release_type",
        "release_behavior",
        "min_note",
        "max_note",
    }:
        normalized = _normalized_metadata_value(value)
        if normalized:
            target[normalized_key] = normalized
        return
    if normalized_key == "note_range":
        if value is not None:
            target["note_range"] = value
        return
    if normalized_key in {"min_note_midi", "min_midi", "max_note_midi", "max_midi"}:
        midi_value = normalize_note_midi(value)
        if midi_value is not None:
            canonical_key = "min_note_midi" if normalized_key in {"min_note_midi", "min_midi"} else "max_note_midi"
            target[canonical_key] = midi_value
        return
    if normalized_key in {"min_duration_ms", "max_duration_ms", "duration_min_ms", "duration_max_ms"}:
        duration_ms = _normalized_duration_ms(value)
        if duration_ms is not None:
            target[normalized_key] = duration_ms
        return
    if normalized_key in {"tags", "bank_tags"}:
        tags = _yaml_list(value)
        if tags:
            target["tags"] = tags


def _register_subbank(subbanks: dict[str, dict[str, object]], payload: dict[str, object]) -> None:
    key = _normalized_metadata_value(
        payload.get("subbank", payload.get("name", payload.get("color", payload.get("dir", payload.get("directory")))))
    )
    if not key:
        return
    record: dict[str, object] = {}
    if key:
        record["subbank"] = key
    color = _normalized_metadata_value(payload.get("color"))
    style = _normalized_metadata_value(payload.get("style", payload.get("tone")))
    directory = _normalized_metadata_value(payload.get("dir", payload.get("directory")))
    tags = _normalized_metadata_tags(payload.get("tags"))
    if color:
        record["color"] = color
    if style:
        record["style"] = style
    if directory:
        record["dir"] = directory
    if tags:
        record["tags"] = tags
    record.update(_normalized_behavior_metadata(payload))
    if record:
        subbanks[key] = record


def _yaml_key_value(raw: str) -> tuple[str | None, object]:
    if ":" not in raw:
        return (None, None)
    key, value = raw.split(":", 1)
    parsed_value = value.strip()
    if parsed_value.startswith("[") and parsed_value.endswith("]"):
        return (key.strip(), _yaml_list(parsed_value))
    return (key.strip(), parsed_value or None)


def _yaml_list(raw: object) -> list[str]:
    if isinstance(raw, list):
        return _normalized_metadata_tags(raw)
    if raw is None:
        return []
    text = str(raw).strip()
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return _normalized_metadata_tags([item.strip() for item in inner.split(",")])
    return _normalized_metadata_tags([text])


def _metadata_for_oto_alias(
    root: Path,
    oto_path: Path,
    bank_metadata: dict[str, object],
    metadata_sources: list[str],
) -> AliasMetadataRecord | None:
    alias_values = _alias_metadata_from_bank_context(root, oto_path, bank_metadata)
    if not alias_values:
        return None
    relative_oto = oto_path.relative_to(root).as_posix()
    origins = [*metadata_sources, f"oto:{relative_oto}"]
    return AliasMetadataRecord(values=alias_values, origins=origins)


def _alias_metadata_from_bank_context(root: Path, oto_path: Path, bank_metadata: dict[str, object]) -> dict[str, object]:
    base = _bank_defaults_as_alias_metadata(bank_metadata)
    relative_parent = oto_path.parent.relative_to(root).as_posix() if oto_path.parent != root else "."
    subbank_record = _matching_subbank_metadata(relative_parent, oto_path.parent.name, bank_metadata)
    if subbank_record:
        base = _merge_metadata_values(base, subbank_record)
    return base


def _bank_defaults_as_alias_metadata(bank_metadata: dict[str, object]) -> dict[str, object]:
    defaults: dict[str, object] = {}
    for key in ("subbank", "color", "style", "timing", "release", "min_note", "max_note"):
        value = _normalized_metadata_value(bank_metadata.get(key))
        if value:
            defaults[key] = value
    tags = _normalized_metadata_tags(bank_metadata.get("tags"))
    if tags:
        defaults["tags"] = tags
    for key in ("min_note_midi", "max_note_midi", "min_duration_ms", "max_duration_ms"):
        value = bank_metadata.get(key)
        if value is not None:
            defaults[key] = value
    return defaults


def _matching_subbank_metadata(
    relative_parent: str,
    parent_name: str,
    bank_metadata: dict[str, object],
) -> dict[str, object]:
    raw_subbanks = bank_metadata.get("subbanks")
    if not isinstance(raw_subbanks, dict):
        return {}
    normalized_parent = _normalized_metadata_value(parent_name)
    normalized_relative = _normalized_metadata_value(relative_parent)
    for key, value in raw_subbanks.items():
        if not isinstance(value, dict):
            continue
        directory = _normalized_metadata_value(value.get("dir"))
        if directory and normalized_relative and directory == normalized_relative:
            return {str(sub_key): sub_value for sub_key, sub_value in value.items() if sub_key != "dir"}
        if directory and normalized_parent and directory == normalized_parent:
            return {str(sub_key): sub_value for sub_key, sub_value in value.items() if sub_key != "dir"}
        if normalized_parent and _normalized_metadata_value(key) == normalized_parent:
            return {str(sub_key): sub_value for sub_key, sub_value in value.items() if sub_key != "dir"}
    return {}


def _merge_alias_metadata_records(
    base: dict[str, list[AliasMetadataRecord]],
    override: dict[str, list[AliasMetadataRecord]],
) -> dict[str, list[AliasMetadataRecord]]:
    merged: dict[str, list[AliasMetadataRecord]] = {
        alias: [AliasMetadataRecord(values=dict(record.values), origins=list(record.origins)) for record in records]
        for alias, records in base.items()
    }
    for alias, override_records in override.items():
        if alias not in merged or not merged[alias]:
            merged.setdefault(alias, []).extend(
                AliasMetadataRecord(values=dict(record.values), origins=list(record.origins))
                for record in override_records
            )
            continue
        combined: list[AliasMetadataRecord] = []
        for base_record in merged[alias]:
            for override_record in override_records:
                combined.append(
                    AliasMetadataRecord(
                        values=_merge_metadata_values(base_record.values, override_record.values),
                        origins=_dedupe_strings([*base_record.origins, *override_record.origins]),
                    )
                )
        merged[alias] = combined
    return merged


def _merge_metadata_values(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key in ("subbank", "color", "style", "timing", "release", "min_note", "max_note"):
        value = _normalized_metadata_value(override.get(key, merged.get(key)))
        if value:
            merged[key] = value
    for key, normalizer in (
        ("min_note_midi", normalize_note_midi),
        ("max_note_midi", normalize_note_midi),
        ("min_duration_ms", _normalized_duration_ms),
        ("max_duration_ms", _normalized_duration_ms),
    ):
        value = normalizer(override.get(key, merged.get(key)))
        if value is not None:
            merged[key] = value
    tags = _dedupe_strings([*_normalized_metadata_tags(base.get("tags")), *_normalized_metadata_tags(override.get("tags"))])
    if tags:
        merged["tags"] = tags
    return merged


def _merge_bank_metadata(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key in ("singer_name", "subbank", "color", "style", "timing", "release", "min_note", "max_note"):
        value = _normalized_metadata_value(override.get(key, merged.get(key)))
        if value:
            merged[key] = value
    for key, normalizer in (
        ("min_note_midi", normalize_note_midi),
        ("max_note_midi", normalize_note_midi),
        ("min_duration_ms", _normalized_duration_ms),
        ("max_duration_ms", _normalized_duration_ms),
    ):
        value = normalizer(override.get(key, merged.get(key)))
        if value is not None:
            merged[key] = value
    tags = _dedupe_strings([*_normalized_metadata_tags(base.get("tags")), *_normalized_metadata_tags(override.get("tags"))])
    if tags:
        merged["tags"] = tags
    base_subbanks = base.get("subbanks", {})
    override_subbanks = override.get("subbanks", {})
    if isinstance(base_subbanks, dict) or isinstance(override_subbanks, dict):
        merged_subbanks: dict[str, dict[str, object]] = {}
        if isinstance(base_subbanks, dict):
            for key, value in base_subbanks.items():
                if isinstance(value, dict):
                    merged_subbanks[str(key)] = {str(sub_key): sub_value for sub_key, sub_value in value.items()}
        if isinstance(override_subbanks, dict):
            for key, value in override_subbanks.items():
                if not isinstance(value, dict):
                    continue
                existing = merged_subbanks.get(str(key), {})
                merged_subbanks[str(key)] = _merge_metadata_values(existing, value)
                directory = _normalized_metadata_value(value.get("dir", existing.get("dir")))
                if directory:
                    merged_subbanks[str(key)]["dir"] = directory
        if merged_subbanks:
            merged["subbanks"] = merged_subbanks
    return merged


def _serializable_bank_metadata(bank_metadata: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in bank_metadata.items():
        if isinstance(value, dict):
            payload[str(key)] = _serializable_bank_metadata(value)
            continue
        if isinstance(value, list):
            payload[str(key)] = [str(item) for item in value]
            continue
        payload[str(key)] = value
    return payload


def _normalized_metadata_value(value: object) -> str | None:
    text = str(value).strip().lower() if value is not None else ""
    return text or None


def _normalized_metadata_tags(raw: object) -> list[str]:
    if raw is None:
        return []
    values = raw if isinstance(raw, list) else [raw]
    tags: list[str] = []
    for value in values:
        tag = _normalized_metadata_value(value)
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in paths:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _match_kind_order(match_kind: str) -> int:
    order = {"exact": 0, "repair": 1, "unresolved": 2}
    return order.get(match_kind, 3)


def _unique_aliases(aliases: object) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for alias in aliases:
        text = str(alias)
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique
