"""Deterministic identity-grouped dataset splitting.

This is a functional reconstruction for the recovery workspace.  The module
uses only the Python standard library and deliberately keeps split assignment
separate from any recognizer or export layer.  Singer, song, and phrase
identities are treated as leakage boundaries: records sharing any one of those
identities are assigned to the same split through a connected component.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence
import unicodedata


SCHEMA_VERSION = "deterministic-split/v1"
ALGORITHM_VERSION = "connected-components-sha256/v3"
SPLIT_NAMES = ("train", "validation", "test")
IDENTITY_FIELDS = ("singer_id", "song_id", "phrase_id")
CollisionPolicy = Literal["error", "merge"]


class SplitError(ValueError):
    """Base class for invalid split input or configuration."""


class MissingIdentityError(SplitError):
    """Raised when an identity field is absent or blank."""


class IdentityCollisionError(SplitError):
    """Raised when normalization would silently merge raw identity values."""


class DuplicateExampleError(SplitError):
    """Raised when more than one input row has the same example identity."""


class InvalidRatiosError(SplitError):
    """Raised when split ratios are malformed or do not sum to one."""


@dataclass(frozen=True, slots=True)
class SplitRecord:
    """Minimal record needed for leakage-safe splitting.

    ``metadata`` is intentionally opaque.  Phone/IPA/articulatory truth and
    label lineage remain payload data and are not rewritten by this module.
    """

    example_id: str
    singer_id: str
    song_id: str
    phrase_id: str
    content_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SplitRecord":
        """Create a record from a JSON-like mapping.

        The adapter intentionally accepts only the stable field names.  A
        caller can map corpus-specific names before calling this function.
        """

        missing = [
            name
            for name in ("example_id", *IDENTITY_FIELDS)
            if name not in value or value[name] is None
        ]
        if missing:
            raise MissingIdentityError(
                "record is missing required field(s): " + ", ".join(missing)
            )
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise SplitError("metadata must be a mapping when provided")
        return cls(
            example_id=_text(value["example_id"]),
            singer_id=_text(value["singer_id"]),
            song_id=_text(value["song_id"]),
            phrase_id=_text(value["phrase_id"]),
            content_hash=(
                None if value.get("content_hash") is None else _text(value["content_hash"])
            ),
            metadata=dict(metadata),
        )


@dataclass(frozen=True, slots=True)
class GroupSummary:
    """Public, serializable description of one indivisible identity group."""

    group_id: str
    split: str
    record_ids: tuple[str, ...]
    identities: tuple[tuple[str, str], ...]

    @property
    def record_count(self) -> int:
        return len(self.record_ids)

    def to_dict(self) -> dict[str, Any]:
        identity_map: dict[str, list[str]] = {field_name: [] for field_name in IDENTITY_FIELDS}
        for field_name, value in self.identities:
            identity_map.setdefault(field_name, []).append(value)
        return {
            "group_id": self.group_id,
            "split": self.split,
            "record_count": self.record_count,
            "record_ids": list(self.record_ids),
            "identities": identity_map,
        }


@dataclass(frozen=True, slots=True)
class SplitResult:
    """Result of :func:`split_records` with a stable manifest."""

    assignments: Mapping[str, str]
    groups: tuple[GroupSummary, ...]
    manifest: Mapping[str, Any]

    def ids_for(self, split: str) -> tuple[str, ...]:
        """Return example IDs in canonical order for ``split``."""

        if split not in SPLIT_NAMES:
            raise SplitError(f"unknown split {split!r}")
        return tuple(
            example_id
            for example_id in sorted(self.assignments)
            if self.assignments[example_id] == split
        )

    def write_manifest(self, path: str | Path) -> None:
        """Write the already-canonical manifest as UTF-8 JSON."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True, slots=True)
class _CanonicalRecord:
    record: SplitRecord
    example_key: str
    identities: tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class _IdentityGroup:
    key: tuple[tuple[str, str], ...]
    members: tuple[_CanonicalRecord, ...]

    @property
    def size(self) -> int:
        return len(self.members)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _text(value: Any) -> str:
    if value is None:
        raise MissingIdentityError("identity values must not be null")
    if isinstance(value, bytes):
        raise SplitError("identity values must be text, not bytes")
    return str(value)


def _normalise(value: Any, field_name: str) -> tuple[str, str]:
    raw = unicodedata.normalize("NFKC", _text(value)).strip()
    normalised = raw.casefold()
    if not normalised:
        raise MissingIdentityError(f"{field_name} must be present and non-blank")
    return raw, normalised


def _canonical_record(record: SplitRecord) -> tuple[_CanonicalRecord, dict[str, tuple[str, str]]]:
    example_raw, example_key = _normalise(record.example_id, "example_id")
    if example_raw != record.example_id:
        record = SplitRecord(
            example_id=example_raw,
            singer_id=record.singer_id,
            song_id=record.song_id,
            phrase_id=record.phrase_id,
            content_hash=record.content_hash,
            metadata=record.metadata,
        )
    identity_values: list[str] = []
    raw_values: dict[str, tuple[str, str]] = {}
    for field_name in IDENTITY_FIELDS:
        raw, normalised = _normalise(getattr(record, field_name), field_name)
        identity_values.append(normalised)
        raw_values[field_name] = (raw, normalised)
    return _CanonicalRecord(record, example_key, tuple(identity_values)), raw_values


def _coerce_records(records: Iterable[SplitRecord | Mapping[str, Any]]) -> list[SplitRecord]:
    result: list[SplitRecord] = []
    for value in records:
        if isinstance(value, SplitRecord):
            result.append(value)
        elif isinstance(value, Mapping):
            result.append(SplitRecord.from_mapping(value))
        else:
            raise SplitError("each record must be SplitRecord or a mapping")
    if not result:
        raise SplitError("at least one record is required")
    return result


def _parse_ratios(ratios: Mapping[str, float] | Sequence[float]) -> dict[str, float]:
    if isinstance(ratios, Mapping):
        unknown = set(ratios) - set(SPLIT_NAMES)
        if unknown:
            raise InvalidRatiosError("unknown split ratio key(s): " + ", ".join(sorted(unknown)))
        values = [ratios.get(name, 0.0) for name in SPLIT_NAMES]
    else:
        if len(ratios) != len(SPLIT_NAMES):
            raise InvalidRatiosError("ratios must contain train, validation, and test values")
        values = list(ratios)
    parsed: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidRatiosError("ratios must be finite numbers") from exc
        if not math.isfinite(numeric) or numeric < 0:
            raise InvalidRatiosError("ratios must be finite and non-negative")
        parsed.append(numeric)
    total = sum(parsed)
    if total <= 0 or not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise InvalidRatiosError("ratios must sum to 1.0")
    return dict(zip(SPLIT_NAMES, parsed))


def _target_counts(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    exact = {name: total * ratios[name] for name in SPLIT_NAMES}
    counts = {name: math.floor(exact[name]) for name in SPLIT_NAMES}
    remainder = total - sum(counts.values())
    order = sorted(
        SPLIT_NAMES,
        key=lambda name: (-(exact[name] - counts[name]), SPLIT_NAMES.index(name)),
    )
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def _group_sort_key(group: _IdentityGroup, seed: int) -> tuple[str, tuple[tuple[str, str], ...]]:
    identity_text = "\x1f".join(f"{field_name}={value}" for field_name, value in group.key)
    digest = hashlib.sha256(f"{seed}\x1e{identity_text}".encode("utf-8")).hexdigest()
    return digest, group.key


def _group_id(group: _IdentityGroup) -> str:
    identity_text = "\x1f".join(f"{field_name}={value}" for field_name, value in group.key)
    return "group-" + hashlib.sha256(identity_text.encode("utf-8")).hexdigest()[:24]


def _build_groups(records: Sequence[SplitRecord], collision_policy: CollisionPolicy) -> tuple[_IdentityGroup, ...]:
    canonical: list[_CanonicalRecord] = []
    raw_forms: dict[str, dict[str, set[str]]] = {
        field_name: {} for field_name in (*IDENTITY_FIELDS, "example_id")
    }
    seen_examples: dict[str, _CanonicalRecord] = {}
    for record in records:
        item, raw_values = _canonical_record(record)
        example_raw = item.record.example_id
        raw_forms["example_id"].setdefault(item.example_key, set()).add(example_raw)
        for field_name, (raw, normalised) in raw_values.items():
            raw_forms[field_name].setdefault(normalised, set()).add(raw)
        previous = seen_examples.get(item.example_key)
        if previous is not None:
            raise DuplicateExampleError(
                f"duplicate example_id after normalization: {item.record.example_id!r}"
            )
        seen_examples[item.example_key] = item
        canonical.append(item)

    if collision_policy not in ("error", "merge"):
        raise SplitError("collision_policy must be 'error' or 'merge'")
    if collision_policy == "error":
        collisions = {
            field_name: {
                key: sorted(values)
                for key, values in values_by_key.items()
                if len(values) > 1
            }
            for field_name, values_by_key in raw_forms.items()
        }
        collisions = {field_name: values for field_name, values in collisions.items() if values}
        if collisions:
            details = "; ".join(
                f"{field_name}={list(values)}" for field_name, values in sorted(collisions.items())
            )
            raise IdentityCollisionError(
                "distinct raw identity values normalize to the same key; "
                f"use collision_policy='merge' to accept explicitly ({details})"
            )

    union_find = _UnionFind(len(canonical))
    token_owner: dict[tuple[str, str], int] = {}
    for index, item in enumerate(canonical):
        tokens = list(zip(IDENTITY_FIELDS, item.identities))
        if item.record.content_hash is not None:
            _, content_hash = _normalise(item.record.content_hash, "content_hash")
            tokens.append(("content_hash", content_hash))
        source_hashes = item.record.metadata.get("source_hashes", ())
        if isinstance(source_hashes, Sequence) and not isinstance(
            source_hashes, (str, bytes, bytearray)
        ):
            for source_hash in source_hashes:
                _, normalized_hash = _normalise(source_hash, "source_hashes")
                tokens.append(("source_hash", normalized_hash))
        for field_name, value in tokens:
            token = (field_name, value)
            owner = token_owner.get(token)
            if owner is not None:
                union_find.union(index, owner)
            else:
                token_owner[token] = index

    grouped: dict[int, list[_CanonicalRecord]] = {}
    for index, item in enumerate(canonical):
        grouped.setdefault(union_find.find(index), []).append(item)

    result: list[_IdentityGroup] = []
    for members in grouped.values():
        members.sort(key=lambda item: (item.example_key, item.record.example_id))
        identities = sorted(
            {
                (field_name, value)
                for item in members
                for field_name, value in zip(IDENTITY_FIELDS, item.identities)
            }
        )
        identities.extend(
            sorted(
                {
                    ("content_hash", _normalise(item.record.content_hash, "content_hash")[1])
                    for item in members
                    if item.record.content_hash is not None
                }
            )
        )
        identities.extend(
            sorted(
                {
                    ("source_hash", _normalise(source_hash, "source_hashes")[1])
                    for item in members
                    for source_hash in (
                        item.record.metadata.get("source_hashes", ())
                        if isinstance(item.record.metadata.get("source_hashes", ()), Sequence)
                        and not isinstance(
                            item.record.metadata.get("source_hashes", ()),
                            (str, bytes, bytearray),
                        )
                        else ()
                    )
                }
            )
        )
        result.append(_IdentityGroup(tuple(identities), tuple(members)))
    result.sort(key=lambda group: group.key)
    return tuple(result)


def _canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return json.dumps(
        manifest,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def split_records(
    records: Iterable[SplitRecord | Mapping[str, Any]],
    *,
    ratios: Mapping[str, float] | Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    collision_policy: CollisionPolicy = "error",
) -> SplitResult:
    """Split records without identity leakage.

    Identity groups are connected components over singer, song, and phrase
    values.  Consequently a singer, song, or phrase appears in at most one
    output split.  ``seed`` changes only the stable hash ordering of groups;
    it never depends on input row order.  Ratios are record-count targets, but
    groups remain indivisible, so exact counts are not promised.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SplitError("seed must be an integer")
    parsed_ratios = _parse_ratios(ratios)
    input_records = _coerce_records(records)
    groups = _build_groups(input_records, collision_policy)
    ordered_groups = sorted(groups, key=lambda group: _group_sort_key(group, seed))
    targets = _target_counts(len(input_records), parsed_ratios)
    active = [name for name in SPLIT_NAMES if parsed_ratios[name] > 0]
    if len(groups) < len(active):
        raise SplitError(
            "identity grouping cannot populate every active split: "
            f"groups={len(groups)} active_splits={len(active)}"
        )
    current = {name: 0 for name in SPLIT_NAMES}
    assigned_splits: dict[int, str] = {}
    seen_splits: set[str] = set()

    for index, group in enumerate(ordered_groups):
        unassigned = [name for name in active if name not in seen_splits]
        remaining_groups = len(ordered_groups) - index
        if unassigned and remaining_groups <= len(unassigned):
            # Reserve one remaining group for every active split when possible.
            split = min(
                unassigned,
                key=lambda name: (-parsed_ratios[name], SPLIT_NAMES.index(name)),
            )
        else:
            candidates = active or ["train"]
            split = min(
                candidates,
                key=lambda name: (
                    (current[name] + group.size) / max(targets[name], 1),
                    -parsed_ratios[name],
                    SPLIT_NAMES.index(name),
                ),
            )
        assigned_splits[id(group)] = split
        seen_splits.add(split)
        current[split] += group.size

    assignments: dict[str, str] = {}
    summaries: list[GroupSummary] = []
    for group in sorted(groups, key=lambda value: value.key):
        split = assigned_splits[id(group)]
        record_ids = tuple(item.record.example_id for item in group.members)
        summaries.append(
            GroupSummary(
                group_id=_group_id(group),
                split=split,
                record_ids=record_ids,
                identities=group.key,
            )
        )
        for item in group.members:
            assignments[item.record.example_id] = split

    manifest_groups = [summary.to_dict() for summary in summaries]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "seed": seed,
        "ratios": {name: parsed_ratios[name] for name in SPLIT_NAMES},
        "collision_policy": collision_policy,
        "record_count": len(input_records),
        "group_count": len(summaries),
        "target_record_counts": targets,
        "actual_record_counts": current,
        "assignments": {key: assignments[key] for key in sorted(assignments)},
        "groups": manifest_groups,
    }
    manifest["manifest_hash"] = hashlib.sha256(_canonical_manifest_bytes(manifest)).hexdigest()
    return SplitResult(assignments=assignments, groups=tuple(summaries), manifest=manifest)


deterministic_split = split_records


__all__ = [
    "CollisionPolicy",
    "DuplicateExampleError",
    "GroupSummary",
    "IDENTITY_FIELDS",
    "IdentityCollisionError",
    "InvalidRatiosError",
    "MissingIdentityError",
    "SCHEMA_VERSION",
    "SPLIT_NAMES",
    "SplitError",
    "SplitRecord",
    "SplitResult",
    "deterministic_split",
    "split_records",
]
