from __future__ import annotations

from pathlib import Path
from typing import Any

from any2xsampa.core.interfaces import EnginePhoneExporter
from any2xsampa.core.io import load_json
from any2xsampa.core.models import EnginePhoneExport, PhoneSegment
from any2xsampa.exporters.inventory import (
    AliasCandidate,
    AliasInventoryValidator,
    AliasSelectionContext,
)


class MappingEnginePhoneExporter(EnginePhoneExporter):
    """Config-driven engine export that leaves IPA/X-SAMPA truth untouched."""

    def __init__(self, format_name: str, config_path: Path) -> None:
        self.format_name = format_name
        self.config_path = Path(config_path)
        self.config = load_json(self.config_path)
        self.mapping = self.config.get("mapping", {})
        if not isinstance(self.mapping, dict):
            raise ValueError(f"Exporter mapping must be an object: {self.config_path}")
        self.default_unmapped_phone = str(
            self.config.get("default_unmapped_phone", "UNMAPPED")
        )

    def export(self, segments: list[PhoneSegment]) -> list[PhoneSegment]:
        for segment in segments:
            source = segment.normalized_ipa or segment.xsampa
            raw_rule = self.mapping.get(source)
            rule = dict(raw_rule) if isinstance(raw_rule, dict) else {}
            unmapped = not rule
            engine_phone = str(rule.get("engine_phone", self.default_unmapped_phone))
            notes = [str(value) for value in rule.get("approximation_notes", [])]
            segment.engine_export = EnginePhoneExport(
                format=self.format_name,
                original_ipa=segment.normalized_ipa,
                xsampa=segment.xsampa,
                engine_phone=engine_phone,
                approximation_applied=bool(rule.get("approximation_applied", unmapped)),
                aliases=[engine_phone],
                approximation_notes=notes,
                unmapped=unmapped,
                fallback_used=unmapped,
                selected_alias=engine_phone,
                selection_reason="config mapping" if not unmapped else "default unmapped token",
            )
            segment.metadata["engine_exported_from"] = source
            segment.metadata["engine_export_format"] = self.format_name
        return segments


class OpenUtauAliasEngineExporter(MappingEnginePhoneExporter):
    def __init__(
        self,
        format_name: str,
        config_path: Path,
        *,
        profile_name: str | None = None,
        inventory_path: Path | None = None,
        metadata_preferences: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__(format_name, config_path)
        self.profile_name = profile_name or str(
            self.config.get("default_profile", "generic_openutau")
        )
        profiles = self.config.get("bank_profiles", {})
        if not isinstance(profiles, dict) or self.profile_name not in profiles:
            available = ", ".join(sorted(profiles)) if isinstance(profiles, dict) else ""
            raise ValueError(
                f"Unknown OpenUtau profile '{self.profile_name}'. Available profiles: {available}"
            )
        self.profile = dict(profiles[self.profile_name])
        self.inventory = AliasInventoryValidator(inventory_path)
        self.metadata_preferences = dict(metadata_preferences or {})

    def export(self, segments: list[PhoneSegment]) -> list[PhoneSegment]:
        super().export(segments)
        for segment in segments:
            export = segment.engine_export
            if export is None:
                continue
            candidates = [
                AliasCandidate(
                    alias=alias,
                    alias_type="plain",
                    role=_segment_role(segment),
                    is_fallback=export.fallback_used,
                )
                for alias in export.aliases
            ]
            context = AliasSelectionContext(
                current_role=_segment_role(segment),
                bank_profile=self.profile_name,
                alias_strategy=str(self.profile.get("alias_strategy", "plain")),
                preferred_subbanks=list(self.metadata_preferences.get("subbanks", [])),
                preferred_colors=list(self.metadata_preferences.get("colors", [])),
                preferred_tags=list(self.metadata_preferences.get("tags", [])),
                preferred_styles=list(self.metadata_preferences.get("styles", [])),
            )
            result = self.inventory.validate_candidates(candidates, selection_context=context)
            export.bank_profile = self.profile_name
            export.alias_strategy = context.alias_strategy
            export.alias_inventory_checked = result.checked
            export.alias_inventory_hits = result.hits
            export.alias_inventory_misses = result.misses
            export.validated_aliases = result.validated_aliases
            export.repaired_aliases = result.repaired_aliases
            export.unresolved_aliases = result.unresolved_aliases
            export.inventory_fallback_used = result.fallback_used
            export.candidate_aliases = result.candidate_aliases
            export.selected_alias = result.selected_alias or export.engine_phone
            export.selection_reason = result.selection_reason
            export.ranking_features = result.ranking_features
            export.engine_phone = export.selected_alias
        return segments


def _segment_role(segment: PhoneSegment) -> str:
    group = segment.syllable_group
    if group is None:
        return "unknown"
    index = segment.metadata.get("segment_index")
    if isinstance(index, int):
        if index in group.onset_segment_indices:
            return "onset"
        if index == group.nucleus_segment_index:
            return "nucleus"
        if index in group.coda_segment_indices:
            return "coda"
    return "unknown"


__all__ = ["MappingEnginePhoneExporter", "OpenUtauAliasEngineExporter"]
