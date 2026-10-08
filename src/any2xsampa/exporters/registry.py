from __future__ import annotations

from pathlib import Path

from any2xsampa.core.interfaces import EnginePhoneExporter
from any2xsampa.exporters.engine import MappingEnginePhoneExporter, OpenUtauAliasEngineExporter


class ExporterRegistry:
    def __init__(self, config_root: Path) -> None:
        self.config_root = config_root
        self._exporters: dict[str, EnginePhoneExporter] = {
            "vocaloid_like": MappingEnginePhoneExporter(
                "vocaloid_like",
                config_root / "exporter_vocaloid_like.json",
            ),
        }

    def get(
        self,
        export_format: str,
        engine_profile: str | None = None,
        engine_inventory: Path | None = None,
        engine_metadata_preferences: dict[str, list[str]] | None = None,
    ) -> EnginePhoneExporter | None:
        key = export_format.lower()
        if key == "xsampa":
            return None
        if key == "openutau_like":
            return OpenUtauAliasEngineExporter(
                "openutau_like",
                self.config_root / "exporter_openutau_like.json",
                profile_name=engine_profile,
                inventory_path=engine_inventory,
                metadata_preferences=engine_metadata_preferences,
            )
        if key not in self._exporters:
            available = ", ".join(["xsampa", *sorted(self._exporters)])
            raise KeyError(f"Unknown export format '{export_format}'. Available formats: {available}")
        return self._exporters[key]
