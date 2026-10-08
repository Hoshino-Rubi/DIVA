from __future__ import annotations

from any2xsampa.core.interfaces import DialectPlugin
from any2xsampa.dialects.cantonese import CantoneseDialectPlugin
from any2xsampa.dialects.taigi import TaigiDialectPlugin


class DialectRegistry:
    def __init__(self) -> None:
        cantonese = CantoneseDialectPlugin()
        taigi = TaigiDialectPlugin()
        self._plugins: dict[str, DialectPlugin] = {
            "cantonese": cantonese,
            "taigi": taigi,
        }
        self._aliases: dict[str, str] = {"taiwanese": "taigi"}

    def get(self, name: str) -> DialectPlugin:
        key = name.lower()
        canonical_key = self._aliases.get(key, key)
        if canonical_key not in self._plugins:
            available = ", ".join(self.list_names())
            raise KeyError(f"Unknown dialect '{name}'. Available dialects: {available}")
        return self._plugins[canonical_key]

    def list_names(self) -> list[str]:
        return sorted([*self._plugins, *self._aliases])
