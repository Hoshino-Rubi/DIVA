from __future__ import annotations

from pathlib import Path

from any2xsampa.dialects.base import InventoryDialectPlugin


class TaigiDialectPlugin(InventoryDialectPlugin):
    def __init__(self) -> None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "dialect_taigi.json"
        super().__init__(config_path)
