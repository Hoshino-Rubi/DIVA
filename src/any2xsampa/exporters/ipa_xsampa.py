"""Strict, loss-preserving IPA transcription for the local export layer."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import unicodedata

CONFIG = Path(__file__).resolve().parents[1] / "config/ipa_to_xsampa_v2.json"


@dataclass(frozen=True)
class Conversion:
    source_ipa: str
    xsampa: str
    status: str


class IpaXSampaConverter:
    """Convert whole labels, never discard an unsupported character."""

    def __init__(self, config_path: Path = CONFIG) -> None:
        raw = config_path.read_bytes()
        config = json.loads(raw)
        if config.get("schema_version") != "ipa_to_xsampa_export_v2":
            raise ValueError("Unsupported X-SAMPA export mapping")
        self.version = config["schema_version"]
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.bases = {unicodedata.normalize("NFD", k):v for k,v in config["base_symbols"].items()}
        self.base_keys = sorted(self.bases, key=len, reverse=True)
        self.modifiers = config["post_modifiers"]
        self.ties = config["ties"]
        self.aliases = config["legacy_aliases"]
        for table in (self.bases, self.modifiers, self.ties, self.aliases):
            if not all(isinstance(k, str) and k and isinstance(v, str)
                       and v and v.isascii() for k, v in table.items()):
                raise ValueError("Invalid X-SAMPA mapping entry")

    def convert(self, ipa: str) -> Conversion:
        if not isinstance(ipa, str) or not ipa:
            return Conversion(ipa, "", "unmapped")
        if ipa in self.aliases:
            return Conversion(ipa, self.aliases[ipa], "legacy_alias")
        normalized = unicodedata.normalize("NFD", ipa)
        pieces: list[str] = []
        has_base = False
        need_tied_base = False
        index = 0
        while index < len(normalized):
            char = normalized[index]
            base = next((b for b in self.base_keys if normalized.startswith(b, index)), None)
            if base is not None:
                pieces.append(self.bases[base])
                has_base, need_tied_base = True, False
                index += len(base)
                continue
            elif char in self.ties and has_base and not need_tied_base:
                pieces.append(self.ties[char])
                need_tied_base = True
            elif char in self.modifiers and has_base and not need_tied_base:
                pieces.append(self.modifiers[char])
            else:
                return Conversion(ipa, "", "unmapped")
            index += 1
        if not has_base or need_tied_base:
            return Conversion(ipa, "", "unmapped")
        return Conversion(ipa, "".join(pieces), "mapped")

    def project(self, phones: list[str]) -> dict:
        converted = [self.convert(phone) for phone in phones]
        missing = Counter(c.source_ipa for c in converted if not c.xsampa)
        total, unmapped = len(converted), sum(missing.values())
        return {
            "mapping_version": self.version,
            "mapping_sha256": self.sha256,
            "xsampa": [c.xsampa for c in converted],
            "phone_count": total,
            "mapped_phone_count": total - unmapped,
            "unmapped_phone_count": unmapped,
            "unmapped_phone_ratio": unmapped / total if total else 0.0,
            "unmapped_label_counts": dict(sorted(missing.items())),
            "legacy_alias_phone_count": sum(c.status == "legacy_alias" for c in converted),
            "recognition_accuracy_measured": False,
        }
