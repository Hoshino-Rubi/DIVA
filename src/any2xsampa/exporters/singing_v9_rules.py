"""Development-only, whole-syllable pronunciation witnesses, never a Cartesian allowlist."""
from pathlib import Path
import json
from .miku_v4c import is_legal
from ..vocaloid.phoneme_policy import validate_miku_v4c_note_phonemes

CONFIG = Path(__file__).resolve().parents[1] / "config"
RULES = CONFIG / "cantonese_singing_v9_rules.json"

def load_rules() -> dict:
    return json.loads(RULES.read_text(encoding="utf8"))

def allowed(phones, rules) -> bool:
    return (is_legal(phones) or " ".join(phones) in rules["extensions"]) and not validate_miku_v4c_note_phonemes(phones)

def equivalent(a, b, rules) -> bool:
    if not (allowed(a,rules) and allowed(b,rules)):
        return False
    if a==b: return True
    # Only the catalogued, same-initial Mandarin ong allographs. Preserve stop type.
    def key(p):
        tail=p[-1] if p[-1] in ("t","k") else ""
        base=" ".join(p[:-1] if tail else p)
        return rules["equivalent_bases"].get(base,base),tail
    return key(a)==key(b)
