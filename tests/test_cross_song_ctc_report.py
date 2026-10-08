from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "report_cross_song_ctc.py"
SPEC = importlib.util.spec_from_file_location("report_cross_song_ctc", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_alignment_records_substitution_insertion_and_deletion() -> None:
    aligned = MODULE._alignment(["a", "b", "c"], ["a", "x", "c", "d"])
    assert aligned[0] == ("a", "a")
    assert ("b", "x") in aligned
    assert (None, "d") in aligned
