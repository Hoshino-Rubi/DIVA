"""Read only named dependency metadata / explicit flat wheelhouses for CPython3.12.

No dependency imports, installation, copying, inference or implicit cache search.
Wheel tags establish ABI eligibility only; they do not prove model/API compatibility.
"""
from __future__ import annotations

import argparse
import csv
from email.parser import Parser
import json
from pathlib import Path
import re
from typing import Any


REUSED = (
    "torch", "torchaudio", "transformers", "numpy", "scipy", "soundfile",
    "PyYAML", "packaging", "typing_extensions", "requests", "cffi",
    "safetensors", "tokenizers", "huggingface_hub",
)
EXTRAS = (
    "beartype", "rotary_embedding_torch", "einops", "librosa", "soxr", "numba",
    "llvmlite", "audioread", "pooch", "lazy_loader", "msgpack", "scikit_learn",
    "joblib", "decorator", "platformdirs", "threadpoolctl", "narwhals", "psutil",
)


def compatible_tag(tag: str) -> bool:
    python, abi, platform = tag.strip().split("-")
    if platform == "any":
        return abi == "none" and "py3" in python.split(".")
    if platform != "win_amd64":
        return False
    if abi == "none" and "py3" in python.split("."):
        return True
    if python == "cp312" and abi == "cp312":
        return True
    match = re.fullmatch(r"cp(3\d+)", python)
    return abi == "abi3" and match is not None and 32 <= int(match[1]) <= 312


def installed(site: Path, name: str) -> dict[str, Any] | None:
    matches = list(site.glob(name + "-*.dist-info"))
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("Ambiguous distribution metadata: " + name)
    root = matches[0]
    metadata = Parser().parsestr((root / "METADATA").read_text(encoding="utf-8"))
    wheel = Parser().parsestr((root / "WHEEL").read_text(encoding="utf-8"))
    tags = wheel.get_all("Tag", [])
    size = 0
    with (root / "RECORD").open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) == 3 and row[2].isdigit():
                size += int(row[2])
    return {
        "metadata": str(root), "version": metadata["Version"], "wheel_tags": tags,
        "cp312_windows_tag_eligible": any(compatible_tag(tag) for tag in tags),
        "requires_python": metadata["Requires-Python"],
        "requires_dist": metadata.get_all("Requires-Dist", []),
        "license": metadata["License-Expression"] or metadata["License"],
        "license_files": metadata.get_all("License-File", []), "record_bytes": size,
    }


def cached_wheels(roots: list[Path], name: str) -> list[str]:
    result = []
    for root in roots:
        for wheel in root.glob(name + "-*.whl"):
            parts = wheel.stem.rsplit("-", 3)
            if len(parts) == 4 and compatible_tag("-".join(parts[1:])):
                result.append(str(wheel))
    return sorted(result)


def inspect(runtime: Path, donor: Path, wheelhouses: list[Path]) -> dict[str, Any]:
    for path in (runtime, donor, *wheelhouses):
        if not path.is_dir():
            raise NotADirectoryError(path)
    rows = []
    for name in REUSED + EXTRAS:
        current, candidate = installed(runtime, name), installed(donor, name)
        wheels = cached_wheels(wheelhouses, name)
        present = bool(current and current["cp312_windows_tag_eligible"])
        donor_ok = bool(candidate and candidate["cp312_windows_tag_eligible"])
        rows.append({
            "name": name, "optional_monitor": name == "psutil", "runtime": current,
            "donor": candidate, "eligible_cached_wheels": wheels,
            "plan": ("reuse_runtime" if present else "candidate_from_donor" if donor_ok
                     else "candidate_from_wheelhouse" if wheels else "unresolved"),
        })
    return {
        "target": "cp312-win_amd64", "packages": rows,
        "unresolved": [r["name"] for r in rows if r["plan"] == "unresolved" and not r["optional_monitor"]],
        "verification": "metadata and tags only; no dependency imports or runtime qualification",
        "dependency_plan": "Reuse phonetic runtime; add compatible extras with license files. Never copy cp314-only binaries.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-site-packages", type=Path, required=True)
    parser.add_argument("--donor-site-packages", type=Path, required=True)
    parser.add_argument("--wheel-cache", type=Path, action="append", default=[],
                        help="Explicit flat directory of .whl files; no recursive or implicit cache search")
    args = parser.parse_args(argv)
    result = inspect(args.runtime_site_packages, args.donor_site_packages, args.wheel_cache)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if result["unresolved"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
