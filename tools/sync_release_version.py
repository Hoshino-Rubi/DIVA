"""Generate version mirrors from the reviewed release-version.json; --check never writes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def sync(root: Path, *, check: bool) -> list[str]:
    version = json.loads((root / "release-version.json").read_text(encoding="utf-8"))
    number = version["python_version"]
    tag = version["tag"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", number) or tag != "v" + number or version["prerelease"] is not False:
        raise ValueError("Expected an explicit stable release identity")
    if version["assembly_version"] != number + ".0":
        raise ValueError("Assembly identity does not match the release")
    targets = {
        "Directory.Build.props": (
            "<Project>\n <PropertyGroup>\n"
            f"  <Version>{number}</Version>\n"
            f"  <AssemblyVersion>{number}.0</AssemblyVersion>\n"
            f"  <FileVersion>{number}.0</FileVersion>\n"
            f"  <InformationalVersion>{tag}</InformationalVersion>\n"
            " </PropertyGroup>\n</Project>\n"
        ),
    }
    for name, pattern, replacement in [
        ("pyproject.toml", r'(?m)^version = "[^"]+"$', f'version = "{number}"'),
        ("src/any2xsampa/__init__.py", r'(?m)^__version__ = "[^"]+"$', f'__version__ = "{number}"'),
    ]:
        original = (root / name).read_text(encoding="utf-8")
        replaced, count = re.subn(pattern, replacement, original)
        if count != 1:
            raise ValueError("Expected exactly one version in " + name)
        targets[name] = replaced
    drift = []
    for name, content in targets.items():
        if (root / name).read_text(encoding="utf-8") != content:
            drift.append(name)
            if not check:
                (root / name).write_text(content, encoding="utf-8", newline="\n")
    if check and drift:
        raise ValueError("Version mirrors need regeneration: " + ", ".join(drift))
    return drift


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps({"changed": sync(ROOT, check=args.check), "check": args.check}))
