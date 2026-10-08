from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

from pip._vendor.distlib.scripts import ScriptMaker


def main() -> int:
    scripts_dir = Path(sys.executable).resolve().parent
    maker = ScriptMaker(None, str(scripts_dir))
    maker.clobber = True
    maker.variants = {""}
    count = 0
    for entry_point in importlib.metadata.entry_points(group="console_scripts"):
        maker.make(f"{entry_point.name} = {entry_point.value}")
        count += 1
    print(f"rebuilt {count} console scripts in {scripts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
