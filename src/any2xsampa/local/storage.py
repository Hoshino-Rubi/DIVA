"""Version-independent writable state for installed applications; legacy roots stay local."""
from __future__ import annotations

import json
import os
from pathlib import Path


def data_home(root: Path) -> Path | None:
    """An explicit override is useful for portable installs and isolated acceptance."""
    explicit = os.environ.get("ANY2XSAMPA_DATA_HOME")
    if explicit:
        directory = Path(explicit).expanduser()
        if not directory.is_absolute():
            raise ValueError("ANY2XSAMPA_DATA_HOME must be an absolute directory")
        return directory.resolve()
    config = root / "local-app.json"
    if not config.is_file():
        return None
    value = json.loads(config.read_text(encoding="utf-8-sig"))
    mode = value.get("storage_mode", "portable_v1")
    if mode == "portable_v1":
        return None
    if mode != "per_user_v1":
        raise ValueError("Unsupported application storage mode")
    parent = os.environ.get("LOCALAPPDATA")
    if not parent or not Path(parent).is_absolute():
        raise ValueError("Per-user storage requires LOCALAPPDATA or ANY2XSAMPA_DATA_HOME")
    return (Path(parent) / "Any2Xsampa").resolve()


def state_dir(root: Path) -> Path:
    home = data_home(root)
    return home / "state" if home else root / ".state"


def model_cache(root: Path) -> Path:
    home = data_home(root)
    return home / "cache" / "model-downloads" if home else root / ".state/model-downloads"
