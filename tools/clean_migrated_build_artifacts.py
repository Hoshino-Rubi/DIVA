from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def remove_tree(path: Path) -> bool:
    resolved = path.resolve(strict=False)
    resolved.relative_to(ROOT)
    if not path.exists():
        return False
    shutil.rmtree(path)
    return True


def main() -> int:
    removed_directories = 0
    for path in sorted(ROOT.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir() and path.name in {"__pycache__", ".pytest_cache"}:
            removed_directories += int(remove_tree(path))

    for relative in (
        "src/any2xsampa.egg-info",
        "apps/Any2Xsampa.Desktop/bin",
        "apps/Any2Xsampa.Desktop/obj",
    ):
        removed_directories += int(remove_tree(ROOT / relative))

    removed_direct_urls = 0
    site_packages = ROOT / ".venv" / "Lib" / "site-packages"
    for pattern in (
        "allosaurus-*.dist-info/direct_url.json",
        "editdistance-*.dist-info/direct_url.json",
        "pdm_backend-*.dist-info/direct_url.json",
        "qwen_asr-*.dist-info/direct_url.json",
        "torch-*.dist-info/direct_url.json",
    ):
        for path in site_packages.glob(pattern):
            path.unlink()
            removed_direct_urls += 1

    print(
        f"removed_directories={removed_directories} "
        f"removed_direct_url_metadata={removed_direct_urls}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
