"""Download only the frozen POWSM-CTC snapshot and one pinned source archive."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def download(url: str, target: Path, proxy: str, expected: str | None, size: int | None) -> dict:
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    before = time.monotonic()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    count = 0
    with opener.open(url, timeout=60) as response, temporary.open("xb") as stream:
        while data := response.read(4 * 1024 * 1024):
            count += len(data)
            if count > (size if size is not None else 100 * 1024 * 1024):
                raise ValueError("download exceeds declared limit")
            stream.write(data)
            digest.update(data)
    actual = digest.hexdigest()
    if (expected is not None and actual != expected) or (size is not None and count != size):
        raise ValueError("download hash or size mismatch")
    temporary.rename(target)
    return {"path": str(target.resolve()), "sha256": actual, "bytes": count,
            "elapsed_seconds": time.monotonic() - before, "source_url": url,
            "expected_hash_verified": expected is not None}


def main(run: Path, proxy: str) -> None:
    doc = json.loads((run / "espnet-powsm_ctc.metadata.json").read_text())
    commit = json.loads((run / "espnet-202604-commit.json").read_text())[0]["sha"]
    weight = next(x for x in doc["siblings"] if x["rfilename"].endswith(".pth"))
    rows = []
    for name, url, digest, size in [
        ("espnet-source.zip", "https://codeload.github.com/espnet/espnet/zip/" + commit, None, None),
        ("model/" + weight["rfilename"], "https://huggingface.co/espnet/powsm_ctc/resolve/" + doc["sha"] + "/" + weight["rfilename"], weight["lfs"]["sha256"], weight["size"]),
    ]:
        record = download(url, run / name, proxy, digest, size)
        rows.append(record)
        print(json.dumps(record), flush=True)
    result = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "proxy": proxy,
              "direct_fallback": False, "espnet_revision": commit, "files": rows}
    with (run / "DOWNLOADS.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--proxy", required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.proxy)
