"""Fetch pinned POWSM metadata through an explicit proxy, without direct fallback."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request


def fetch(url: str, proxy: str) -> bytes:
    parsed = urllib.parse.urlparse(proxy)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or not parsed.port:
        raise ValueError("an explicit HTTP(S) proxy with a port is required")
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    )
    request = urllib.request.Request(url, headers={"User-Agent": "any2xsampa-powsm-probe/1"})
    with opener.open(request, timeout=20) as response:
        data = response.read(8 * 1024 * 1024 + 1)
    if len(data) > 8 * 1024 * 1024:
        raise ValueError("metadata response exceeds bounded size")
    return data


def inspect(output: Path, proxy: str) -> None:
    output.mkdir(parents=True, exist_ok=False)
    records = []
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "proxy": proxy,
        "direct_fallback": False,
        "models": records,
        "model_weights_downloaded": False,
    }
    try:
        for model_id in ("espnet/powsm_ctc", "espnet/powsm"):
            url = "https://huggingface.co/api/models/" + model_id + "?blobs=true"
            raw = fetch(url, proxy)
            doc = json.loads(raw)
            revision = doc.get("sha", "")
            if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                raise ValueError("metadata lacks an immutable revision")
            destination = output / (model_id.replace("/", "-") + ".metadata.json")
            destination.write_bytes(raw)
            record = {
                "model_id": model_id,
                "revision": revision,
                "license": doc.get("cardData", {}).get("license"),
                "metadata_sha256": hashlib.sha256(raw).hexdigest(),
                "metadata_url": url,
                "files": doc.get("siblings", []),
            }
            records.append(record)
        result["status"] = "metadata_available"
    except (OSError, ValueError) as exc:
        result.update(status="blocked_proxy_or_metadata", error=str(exc))
    (output / "PREFLIGHT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--proxy", required=True)
    args = parser.parse_args()
    inspect(args.output.resolve(), args.proxy)
