"""Independent audit of the non-human portions of the stepwise continuation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file, write_json_new  # noqa: E402


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__(); self.audio: list[str] = []; self.images: list[str] = []
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "audio": self.audio.append(values["src"])
        if tag == "img": self.images.append(values["src"])


def inspect_packet(root: Path, folder: str, expected_items: int, expected_media: int) -> dict:
    directory = root / folder
    packet = read_json(directory / "review-packet.json")
    if packet["count"] != expected_items or packet.get("test_songs_opened") is True:
        raise ValueError(f"invalid packet scope: {folder}")
    parser = Links(); parser.feed((directory / "index.html").read_text(encoding="utf-8"))
    if len(parser.audio) != expected_media or len(parser.images) != expected_media:
        raise ValueError(f"invalid media count: {folder}")
    for link in parser.audio + parser.images:
        path = (directory / link).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f"invalid media link: {folder}/{link}")
    return {"packet_sha256": sha256_file(directory / "review-packet.json"), "items": packet["count"], "audio_links": len(parser.audio), "image_links": len(parser.images), "links_valid": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_root.resolve()
    shalong = inspect_packet(run, "shalong-review-v2-final", 6, 18)
    training = inspect_packet(run, "training-candidate-review-v5", 20, 40)
    source_run = ROOT / ".state/runs/quality-first8h-20260907T010738"
    before = read_json(source_run / "preservation-before.json")
    for source in before["sources"]: verify_file(Path(source["path"]), source["sha256"])
    for name, digest in before["artifacts"].items(): verify_file(ROOT / name, digest)
    result = {"schema_version": "original_reviewed_stepwise_delivery_audit_v1", "verified_at_utc": datetime.now(timezone.utc).isoformat(), "packets": {"shalong": shalong, "training": training}, "external_sources_verified": len(before["sources"]), "historical_artifacts_verified": len(before["artifacts"]), "new_training_started": False, "original_evaluation_started": False, "test_predictions_opened": False, "source_wav_vsqx_modified": False, "human_confirmation_required": True, "overall_goal_complete": False}
    output = run / "DELIVERY_AUDIT.json"
    if output.exists(): raise FileExistsError(output)
    digest = write_json_new(output, result)
    print(f"audit={output}; sha256={digest}; sources={len(before['sources'])}; historical={len(before['artifacts'])}; goal_complete=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
