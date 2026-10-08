from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, write_json_new  # noqa: E402


USER_STATEMENT = "沙龙是存在开头偏移，但是音素，音符排列顺序均严格按照歌曲进行排列，我已确认（即：每个字都在正确的顺序上）"


def main() -> int:
    parser = argparse.ArgumentParser(description="Record the user's exact Shalong order/offset confirmation without changing source annotations.")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    template = read_json(args.template)
    if template.get("schema_version") != "original_clip_binding_reviews_v1" or len(template.get("items", [])) != 3:
        raise ValueError("expected the three-item Shalong pending review template")
    items = []
    for item in template["items"]:
        if item.get("song_id") != "shalong" or item.get("split") != "dev":
            raise ValueError("confirmation is limited to Shalong development clips")
        row = dict(item)
        row.update({
            "status": "human_confirmed_order_only",
            "reviewer": "project_user",
            "review_statement": USER_STATEMENT,
            "complete_sequence_confirmed": False,
            "suggestion_only": True,
            "confirmation_scope": "The full-song phoneme/note order is confirmed and an opening offset exists. This does not confirm the machine-selected crop boundary, exact rendered interval, or millisecond phone boundaries.",
            "confirmed_local_date": "2026-09-06",
            "confirmed_timezone": "Asia/Taipei",
            "source_phoneme_approval_preserved": True,
            "raw_vsqx_modified": False,
        })
        items.append(row)
    payload = {
        "schema_version": "original_clip_binding_reviews_v1",
        "items": items,
        "scope": "Shalong full-song phoneme/note order only; a leading offset exists. Exact fixed-clip boundaries and numeric clock offset are not confirmed.",
        "evidence": {
            "source": "direct_user_statement_in_current_conversation",
            "statement": USER_STATEMENT,
            "statement_sha256": __import__("hashlib").sha256(USER_STATEMENT.encode("utf-8")).hexdigest(),
            "template_path": str(args.template.resolve()),
            "template_sha256": sha256_file(args.template),
            "new_complete_songs_requested": 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_new(args.output, payload)
    print(f"confirmed_items={len(items)}; output_sha256={sha256_file(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
