from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from any2xsampa.training import cantonese_singing_ctc as ctc  # noqa: E402


def _alignment(reference: Sequence[str], prediction: Sequence[str]) -> list[tuple[str | None, str | None]]:
    rows, cols = len(reference) + 1, len(prediction) + 1
    costs = [[0] * cols for _ in range(rows)]
    back: list[list[str | None]] = [[None] * cols for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0], back[i][0] = i, "del"
    for j in range(1, cols):
        costs[0][j], back[0][j] = j, "ins"
    for i in range(1, rows):
        for j in range(1, cols):
            options = [
                (costs[i - 1][j] + 1, "del"),
                (costs[i][j - 1] + 1, "ins"),
                (costs[i - 1][j - 1] + (reference[i - 1] != prediction[j - 1]), "diag"),
            ]
            costs[i][j], back[i][j] = min(options, key=lambda item: (item[0], item[1] != "diag"))
    aligned: list[tuple[str | None, str | None]] = []
    i, j = len(reference), len(prediction)
    while i or j:
        action = back[i][j]
        if action == "diag":
            aligned.append((reference[i - 1], prediction[j - 1])); i -= 1; j -= 1
        elif action == "del":
            aligned.append((reference[i - 1], None)); i -= 1
        else:
            aligned.append((None, prediction[j - 1])); j -= 1
    return list(reversed(aligned))


def build_report(dataset_path: Path, feature_cache: Path, checkpoint: Path, fold_index: int) -> dict[str, Any]:
    import torch

    dataset = json.loads(dataset_path.read_text(encoding="utf-8-sig"))
    examples = ctc._examples(dataset)
    _, validation, split_identity = ctc._split(dataset, examples, fold_index)
    cache = torch.load(feature_cache, map_location="cpu", weights_only=True)
    features, _, _ = ctc._validated_feature_cache(cache, examples=examples, torch=torch)
    head, checkpoint_payload = ctc.load_ctc_head(checkpoint, device_name="cpu")
    vocabulary = [str(value) for value in checkpoint_payload["vocabulary"]]
    blank_id = int(checkpoint_payload["configuration"]["blank_id"])
    confusion: Counter[tuple[str, str]] = Counter()
    reference_counts: Counter[str] = Counter()
    correct_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    total_edits = 0
    total_reference = 0
    with torch.inference_mode():
        for row in validation:
            example_id = str(row["example_id"])
            logits = head(features[example_id].float().unsqueeze(0))[0]
            ids = ctc._ctc_collapse(logits.argmax(dim=-1).tolist(), blank_id=blank_id)
            predicted = [vocabulary[index] for index in ids]
            reference = [str(value) for value in row["target_phone_tokens"]]
            aligned = _alignment(reference, predicted)
            edits = sum(left != right for left, right in aligned)
            total_edits += edits
            total_reference += len(reference)
            reference_counts.update(reference)
            for left, right in aligned:
                if left is not None and left == right:
                    correct_counts[left] += 1
                if left != right:
                    confusion[(left or "<ins>", right or "<del>")] += 1
            rows.append(
                {
                    "example_id": example_id,
                    "group_id": row["group_id"],
                    "reference": reference,
                    "prediction": predicted,
                    "edit_distance": edits,
                    "token_similarity": max(0.0, 1.0 - edits / max(1, len(reference))),
                    "exact": reference == predicted,
                }
            )
    per_phone = {
        phone: {
            "reference_count": reference_counts[phone],
            "aligned_correct": correct_counts[phone],
            "aligned_recall": correct_counts[phone] / reference_counts[phone],
        }
        for phone in sorted(reference_counts)
    }
    return {
        "schema_version": "cross_song_ctc_evaluation_v1",
        "dataset_path": str(dataset_path.resolve()),
        "feature_cache_path": str(feature_cache.resolve()),
        "checkpoint_path": str(checkpoint.resolve()),
        "fold_index": fold_index,
        "split_identity": split_identity,
        "validation_groups": sorted({str(row["group_id"]) for row in validation}),
        "example_count": len(rows),
        "reference_token_count": total_reference,
        "edit_count": total_edits,
        "token_error_rate": total_edits / max(1, total_reference),
        "token_similarity": max(0.0, 1.0 - total_edits / max(1, total_reference)),
        "exact_sequence_ratio": sum(row["exact"] for row in rows) / max(1, len(rows)),
        "reference_inventory": sorted(reference_counts),
        "prediction_inventory": sorted({token for row in rows for token in row["prediction"]}),
        "per_phone": per_phone,
        "top_confusions": [
            {"reference": left, "prediction": right, "count": count}
            for (left, right), count in confusion.most_common(40)
        ],
        "examples": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report cross-song CTC sequence and per-phone errors.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = build_report(args.dataset, args.feature_cache, args.checkpoint, args.fold_index)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: report[key] for key in ("fold_index", "validation_groups", "example_count", "token_similarity", "exact_sequence_ratio")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
