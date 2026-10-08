from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from any2xsampa.training.acoustic_features import (  # noqa: E402
    FrozenWavLMFeatureExtractor,
    FrozenXLSRFeatureExtractor,
    LibrosaLogMelFeatureExtractor,
    OnnxEspeakLogitFeatureExtractor,
)
from any2xsampa.training.cantonese_singing_ctc import write_feature_cache  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract pinned acoustic features for a CTC dataset.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--extractor",
        choices=["logmel", "onnx-espeak-logits", "wavlm", "xls-r"],
        default="logmel",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--backbone-manifest", type=Path)
    parser.add_argument("--hidden-layer", type=int, default=-1)
    args = parser.parse_args(argv)
    try:
        dataset = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
        examples = dataset.get("examples", [])
        if args.extractor == "logmel":
            extractor = LibrosaLogMelFeatureExtractor()
        elif args.extractor == "onnx-espeak-logits":
            extractor = OnnxEspeakLogitFeatureExtractor(
                device=args.device,
                cpu_thread_limit=args.cpu_threads,
            )
        elif args.extractor == "wavlm":
            if args.backbone_manifest is None:
                raise ValueError("--backbone-manifest is required for the wavlm extractor")
            extractor = FrozenWavLMFeatureExtractor(
                manifest_path=args.backbone_manifest,
                device=args.device,
                hidden_layer=args.hidden_layer,
                cpu_thread_limit=args.cpu_threads,
            )
        else:
            if args.backbone_manifest is None:
                raise ValueError("--backbone-manifest is required for the xls-r extractor")
            extractor = FrozenXLSRFeatureExtractor(
                manifest_path=args.backbone_manifest,
                device=args.device,
                hidden_layer=args.hidden_layer,
                cpu_thread_limit=args.cpu_threads,
            )
        features = extractor.extract(examples)
        metadata = extractor.metadata()
        cache = write_feature_cache(
            dataset_path=args.dataset,
            features=features,
            output_path=args.output,
            model_sha256=str(metadata["model_sha256"]),
        )
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"extractor": metadata, "cache": cache}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
