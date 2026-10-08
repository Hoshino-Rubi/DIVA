from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock offline recognizer command for real-mode demos.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lyrics", default="")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    payload = {
        "segments": [
            {
                "start_sec": 0.0,
                "end_sec": 0.36,
                "confidence": 0.9,
                "candidate_phones": [
                    {"symbol": "l", "confidence": 0.9},
                    {"symbol": "n", "confidence": 0.18},
                ],
                "metadata": {"producer": "mock_real_recognizer"},
            },
            {
                "start_sec": 0.36,
                "end_sec": 0.78,
                "confidence": 0.86,
                "candidate_phones": [
                    {"symbol": "aa", "confidence": 0.86},
                    {"symbol": "ax", "confidence": 0.21},
                ],
                "metadata": {"producer": "mock_real_recognizer"},
            },
            {
                "start_sec": 0.78,
                "end_sec": 1.14,
                "confidence": 0.82,
                "candidate_phones": [
                    {"symbol": "ts_h", "confidence": 0.82},
                    {"symbol": "ts", "confidence": 0.35},
                ],
                "metadata": {"producer": "mock_real_recognizer", "lyrics": args.lyrics},
            },
        ]
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
