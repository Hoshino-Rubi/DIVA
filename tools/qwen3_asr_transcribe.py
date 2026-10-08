from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from any2xsampa.asr import (  # noqa: E402
    ASRTranscriptError,
    Qwen3ASRAdapter,
    Qwen3ASRLoadError,
    Qwen3ASRRequest,
    dumps_transcript_json,
)
from any2xsampa.asr.qwen3 import transcript_from_qwen_result  # noqa: E402


class Qwen3ASRCLIError(RuntimeError):
    def __init__(self, message: str, *, error_code: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.exit_code = exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe audio with Qwen3-ASR and write asr_transcript_v1 evidence JSON. "
            "ASR text is evidence only; phones/IPA remain the project truth layer."
        )
    )
    parser.add_argument("--audio", required=True, help="Input audio path.")
    parser.add_argument(
        "--model",
        default=None,
        help="Qwen3-ASR local model path or HuggingFace id. Required unless --fake/--fixture-result is used.",
    )
    parser.add_argument("--language", default=None, help="Optional Qwen3-ASR language hint.")
    parser.add_argument("--context", default="", help="Optional ASR context prompt/evidence hint.")
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Request optional char/token timestamps via Qwen3 forced aligner.",
    )
    parser.add_argument("--output", default=None, help="Write JSON to this path instead of stdout.")
    parser.add_argument(
        "--forced-aligner",
        default=None,
        help="Optional Qwen3 forced aligner model path/id for --timestamps with real backend.",
    )
    parser.add_argument("--device-map", default=None, help="Optional device_map forwarded to qwen_asr.")
    parser.add_argument("--dtype", default=None, help="Optional dtype name forwarded to qwen_asr.")
    parser.add_argument(
        "--fixture-result",
        default=None,
        help="Normalize saved raw/fake Qwen result JSON instead of loading a model.",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = run_from_args(args)
        output = dumps_transcript_json(payload)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output, encoding="utf-8")
        else:
            sys.stdout.write(output)
        return 0
    except Qwen3ASRCLIError as exc:
        _write_error(exc.error_code, str(exc), exit_code=exc.exit_code)
        return exc.exit_code
    except FileNotFoundError as exc:
        _write_error("qwen3_asr.audio_not_found", str(exc), exit_code=1)
        return 1
    except Qwen3ASRLoadError as exc:
        _write_error("qwen3_asr.model_unavailable", str(exc), exit_code=1)
        return 1
    except (ASRTranscriptError, ValueError) as exc:
        _write_error("qwen3_asr.transcript_error", str(exc), exit_code=1)
        return 1


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    audio_path = Path(args.audio)
    if args.fixture_result:
        raw = json.loads(Path(args.fixture_result).read_text(encoding="utf-8"))
        return transcript_from_qwen_result(
            raw,
            audio_path=audio_path,
            model_name=args.model or "fixture-qwen3-asr",
            language=args.language,
            context=args.context or "",
            forced_aligner=args.forced_aligner,
            device=args.device_map,
            dtype=args.dtype,
            model_output=False,
            extra_provenance={"fixture_result": str(args.fixture_result)},
        ).to_dict()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not args.fake and not _optional_text(args.model):
        raise Qwen3ASRCLIError(
            "Qwen3-ASR model is required. Pass --model with a local model path or "
            "HuggingFace id. Unit tests may use --fake to avoid loading a real model.",
            error_code="qwen3_asr.model_required",
        )
    backend = _FakeQwen3ASRBackend() if args.fake else None
    model_kwargs: dict[str, Any] = {}
    if _optional_text(getattr(args, "device_map", None)):
        model_kwargs["device_map"] = args.device_map
    if _optional_text(getattr(args, "dtype", None)):
        model_kwargs["dtype"] = args.dtype
    adapter = Qwen3ASRAdapter(
        model_name_or_path=args.model,
        backend=backend,
        forced_aligner=args.forced_aligner,
        model_kwargs=model_kwargs,
    )
    return adapter.transcribe(
        Qwen3ASRRequest(
            audio_path=audio_path,
            model=args.model,
            language=args.language,
            context=args.context or "",
            timestamps=bool(args.timestamps),
        )
    )


def _write_error(error_code: str, message: str, *, exit_code: int) -> None:
    payload = {
        "error": message,
        "error_code": error_code,
        "error_kind": error_code.split(".")[-1],
        "exit_code": exit_code,
    }
    sys.stderr.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


class _FakeQwen3ASRBackend:
    backend = "fake"
    device = "cpu"
    dtype = "float32"
    model_name_or_path = "fake-qwen3-asr"
    processor = None
    model = None
    forced_aligner = object()

    def transcribe(
        self,
        audio: Any,
        context: str = "",
        language: str | None = None,
        return_time_stamps: bool = False,
    ) -> list[dict[str, Any]]:
        text = "你好" if not context else context
        time_stamps = None
        if return_time_stamps:
            time_stamps = {
                "items": [
                    {
                        "text": char,
                        "start_time": round(index * 0.25, 3),
                        "end_time": round((index + 1) * 0.25, 3),
                    }
                    for index, char in enumerate(text)
                ]
            }
        return [
            {
                "language": language or "Chinese",
                "text": text,
                "time_stamps": time_stamps,
            }
        ]


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
