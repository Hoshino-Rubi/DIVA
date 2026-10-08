from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from any2xsampa.core.io import dump_json, write_text
from any2xsampa.presentation import augment_pipeline_payload, build_transport_exception_payload
from any2xsampa.services.inference import InferenceRequest, InferenceService, RecognizerContractError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Universal dialect singing-to-phones frontend MVP")
    subparsers = parser.add_subparsers(dest="command", required=True)

    open_parser = subparsers.add_parser("infer-open", help="Infer phones from audio without lyrics.")
    open_parser.add_argument("--audio", required=True, help="Path to input audio.")
    open_parser.add_argument("--dialect", required=True, help="Dialect plugin name.")
    _add_recognizer_args(open_parser)

    lyric_parser = subparsers.add_parser("infer-with-lyrics", help="Infer phones with lyric assistance.")
    lyric_parser.add_argument("--audio", required=True, help="Path to input audio.")
    lyric_parser.add_argument("--dialect", required=True, help="Dialect plugin name.")
    lyric_parser.add_argument("--lyrics", required=True, help="Lyrics used as auxiliary input.")
    _add_recognizer_args(lyric_parser)

    from any2xsampa.local.cli import add_parsers
    add_parsers(subparsers)
    _add_candidate_parser(subparsers)
    return parser


def _add_candidate_parser(subparsers) -> None:
    parser = subparsers.add_parser("infer-candidate", help="Explicitly run the frozen research candidate on a WAV crop.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--start", type=float, default=0.0, help="Crop start in seconds in the original WAV.")
    parser.add_argument("--duration", type=float, default=8.0, help="Crop duration, 0.1 to 20 seconds.")
    parser.add_argument("--bundle", type=Path, help="Hash-bound isolated candidate runtime bundle.")
    parser.add_argument("--fixture", type=Path, help="Replay a matching existing posterior without CUDA.")
    parser.add_argument("--output-dir", type=Path, help="New directory under .state/runs for audio, IPA, X-SAMPA, JSON and HTML.")
    parser.add_argument("--benchmark-warm", type=int, choices=[0, 5], default=0, help=argparse.SUPPRESS)
    parser.add_argument("--audio-cache", choices=["on", "off"], default="on", help=argparse.SUPPRESS)


def _candidate_command(args: argparse.Namespace) -> dict:
    from datetime import datetime, timezone
    from any2xsampa.adapters.candidate_inference import ROOT, run_candidate
    if args.bundle is not None and args.fixture is not None:
        raise ValueError("Choose either a runtime bundle or a posterior fixture.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir or ROOT / ".state/runs" / ("candidate-use-" + stamp)
    result = run_candidate(audio=Path(args.audio), start=args.start, duration=args.duration,
                           output=output, bundle_path=args.bundle, fixture_path=args.fixture,
                           warm_repeats=args.benchmark_warm, audio_cache=args.audio_cache == "on")
    return {"status": "complete", "mode": result["mode"], "output_dir": str(output.resolve()),
            "result_path": str(output.resolve() / "result.json"), "page": str(output.resolve() / "index.html"),
            "ipa_sequence": result["ipa_sequence"], "default_model_changed": False}


def _add_recognizer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--recognizer-mode",
        default="fixture",
        choices=["fixture", "real"],
        help="Recognizer input source mode.",
    )
    parser.add_argument(
        "--recognizer-fixture",
        default=None,
        help="Path to posterior/logits/phones JSON fixture for fixture mode.",
    )
    parser.add_argument(
        "--recognizer-real-backend",
        default=None,
        help="Real backend for real mode, for example onnx-espeak or posterior-json. Diagnostic baselines are not real backends.",
    )
    parser.add_argument(
        "--recognizer-real-command",
        default=None,
        help="External command for real mode. Use {audio} and {output}; recognizers never receive lyrics.",
    )
    parser.add_argument(
        "--recognizer-real-output",
        default="real_recognizer_output.json",
        help="JSON output path produced by the real recognizer command.",
    )
    parser.add_argument(
        "--recognizer-model",
        default=None,
        help="Local recognizer bundle directory required by model-backed real recognizers.",
    )
    parser.add_argument(
        "--recognizer-device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for a model-backed real recognizer.",
    )
    parser.add_argument(
        "--disable-recognizer-fallback",
        action="store_true",
        help="Reject any non-debug fallback when a real recognizer fails.",
    )
    parser.add_argument(
        "--export-format",
        default="xsampa",
        choices=["xsampa", "vocaloid_like", "openutau_like"],
        help="Export target format. xsampa remains the standard export layer.",
    )
    parser.add_argument(
        "--engine-profile",
        default=None,
        help="Optional engine-specific profile, for example generic_openutau, cvvc_like, or plain_cv_like.",
    )
    parser.add_argument(
        "--engine-inventory",
        default=None,
        help="Optional path to a bank alias inventory source for openutau_like validation, such as JSON, oto.ini, or an OpenUtau singer root/character.yaml.",
    )
    parser.add_argument(
        "--engine-subbank",
        default=None,
        help="Optional preferred subbank label for openutau_like alias ranking.",
    )
    parser.add_argument(
        "--engine-color",
        default=None,
        help="Optional preferred color label for openutau_like alias ranking.",
    )
    parser.add_argument(
        "--engine-style",
        default=None,
        help="Optional preferred style label for openutau_like alias ranking.",
    )
    parser.add_argument(
        "--engine-tag",
        action="append",
        default=None,
        help="Optional preferred alias tag for openutau_like alias ranking. Can be repeated.",
    )
    parser.add_argument(
        "--note-context",
        default=None,
        help="Optional path to note/score context input for exporter-side note-aware ranking.",
    )
    parser.add_argument(
        "--note-context-format",
        default="auto",
        choices=["auto", "json", "ustx", "ust"],
        help="Optional explicit note/score context format. Defaults to auto-detect from extension/content.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the same presented JSON payload that is printed to stdout.",
    )


def _write_output(payload: dict[str, object], output_path: str | None) -> None:
    if not output_path:
        return
    path = Path(output_path)
    write_text(path, dump_json(payload, ensure_ascii=False, indent=2), atomic=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in {"recognize", "replay", "export", "export-vsqx", "score", "import-score", "models", "doctor"}:
            from any2xsampa.local.cli import run
            result = run(args)
        elif args.command == "infer-candidate":
            result = _candidate_command(args)
        else:
            request = InferenceRequest.from_args_namespace(
                args,
                mode="infer-open" if args.command == "infer-open" else "infer-with-lyrics",
            )
            result = augment_pipeline_payload(InferenceService().run(request))
            _write_output(result, args.output)
    except (RecognizerContractError, FileNotFoundError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(json.dumps(_cli_error_payload(exc), ensure_ascii=True), file=sys.stderr)
        return 1
    # Console JSON stays ASCII-safe on Windows legacy code pages. The optional
    # output file remains UTF-8 with literal Unicode, and both decode to the
    # same JSON contract.
    print(dump_json(result, ensure_ascii=True, indent=2))
    return 0


def _cli_error_payload(exc: Exception) -> dict[str, object]:
    if isinstance(exc, RecognizerContractError):
        return build_transport_exception_payload(
            exc,
            transport="cli",
            exit_code=1,
            details={"recognizer_error": exc.error_kind},
        ) | {
            "error_code": f"cli.{exc.error_code_suffix}",
            "error_kind": exc.error_kind,
        }
    return build_transport_exception_payload(
        exc,
        transport="cli",
        exit_code=1,
    )


def _request_mapping_from_args(args: argparse.Namespace) -> dict[str, object]:
    return InferenceRequest.from_args_namespace(
        args,
        mode="infer-open" if args.command == "infer-open" else "infer-with-lyrics",
    ).to_mapping()


if __name__ == "__main__":
    raise SystemExit(main())
