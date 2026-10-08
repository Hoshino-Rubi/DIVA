from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from any2xsampa.core.io import write_json  # noqa: E402
from any2xsampa.vocaloid.vsqx_alignment import build_vsqx_phone_timing_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare audio-only PhoneSegment timing with a VSQX note timeline after recognition."
    )
    parser.add_argument("--phone-segments", required=True, help="Path to infer-open PhoneSegment JSON.")
    parser.add_argument("--vsqx", required=True, help="Path to the human-authored VSQX reference.")
    parser.add_argument("--output", default=None, help="Optional JSON report output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_vsqx_phone_timing_report(args.phone_segments, args.vsqx)
    except (ETParseError, FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as exc:
        _print_utf8(
            json.dumps(
                {
                    "error_schema_version": "vsqx_phone_timing_comparison_cli_error_v1",
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
            ),
            stream=sys.stderr,
        )
        return 1

    if args.output:
        write_json(Path(args.output), report, append_newline=True, atomic=True)
    _print_utf8(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _print_utf8(text: str, *, stream: object = sys.stdout) -> None:
    buffer = getattr(stream, "buffer", None)
    if buffer is not None:
        buffer.write((text + "\n").encode("utf-8"))
        buffer.flush()
        return
    print(text, file=stream)


try:
    from xml.etree.ElementTree import ParseError as ETParseError
except ImportError:  # pragma: no cover
    ETParseError = ValueError


if __name__ == "__main__":
    raise SystemExit(main())
