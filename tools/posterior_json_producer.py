from __future__ import annotations

import argparse
import json
from pathlib import Path

from any2xsampa.adapters.recognizer_producer_runtime import (
    ScoreAwarePosteriorJSONProducer,
    available_producer_backend_names,
    build_onnx_espeak_runtime_diagnostics_snapshot,
    create_posterior_json_producer,
    describe_onnx_espeak_runtime,
    write_onnx_espeak_runtime_diagnostics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce posterior-json-compatible recognizer output from audio.")
    parser.add_argument("--audio", help="Path to PCM WAV audio input.")
    parser.add_argument("--output", help="Path to write posterior/logits JSON.")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=available_producer_backend_names(),
        help=(
            "Producer backend. 'auto' resolves to ONNX/eSpeak and fails clearly if unavailable; "
            "prototype-logit and heuristic-dsp are explicit diagnostic baselines."
        ),
    )
    parser.add_argument(
        "--diagnostic-backend",
        default=None,
        choices=["prototype-logit", "heuristic-dsp"],
        help="Explicit diagnostic baseline producer. This is an alias kept separate from the trained backend path.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional ONNX/eSpeak model bundle or model path for the real backend.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help=(
            "Execution device for the real ONNX/eSpeak backend. 'cuda' fails clearly "
            "when CUDAExecutionProvider is unavailable; 'auto' may fall back to CPU."
        ),
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Optional ONNX Runtime intra-op CPU thread limit; useful for long CUDA runs.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=None,
        help="Optional bounded waveform chunk size for long real-audio runs.",
    )
    parser.add_argument(
        "--chunk-overlap-seconds",
        type=float,
        default=0.5,
        help="Overlap retained at chunk boundaries when --chunk-seconds is used.",
    )
    parser.add_argument(
        "--note-context",
        default=None,
        help=(
            "Optional note-context path recorded for downstream pipeline alignment metadata; "
            "it is never passed into the acoustic decoder."
        ),
    )
    parser.add_argument(
        "--note-context-format",
        default="auto",
        choices=["auto", "json", "ustx", "ust"],
        help="Optional explicit note-context format override.",
    )
    parser.add_argument("--lyrics", default="", help="Optional lyrics string for producer metadata.")
    parser.add_argument(
        "--diagnose-runtime",
        action="store_true",
        help="Print structured onnx-espeak runtime readiness diagnostics, including runtime_path/env discovery, and exit.",
    )
    parser.add_argument(
        "--prefetch-runtime",
        action="store_true",
        help="Warm the selected onnx-espeak runtime cache, then print the same structured diagnostics and exit.",
    )
    parser.add_argument(
        "--diagnostics-output",
        default=None,
        help="Optional path to write an archivable runtime diagnostics JSON snapshot when using --diagnose-runtime or --prefetch-runtime.",
    )
    args = parser.parse_args()

    if args.diagnose_runtime or args.prefetch_runtime:
        diagnostics = describe_onnx_espeak_runtime(prefetch=args.prefetch_runtime)
        if args.diagnostics_output:
            archived = write_onnx_espeak_runtime_diagnostics(
                Path(args.diagnostics_output),
                runtime_diagnostics=diagnostics,
            )
            print(json.dumps(archived, ensure_ascii=True, indent=2))
            return
        print(json.dumps(build_onnx_espeak_runtime_diagnostics_snapshot(diagnostics), ensure_ascii=True, indent=2))
        return

    if not args.audio or not args.output:
        raise SystemExit("--audio and --output are required unless --diagnose-runtime or --prefetch-runtime is used.")

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    output_path = Path(args.output)
    model_path = Path(args.model) if args.model else None
    if model_path is not None and not model_path.exists():
        raise SystemExit(f"Model path not found: {model_path}")
    note_context_path = Path(args.note_context) if args.note_context else None
    if note_context_path is not None and not note_context_path.exists():
        raise SystemExit(f"Note context file not found: {note_context_path}")

    if args.diagnostic_backend is not None and args.backend != "auto":
        raise SystemExit("--backend and --diagnostic-backend cannot be used together.")
    selected_backend = args.diagnostic_backend or args.backend
    producer = create_posterior_json_producer(
        backend_name=selected_backend,
        model_path=model_path,
        device=args.device,
        cpu_thread_limit=args.cpu_threads,
        chunk_duration_sec=args.chunk_seconds,
        chunk_overlap_sec=args.chunk_overlap_seconds,
    )
    producer.produce(
        audio_path,
        output_path=output_path,
        note_context_path=note_context_path,
        note_context_format=args.note_context_format,
        lyrics=args.lyrics or None,
    )


if __name__ == "__main__":
    main()
