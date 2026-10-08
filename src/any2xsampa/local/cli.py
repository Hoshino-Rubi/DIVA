"""CLI surface for the shared local task engine; recognize is always real."""
from pathlib import Path
import json
import sys
import threading
from .common import Cancelled
from .engine import LocalEngine

def add_parsers(sub):
    p = sub.add_parser("recognize", help="Real offline inference; never falls back to fixtures.")
    p.add_argument("audio")
    p.add_argument("--model", required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--duration", type=float, default=8)
    p.add_argument("--whole-file", action="store_true", help="Recognize the entire WAV with overlapping bounded model windows.")
    p.add_argument("--output-dir")
    p = sub.add_parser("replay", help="Explicit fixture/posterior/result replay; zero model forwards.")
    p.add_argument("fixture")
    p.add_argument("--audio")
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--duration", type=float, default=8)
    p.add_argument("--output-dir")
    p = sub.add_parser("export", help="Export machine IPA or explicit human review draft.")
    p.add_argument("result_path")
    p.add_argument("--format", choices=["ipa", "xsampa"], default="xsampa")
    p.add_argument("--output", required=True)
    p.add_argument("--use-draft", action="store_true")
    p = sub.add_parser("export-vsqx", help="Generate VSQX from saved notes/score; optional legacy template mode. Zero inference.")
    p.add_argument("result_path", help="score.json, note prediction.json, or phone result.json with --notes")
    p.add_argument("template_path", nargs="?", help="Optional legacy template; preserves its note geometry")
    p.add_argument("--notes", dest="notes_path", help="Saved GAME prediction matching the phone result audio")
    p.add_argument("--phones", dest="phones_path", help="Optional local phone result when exporting a note prediction")
    p.add_argument("--tempo", type=float, help="Grid BPM for note predictions; never changes source seconds")
    p.add_argument("--output", required=True)
    p.add_argument("--use-draft", action="store_true")
    p.add_argument("--apply-phone-suggestions", action="store_true", help="Opt in to an unconfirmed post-hoc phone/time proposal; original note geometry is preserved.")
    p = sub.add_parser("score", help="Generate VSQX with GAME note inference or saved notes; reuse recognized IPA.")
    p.add_argument("result_path")
    p.add_argument("--backend", choices=["game", "pyin"], default="game")
    p.add_argument("--snapshot", help="Local verified GAME snapshot directory (no download)")
    p.add_argument("--notes", dest="notes_path", help="Reuse saved GAME notes; zero model forwards")
    p.add_argument("--tempo", type=float, default=120)
    p.add_argument("--output-dir")
    p = sub.add_parser("import-score", help="Preserve an existing VSQX without re-estimating pitch or timing.")
    p.add_argument("result_path")
    p.add_argument("template_path")
    p.add_argument("--output-dir")
    p = sub.add_parser("models", help="Verify local runtime packages.")
    p.add_subparsers(dest="models_command", required=True).add_parser("list")
    sub.add_parser("doctor", help="Check configuration without loading any model.")

def run(args):
    cancel = threading.Event()
    engine = LocalEngine()
    params = {k:v for k,v in vars(args).items() if k not in {"command", "models_command"} and v is not None}
    def progress(data):
        print(json.dumps({"event": "progress", **data}, ensure_ascii=True), file=sys.stderr, flush=True)
    # The main thread must still handle Ctrl-C while the job is waiting on native code.
    import signal
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda *_: cancel.set())
    try:
        action = args.command
        if action in {"score", "import-score"}:
            return engine.execute("build_score" if action == "score" else "import_score", params, cancel, progress)
        if action == "export-vsqx":
            params["output_path"] = params.pop("output")
            return engine.execute("export_vsqx", params, cancel, progress)
        if params.pop("whole_file", False):
            if params.get("start", 0) != 0 or params.get("duration", 8) != 8:
                raise ValueError("--whole-file cannot be combined with a clip interval")
            params.pop("start", None); params.pop("duration", None)
            action = "recognize-whole"
        return engine.execute(action, params, cancel, progress)
    finally:
        engine.close()
        signal.signal(signal.SIGINT, previous)
