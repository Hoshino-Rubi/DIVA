"""Cross-directory completion authorization; prior failed starts still count."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file
from any2xsampa.training.event_head_protocol import load_protocol


def load_completion(run: Path, review_name: str) -> tuple:
    auth = read_json(run / "AUTHORIZATION.json")
    if Path(auth["completion_run"]).resolve() != run.resolve():
        raise ValueError("authorization cannot be copied to reset the budget")
    if (auth["global_quality_start_cap"], auth["prior_quality_starts"], auth["new_quality_start_cap"]) != (11, 9, 2):
        raise ValueError("unapproved budget")
    if auth["allowed_jobs"] != ["E--promise", "E--queen-road-east"] or auth["automatic_retry"] is not False:
        raise ValueError("unapproved jobs or retry policy")
    source = Path(auth["source_run"])
    for name, digest in auth["source_bindings"].items():
        verify_file(Path(name), digest)
    actual = {str(p): sha256_file(p) for p in (source / "commands").glob("*.command.json")}
    if actual != auth["prior_quality_command_bindings"]:
        raise ValueError("historical launch ledger changed")
    review = read_json(run / review_name)
    if not review["approved_for_launch"] or review["authorization_sha256"] != sha256_file(run / "AUTHORIZATION.json"):
        raise ValueError("unapproved or drifted code review")
    for name, digest in review["code_bindings"].items():
        verify_file(Path(name), digest)
    v = review["verification"]
    verify_file(Path(v["path"]), v["sha256"])
    if not read_json(Path(v["path"]))["passed"]:
        raise ValueError("verification did not pass")
    freeze, spec, index = load_protocol(source, "CODE_REVIEW_training_v2.json")
    return auth, source, freeze, spec, index


def check_attempt(run: Path, held: str, auth: dict, *, worker: bool = False) -> int:
    key = "E--" + held
    if key not in auth["allowed_jobs"]:
        raise ValueError("attempt is outside the approved whitelist")
    commands = list((run / "commands").glob("*.command.json"))
    keys = {p.name.removesuffix(".command.json") for p in commands}
    if not keys.issubset(set(auth["allowed_jobs"])):
        raise ValueError("unaccounted completion launch")
    if len(commands) > 2 or (not worker and len(commands) >= 2):
        raise ValueError("global eleven-start budget exhausted")
    if (run / "screen/E" / held / "START.json").exists():
        raise ValueError("job already started; no automatic retry")
    command = run / "commands" / (key + ".command.json")
    if command.exists() != worker:
        raise ValueError("worker requires exactly one pre-recorded command")
    return auth["prior_quality_starts"] + len(commands) + (0 if worker else 1)
