"""Stage existing candidate-4 assets with hard links; never download or load models.

Zip the CONTENTS of --output so extraction produces a selectable bundle root.
Public staging excludes author code. --include-author-code is local validation only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any


ASSET_LOCK = Path(__file__).resolve().parents[1] / "src/any2xsampa/config/cantonese4_assets.json"
RECOGNIZER_RUN = Path(".state/runs/cantonese-parallel-20261113")
SEPARATOR_RUN = Path(".state/runs/quality-first8h-20260907T010738")
SOURCE_ROOTS = {
    "small": RECOGNIZER_RUN / "model-small-fixed",
    "base": RECOGNIZER_RUN / "model-recovery1",
    "separator": SEPARATOR_RUN / "assets/separator",
}
DOWNLOAD_RECORDS = {
    "small": "SMALL_MODEL_DOWNLOADED_FILES.json",
    "base": "MODEL_DOWNLOAD_RECOVERY1_FILES.json",
}
RECOGNIZER_FILES = {
    "README.md", "added_tokens.json", "config.json", "generation_config.json",
    "merges.txt", "normalizer.json", "preprocessor_config.json", "pytorch_model.bin",
    "special_tokens_map.json", "tokenizer_config.json", "vocab.json",
}
AUTHOR_FILES = {
    "models/mel_band_roformer/__init__.py",
    "models/mel_band_roformer/attend.py",
    "models/mel_band_roformer/mel_band_roformer.py",
    "configs/config_vocals_mel_band_roformer.yaml",
}
NOTICE = """DIVA candidate-4 optional model bundle

Select this extracted directory as the model root. It contains small/, base/,
separator/ and cantonese4_assets.json; Python runtime dependencies are separate.

Small/base: AlienKevin Whisper Jyutping without tones, Apache-2.0 as declared
by the pinned model cards. Their README.md files and asset URLs retain attribution.
Separator weights: KimberleyJSN/melbandroformer, MIT as declared by saved HF
metadata. Author implementation: KimberleyJensen/Mel-Band-Roformer-Vocal-Model
has no confirmed code license and is EXCLUDED from the default public bundle.
The asset lock retains pinned URLs and hashes for its four required files.
Obtain them through an explicit user Prepare action or supply your own code.
This notice is not a license grant or a replacement for applicable license texts.

No dataset, example audio, reference lyrics, Python runtime or inference result
is included. These models produce toneless Jyutping; mapped IPA has no measured
phone timing or recovered tone. Default phones/VSQX are a separate pipeline.

STAGING DIRECTORY: payload files are hard links sharing source storage. Do not
edit or chmod them. Creating a ZIP reads them; extraction produces independent
files. Identities reuse existing download/code locks and source file sizes;
this tool does not rehash weights, run models, download, zip or split assets.
"""
LOCAL_CODE_NOTICE = """
LOCAL VALIDATION COPY ONLY: --include-author-code was explicitly selected.
This copy additionally contains three upstream Python files and one YAML.
Their code license is unconfirmed. Do not publicly redistribute this copy.
"""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def allowed_paths() -> set[str]:
    return ({f"{role}/{name}" for role in DOWNLOAD_RECORDS for name in RECOGNIZER_FILES}
            | {"separator/MelBandRoformer.ckpt"}
            | {f"separator/code/{name}" for name in AUTHOR_FILES})


def validate_lock(lock: dict[str, Any]) -> None:
    if lock.get("schema_version") != "cantonese4_assets_v1":
        raise ValueError("Unsupported candidate-4 asset schema")
    if set(lock.get("models", {})) != set(SOURCE_ROOTS):
        raise ValueError("Exactly small/base/separator model identities required")
    for model in lock["models"].values():
        if (set(model) != {"id", "revision"} or not isinstance(model["id"], str)
                or not re.fullmatch(r"[0-9a-f]{40}", model["revision"])):
            raise ValueError("Invalid pinned model identity")
    rows = lock.get("files", [])
    if len(rows) != len(allowed_paths()) or {r["path"] for r in rows} != allowed_paths():
        raise ValueError("Asset members must exactly match the 27 runtime assets")
    for row in rows:
        if (set(row) != {"path", "bytes", "sha256", "url"}
                or type(row["bytes"]) is not int or row["bytes"] <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                or not row["url"].startswith("https://")):
            raise ValueError("Invalid asset identity")


def source_evidence(workspace: Path, lock: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Read only existing small identity JSONs, never model payloads."""
    result: dict[str, dict[str, Any]] = {}
    for role, filename in DOWNLOAD_RECORDS.items():
        records = read_json(workspace / RECOGNIZER_RUN / filename)
        if len(records) != 11 or {r["name"] for r in records} != RECOGNIZER_FILES:
            raise ValueError("Incomplete recognizer download evidence: " + role)
        model = lock["models"][role]
        prefix = f"https://huggingface.co/{model['id']}/"
        for row in records:
            method = "resolve" if row["name"] == "pytorch_model.bin" else "raw"
            url = f"{prefix}{method}/{model['revision']}/{row['name']}"
            if row["status_code"] != 200 or row["url"] != url:
                raise ValueError("Recognizer identity differs from download evidence")
            result[f"{role}/{row['name']}"] = {
                "bytes": row["bytes"], "sha256": row["sha256"], "url": url}
    assets = read_json(workspace / SEPARATOR_RUN / "assets/assets-lock.json")
    model = assets["models"]["separator"]["model"]
    if {key: model[key] for key in ("id", "revision")} != lock["models"]["separator"]:
        raise ValueError("Separator model identity differs from existing asset lock")
    result["separator/MelBandRoformer.ckpt"] = {
        "bytes": model["weight_bytes"], "sha256": model["weight_sha256"],
        "url": f"https://huggingface.co/{model['id']}/resolve/{model['revision']}/MelBandRoformer.ckpt",
    }
    code = read_json(workspace / SOURCE_ROOTS["separator"] / "code-lock.json")
    if code["revision"] != model["code_revision"] or code["repository"] != model["code_repository"]:
        raise ValueError("Author code identity differs from separator asset lock")
    for row in code["files"]:
        if row["file"] in AUTHOR_FILES:
            result[f"separator/code/{row['file']}"] = {
                "bytes": row["bytes"], "sha256": row["sha256"],
                "url": f"https://raw.githubusercontent.com/{code['repository']}/{code['revision']}/{row['file']}",
            }
    return result


def plan_sources(workspace: Path, lock: dict[str, Any], *,
                 include_author_code: bool = False) -> list[tuple[Path, str]]:
    validate_lock(lock)
    evidence = source_evidence(workspace, lock)
    if set(evidence) != allowed_paths():
        raise ValueError("Incomplete local identity evidence")
    plan: list[tuple[Path, str]] = []
    for row in lock["files"]:
        name = row["path"]
        if {key: row[key] for key in ("bytes", "sha256", "url")} != evidence[name]:
            raise ValueError("Asset lock differs from local identity evidence: " + name)
        if name.startswith("separator/code/") and not include_author_code:
            continue
        role, *parts = PurePosixPath(name).parts
        root = (workspace / SOURCE_ROOTS[role]).resolve(strict=True)
        source = (root.joinpath(*parts)).resolve(strict=True)
        if not root.is_relative_to(workspace) or not source.is_relative_to(root):
            raise ValueError("Source escapes its fixed workspace directory: " + name)
        if not source.is_file() or source.stat().st_size != row["bytes"]:
            raise ValueError("Source missing or size differs: " + name)
        plan.append((source, name))
    return plan


def stage(workspace: Path, output: Path, *, include_author_code: bool = False) -> dict[str, Any]:
    workspace = workspace.resolve(strict=True)
    output = output.absolute()
    if os.path.lexists(output):
        raise FileExistsError("Output already exists; never overwrite or resume: " + str(output))
    output = output.resolve()
    for relative in (RECOGNIZER_RUN, SEPARATOR_RUN):
        if output.is_relative_to((workspace / relative).resolve()):
            raise ValueError("Output must not be inside a source run")
    lock_bytes = ASSET_LOCK.read_bytes()
    lock = json.loads(lock_bytes)
    plan = plan_sources(workspace, lock, include_author_code=include_author_code)
    included = {name for _, name in plan}
    payload_bytes = sum(row["bytes"] for row in lock["files"] if row["path"] in included)
    bundle = {
        "schema_version": "cantonese4_bundle_v1", "model_root": ".",
        "assets_lock": "cantonese4_assets.json", "included_files": sorted(included),
        "payload_bytes": payload_bytes,
        "author_code": {
            "included": include_author_code,
            "usage": "local_validation_only" if include_author_code else "user_prepare_or_supplied",
            "required_files": sorted(f"separator/code/{name}" for name in AUTHOR_FILES),
        },
    }
    ancestor = output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if any(source.stat().st_dev != ancestor.stat().st_dev for source, _ in plan):
        raise ValueError("Hard-link staging requires output on the same volume as every source")
    output.mkdir(parents=True, exist_ok=False)
    # On failure retain partial output for inspection; never retry or delete it.
    with (output / "cantonese4_assets.json").open("xb") as stream:
        stream.write(lock_bytes)
    with (output / "NOTICE.txt").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(NOTICE)
        if include_author_code:
            stream.write(LOCAL_CODE_NOTICE)
    with (output / "bundle.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(bundle, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    for source, name in plan:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)
    return {"output": str(output), "linked_files": len(plan),
            "payload_bytes": payload_bytes,
            "identity": "reused local JSON evidence plus file sizes; payloads not rehashed",
            "author_code_included": include_author_code,
            "distribution": "local_validation_only" if include_author_code else "public_model_payload"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-author-code", action="store_true",
                        help="Include unlicensed author code in a LOCAL VALIDATION COPY ONLY")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(stage(args.source_workspace, args.output,
                               include_author_code=args.include_author_code), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Staging failed: {exc}\nAny partial output is retained; no automatic retry.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
