"""An opt-in, provenance-bound restoration of a documented compound convention."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from any2xsampa.training.limited_data_artifacts import read_json, sha256_file, verify_file
from any2xsampa.training.limited_data_reference import load_mapping, load_vocabulary, project_tokens


OVERLAY_PATH = Path(__file__).resolve().parents[1] / "config/v4c_original_reference_overlay_v1.json"


def documented_compound_mapping(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    if overlay.get("schema_version") != "v4c_original_reference_overlay_v1" or overlay.get("label_scope") != "derived_engine_reference":
        raise ValueError("unsupported compound mapping overlay")
    rules = overlay["rules"]
    if set(rules) != {"ua_n"} or rules["ua_n"]["source_tokens"] != ["ua", "n"]:
        raise ValueError("only the documented ua+n compound is authorized by this overlay")
    if "ua_n" in base["tokens"]:
        raise ValueError("compound overlay cannot overwrite an existing base entry")
    derived = [p for token in rules["ua_n"]["source_tokens"] for p in base["tokens"][token]["ipa"]]
    if derived != rules["ua_n"]["ipa"]:
        raise ValueError("compound IPA differs from its intact source constituents")
    result = deepcopy(dict(base))
    result["tokens"]["ua_n"] = {"ipa": derived}
    result["unresolved_tokens"] = [t for t in result["unresolved_tokens"] if t != "ua_n"]
    return result


def apply_original_mapping_overlay(reference: Mapping[str, Any], evidence_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    overlay = read_json(OVERLAY_PATH)
    source = read_json(evidence_path)
    if source.get("schema_version") != "vocaloid_manual_symbol_evidence_v1" or source.get("pages_visually_checked") != [168, 169]:
        raise ValueError("compound mapping requires visually checked source-table evidence")
    if source["source_sha256"] != overlay["manual_sha256"]:
        raise ValueError("manual differs from the pinned evidence")
    verify_file(Path(source["source_path"]), source["source_sha256"])
    for page in source["page_images"]:
        verify_file(Path(page["path"]), page["sha256"])
    mapping_path = Path(reference["bindings"]["mapping"]["path"])
    verify_file(mapping_path, overlay["base_mapping_sha256"])
    legacy_path = mapping_path.parent / overlay["legacy_convention_source"]
    verify_file(legacy_path, overlay["legacy_convention_sha256"])
    legacy = read_json(legacy_path)
    if "ua + n" not in legacy["mapping_basis"]["new_entries"]["ua_n"]:
        raise ValueError("historical compound convention is missing")
    mapping = documented_compound_mapping(load_mapping(mapping_path), overlay)
    vocabulary_path = Path(reference["bindings"]["vocabulary"]["path"])
    verify_file(vocabulary_path, reference["bindings"]["vocabulary"]["sha256"])
    vocabulary = load_vocabulary(vocabulary_path)
    view = deepcopy(dict(reference))
    changes = []
    for song in view["songs"]:
        if song["split"] == "test":
            continue
        for note in song["notes"]:
            if "ua_n" not in note["raw_v4c_tokens"]:
                continue
            before = note["mapping_status"]
            note.update(project_tokens(note["raw_v4c_tokens"], mapping, vocabulary))
            changes.append({"song_id": song["song_id"], "note_id": note["note_id"], "previous_mapping_status": before,
                            "mapping_status": note["mapping_status"], "raw_human_annotation_modified": False})
    binding = {"path": str(OVERLAY_PATH), "sha256": sha256_file(OVERLAY_PATH),
               "manual_evidence_path": str(evidence_path.resolve()), "manual_evidence_sha256": sha256_file(evidence_path),
               "base_mapping_sha256": overlay["base_mapping_sha256"], "changed_projection_notes": changes,
               "label_scope": "derived_engine_reference", "independent_acoustic_ipa_verified": False,
               "historical_reference_modified": False, "test_note_projections_changed": False}
    return view, binding
