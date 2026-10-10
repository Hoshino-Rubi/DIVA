import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).parents[1]
def builder():
    spec=importlib.util.spec_from_file_location("release_builder",ROOT/"tools/build_portable_release.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def manifest(tmp_path):
    bound=[]
    for name in ("px/model/snapshot-lock.json","game/model/snapshot-lock.json","candidate.json"):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("{}")
        bound.append(dict(path=name,sha256=digest(p)))
    runtimes={}
    for role in ("core","phonetic"):
        runtime=tmp_path/role;runtime.mkdir()
        p=runtime/"python.exe";p.write_bytes(b"fixture")
        index=tmp_path/(role+".json")
        index.write_text(json.dumps(dict(schema_version="portable_runtime_inventory_v1",
            files=[dict(path="python.exe",sha256=digest(p))])))
        bound.append(dict(path=index.name,sha256=digest(index)))
        runtimes[role]=dict(directory=role,inventory=index.name)
    file=tmp_path/"inputs.json"
    file.write_text(json.dumps(dict(schema_version="portable_build_inputs_v1",artifact_root=".",
        phoneticxeus_root="px",game_root="game",candidate_freeze="candidate.json",runtimes=runtimes,bindings=bound)))
    return file

def test_explicit_inputs_accept_relocated_source(tmp_path):
    module=builder()
    assert module.load_inputs(manifest(tmp_path))["runtimes"]["core"]["resolved_directory"]==tmp_path/"core"

def test_runtime_drift_invalidates_inputs(tmp_path):
    file=manifest(tmp_path);(tmp_path/"core/python.exe").write_bytes(b"changed")
    with pytest.raises(ValueError,match="Runtime input changed"):builder().load_inputs(file)

def test_added_runtime_code_invalidates_inputs(tmp_path):
    file=manifest(tmp_path);(tmp_path/"core/injected.py").write_text("bad")
    with pytest.raises(ValueError,match="inventory changed"):builder().load_inputs(file)

def test_unbound_or_changed_candidate_is_rejected(tmp_path):
    file=manifest(tmp_path);(tmp_path/"candidate.json").write_text('{"x":1}')
    with pytest.raises(ValueError,match="binding changed"):builder().load_inputs(file)

def test_missing_binding_is_not_filled_from_current_bytes(tmp_path):
    file=manifest(tmp_path);value=json.loads(file.read_text());value["bindings"]=value["bindings"][1:];file.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="Missing required"):builder().load_inputs(file)


def asset_tool():
    spec=importlib.util.spec_from_file_location("release_assets",ROOT/"tools/finalize_release_assets.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_checksums_exactly_match_upload_assets(tmp_path):
    (tmp_path/"app.zip.001").write_bytes(b"volume")
    (tmp_path/"README.md").write_text("docs")
    result=asset_tool().finalize(tmp_path,["app.zip.001","README.md"],"pre-1.0.1a")
    assert len(result["assets"])==2
    lines=(tmp_path/"SHA256SUMS.txt").read_text().splitlines()
    assert len(lines)==2 and "UPLOAD_TO_GITHUB" not in "\n".join(lines)
    with pytest.raises(ValueError):asset_tool().finalize(tmp_path,["app.zip.001","README.md"],"pre-1.0.1a")

def test_extra_or_missing_asset_stops_publication(tmp_path):
    (tmp_path/"app.exe").write_bytes(b"fixture")
    with pytest.raises(ValueError,match="mismatch"):
        asset_tool().finalize(tmp_path,["app.exe","not-uploaded.md"],"pre-1.0.1a")
    assert not (tmp_path/"SHA256SUMS.txt").exists()

def test_unsafe_asset_name_stops_publication(tmp_path):
    with pytest.raises(ValueError,match="simple"):
        asset_tool().finalize(tmp_path,["../secret"],"pre-1.0.1a")
