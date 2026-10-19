import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

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


def test_numpy_runtime_helper_is_not_pruned_as_a_test():
    module=builder()
    assert module.kept(Path("Lib/site-packages/numpy/_core/tests/_natype.py"))
    assert module.kept(Path("numpy/_core/tests/__init__.py"))
    assert not module.kept(Path("numpy/_core/tests/test_numeric.py"))
    assert not module.kept(Path("other/tests/__init__.py"))


def test_runtime_support_overlay_requires_exact_source_binding(tmp_path):
    file=manifest(tmp_path);value=json.loads(file.read_text())
    helper=tmp_path/"helper.py";helper.write_text("pd_NA = None")
    value["runtime_support_files"]=[dict(role="phonetic",module_path="numpy/_core/tests/_natype.py",
        source="helper.py",sha256=digest(helper))]
    file.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="not hash-bound"):builder().load_inputs(file)
    value["bindings"].append(dict(path="helper.py",sha256=digest(helper)))
    file.write_text(json.dumps(value))
    assert builder().load_inputs(file)["runtime_support_files"][0]["resolved_source"]==helper
    value["runtime_support_files"][0]["module_path"]="../injection.py"
    file.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="Unexpected runtime support"):builder().load_inputs(file)


def test_runtime_support_updates_array_dependency_inventory(tmp_path):
    module=builder();source=tmp_path/"support.py";source.write_text("# fixture")
    module.INPUTS={"runtime_support_files":[dict(role="core",module_path="safetensors/__init__.py",resolved_source=source)]}
    out=tmp_path/"app";runtime=out/"runtime/core";runtime.mkdir(parents=True)
    lock=runtime/"DEPENDENCIES.json";lock.write_text('[{"name":"numpy","version":"2.4.6"}]')
    module.copy_runtime_support(out)
    module.copy_runtime_support(out)
    rows=json.loads(lock.read_text())
    assert rows.count(dict(name="safetensors",version="0.8.0"))==1
    assert (runtime/"Lib/site-packages/safetensors/__init__.py").read_text()=="# fixture"
    assert not module.support_module_allowed("safetensors/../../escape.py")


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


def test_stable_tag_is_accepted_without_accepting_arbitrary_name(tmp_path):
    (tmp_path/"app.exe").write_bytes(b"release")
    assert asset_tool().finalize(tmp_path,["app.exe"],"v1.0.1")["tag"]=="v1.0.1"
    other=tmp_path/"other";other.mkdir();(other/"app.exe").write_bytes(b"release")
    with pytest.raises(ValueError,match="tag"):
        asset_tool().finalize(other,["app.exe"],"v1.0.1/evil")


def test_split_manifest_binds_stable_tag_and_exact_parts(tmp_path):
    spec=importlib.util.spec_from_file_location("release_splitter",ROOT/"tools/split_portable_release.py")
    tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
    archive=tmp_path/"app.zip"
    with zipfile.ZipFile(archive,"x") as zip_file:
        zip_file.writestr("Application/file.txt",b"fixture")
    output=tmp_path/"assets";output.mkdir()
    result=tool.split(archive,output,digest(archive),"v1.0.1")
    assert result["schema_version"]=="portable_split_release_v2"
    assert result["release_tag"]=="v1.0.1"
    assert result["download_base_url"]=="https://github.com/Hoshino-Rubi/DIVA/releases/download/v1.0.1/"
    assert (output/result["parts"][0]["name"]).read_bytes()==archive.read_bytes()
    other=tmp_path/"invalid";other.mkdir()
    with pytest.raises(ValueError,match="stable release tag"):
        tool.split(archive,other,digest(archive),"v1.0.1/extra")
    assert not list(other.iterdir())
