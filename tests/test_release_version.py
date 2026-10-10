"""Fail packaging if the manually reviewed release identity drifts between layers."""
from pathlib import Path
import json
import re
import tomllib
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def test_stable_release_identity_is_consistent():
    release = json.loads((ROOT / "release-version.json").read_text(encoding="utf-8"))
    tag = release["tag"]
    assert re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag)
    assert release["prerelease"] is False
    version = release["python_version"]
    assert tag == "v" + version
    props = ET.parse(ROOT / "Directory.Build.props").getroot()
    values = {e.tag: e.text for e in props.iter() if e.text and e.text.strip()}
    assert values["Version"] == version
    assert values["InformationalVersion"] == tag
    assert values["AssemblyVersion"] == release["assembly_version"]
    assert tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"] == version
    source = (ROOT / "src/any2xsampa/__init__.py").read_text(encoding="utf-8")
    assert f'__version__ = "{version}"' in source
    diagnostic = (ROOT / "apps/Any2Xsampa.Desktop/Services/DiagnosticBundle.cs").read_text(encoding="utf-8")
    assert "AssemblyInformationalVersionAttribute" in diagnostic


def test_single_source_generator_detects_drift(tmp_path):
    import importlib.util
    import shutil
    spec = importlib.util.spec_from_file_location("sync_version", ROOT / "tools/sync_release_version.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for name in ["release-version.json", "Directory.Build.props", "pyproject.toml", "src/any2xsampa/__init__.py"]:
        target = tmp_path / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    file = tmp_path / "src/any2xsampa/__init__.py"
    file.write_text(file.read_text().replace("1.0.1", "0.0.1"))
    import pytest
    with pytest.raises(ValueError, match="mirrors"):
        module.sync(tmp_path, check=True)
    assert module.sync(tmp_path, check=False) == ["src/any2xsampa/__init__.py"]
    assert module.sync(tmp_path, check=True) == []
