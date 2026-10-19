"""Display-brand regression checks; legacy runtime identities stay compatible."""
from __future__ import annotations

from pathlib import Path
import tomllib
from xml.etree import ElementTree as ET

from any2xsampa.cli import build_parser
from any2xsampa.local.common import VERSION
from any2xsampa.local.storage import data_home, model_cache, state_dir


ROOT = Path(__file__).resolve().parents[1]
TITLE = "DIVA · 本地音素与 VSQX 工作台"
FULL_NAME = "Dialect Intonation & Vocal Automation"
UI = "{http://schemas.microsoft.com/winfx/2006/xaml/presentation}"
PACKAGE = "{http://schemas.microsoft.com/appx/manifest/foundation/windows10}"
UAP = "{http://schemas.microsoft.com/appx/manifest/uap/windows10}"


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_readme_explains_new_brand_and_unchanged_published_downloads() -> None:
    for name in ("README.md", "README_CN.md"):
        text = read(name)
        assert text.startswith("# DIVA\n")
        assert FULL_NAME in text
        assert "docs/branding_migration.md" in text
        assert "Any2Xsampa-Setup-v1.0.1.exe" in text
        assert "DIVA-Setup-v1.0.2.exe" in text
        assert "https://github.com/Hoshino-Rubi/DIVA" in text
        assert "any2xsampa.cmd" in text
        assert "ANY2XSAMPA_DATA_HOME" in text


def test_cli_help_is_branded_without_renaming_entry_points() -> None:
    help_text = build_parser().format_help()
    assert "DIVA" in help_text and FULL_NAME in help_text
    for command in ("recognize", "score", "export-vsqx", "infer-open"):
        assert command in help_text
    project = tomllib.loads(read("pyproject.toml"))["project"]
    assert project["name"] == "any2xsampa"
    assert project["description"].startswith("DIVA")
    assert project["scripts"] == {
        "any2xsampa": "any2xsampa.cli:main",
        "any2xsampa-api": "any2xsampa.api.server:main",
        "any2xsampa-gui": "any2xsampa.gui.app:main",
        "any2xsampa-grpc": "any2xsampa.grpc.server:main",
    }


def test_desktop_titles_and_new_export_names_use_diva() -> None:
    window = ET.parse(ROOT / "apps/Any2Xsampa.Desktop/MainWindow.xaml").getroot()
    assert window.attrib["Title"] == TITLE
    assert next(window.iter(f"{UI}TitleBar")).attrib["Title"] == TITLE
    page = ET.parse(ROOT / "apps/Any2Xsampa.Desktop/MainPage.xaml").getroot()
    assert any(node.attrib.get("Text") == TITLE for node in page.iter(f"{UI}TextBlock"))
    assert f'Title = "{TITLE}";' in read("apps/Any2Xsampa.Desktop/MainWindow.xaml.cs")
    source = read("apps/Any2Xsampa.Desktop/MainPage.xaml.cs")
    assert '$"DIVA-{DateTime.Now:yyyyMMdd-HHmmssfff}.vsqx"' in source
    assert '$"DIVA-diagnostics-{DateTime.UtcNow:yyyyMMdd-HHmmssfff}.zip"' in source


def test_windows_metadata_preserves_package_and_assembly_identity() -> None:
    manifest = ET.parse(ROOT / "apps/Any2Xsampa.Desktop/Package.appxmanifest").getroot()
    identity = manifest.find(f"{PACKAGE}Identity")
    assert identity.attrib["Name"] == "E853BF93-9275-4659-8D77-B5AFC2B701BF"
    assert identity.attrib["Publisher"] == "CN=any2xsampa"
    assert manifest.find(f"{PACKAGE}Properties/{PACKAGE}DisplayName").text == "DIVA"
    visual = next(manifest.iter(f"{UAP}VisualElements"))
    assert visual.attrib["DisplayName"] == "DIVA"
    assert visual.attrib["Description"] == FULL_NAME
    for name in ("Desktop", "Setup"):
        project = ET.parse(ROOT / f"apps/Any2Xsampa.{name}/Any2Xsampa.{name}.csproj").getroot()
        assert project.find(".//Product").text == "DIVA"
        assert project.find(".//AssemblyTitle").text == ("DIVA Setup" if name == "Setup" else "DIVA")
        if name == "Setup":
            assert project.find(".//AssemblyName").text == "Any2Xsampa-Setup"
        else:
            assert project.find(".//RootNamespace").text == "Any2Xsampa_Desktop"
            assert project.find(".//AssemblyName") is None  # Keep the project-derived filename.


def test_setup_and_legacy_gui_display_names_keep_old_runtime_targets() -> None:
    setup = read("apps/Any2Xsampa.Setup/Program.cs")
    for caption in ("DIVA 安装失败", TITLE, "卸载 DIVA"):
        assert f'"{caption}"' in setup
    assert 'Text = "DIVA "' in setup
    assert '"Any2Xsampa.Desktop.exe"' in setup
    assert '"Any2Xsampa", "InstallerCache"' in setup
    gui = read("src/any2xsampa/gui/app.py")
    assert 'self.root.title("DIVA")' in gui
    assert gui.count('messagebox.showerror("DIVA job failed", message)') == 2
    assert 'Path(appdata) / "any2xsampa" / "gui_recent_runs.json"' in gui


def test_data_locations_and_protocol_remain_compatible(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANY2XSAMPA_DATA_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "profile"))
    root = tmp_path / "app"
    root.mkdir()
    (root / "local-app.json").write_text('{"storage_mode":"per_user_v1"}', encoding="utf-8")
    expected = (tmp_path / "profile" / "Any2Xsampa").resolve()
    assert data_home(root) == expected
    assert state_dir(root) == expected / "state"
    assert model_cache(root) == expected / "cache/model-downloads"
    monkeypatch.setenv("ANY2XSAMPA_DATA_HOME", str(tmp_path / "portable"))
    assert data_home(root) == (tmp_path / "portable").resolve()
    assert VERSION == "any2xsampa.local.v1"
    transport = read("apps/Any2Xsampa.Desktop/Services/LocalTaskService.cs")
    assert f'Protocol = "{VERSION}"' in transport
    assert '"Any2Xsampa"' in transport
    assert '"any2xsampa.local.server"' in transport
