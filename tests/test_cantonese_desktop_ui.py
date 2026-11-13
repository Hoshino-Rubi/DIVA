"""Cantonese UI contract checks, without building or running the desktop app."""
from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree as ET


DESKTOP = Path(__file__).resolve().parents[1] / "apps/Any2Xsampa.Desktop"
UI = "{http://schemas.microsoft.com/winfx/2006/xaml/presentation}"
NAME = "{http://schemas.microsoft.com/winfx/2006/xaml}Name"


def test_cantonese_tab_has_four_actions_and_copyable_multiline_outputs() -> None:
    root = ET.parse(DESKTOP / "MainPage.xaml").getroot()
    names = {node.attrib[NAME]: node for node in root.iter() if NAME in node.attrib}
    tab = names["CantoneseTab"]
    assert tab.tag == f"{UI}PivotItem"
    assert {button.attrib["Click"] for button in tab.iter(f"{UI}Button")} == {
        "CantoneseModel_Click", "CantonesePrepare_Click", "CantoneseRun_Click", "CantoneseOpen_Click",
    }
    assert len(list(tab.iter(f"{UI}Button"))) == 4
    assert not list(tab.iter(f"{UI}NumberBox"))
    assert not list(tab.iter(f"{UI}RadioButton"))
    assert any(node.attrib.get("Text") == "{Binding Text, ElementName=AudioPathBox}" for node in tab.iter())
    for name in ("CantoneseJyutpingBox", "CantoneseIpaBox"):
        output = names[name]
        assert output.attrib["IsReadOnly"] == "True"
        assert output.attrib["AcceptsReturn"] == "True"
        assert output.attrib["TextWrapping"] == "Wrap"
        assert output.attrib["ScrollViewer.VerticalScrollBarVisibility"] == "Auto"
        assert output.attrib.get("IsEnabled", "True") == "True"
    assert names["CantoneseOpenButton"].attrib["IsEnabled"] == "False"


def test_cantonese_shared_lifecycle_hooks_are_wired() -> None:
    page = (DESKTOP / "MainPage.xaml.cs").read_text(encoding="utf-8-sig")
    assert "InitializeCantonese();" in page
    assert "await RefreshCantoneseStatus();" in page
    assert re.search(r"Language_Changed\(.*?RenderCantoneseLanguage\(\);", page, re.S)
    assert re.search(r"private void SetBusy\(bool busy\)\s*\{\s*SetCantoneseBusy\(busy\);", page)
    assert 'if(!string.Equals(audio,file,StringComparison.OrdinalIgnoreCase))ResetCantoneseResult();' in page


def test_cantonese_results_do_not_enter_phone_score_or_vsqx_pipeline() -> None:
    code = (DESKTOP / "MainPage.Cantonese.cs").read_text(encoding="utf-8-sig")
    for identifier in ("resultPath", "resultHash", "outputDir", "scorePath", "scoreHash", "scoreSource",
                       "LoadResult", "LoadScore", "RunTask", "SaveDraft", "GenerateGameScore"):
        assert not re.search(rf"\b{identifier}\b", code), identifier
    assert "AutoVsqxBox.IsChecked" not in code
    assert 'StorageFolder.GetFolderFromPathAsync(cantoneseOutputDir)' in code
    assert 'new { bundle_dir = cantoneseBundleDir }' in code
    assert 'start = StartBox.Value, duration = DurationBox.Value' in code
    assert 'scope = WholeMode.IsChecked == true ? "whole_file" : "clip"' in code


def test_cantonese_static_copy_is_short_and_language_switch_has_no_io() -> None:
    code = (DESKTOP / "MainPage.Cantonese.cs").read_text(encoding="utf-8-sig")
    render = code.split("private void RenderCantoneseLanguage()", 1)[1].split("private void SetCantoneseBusy", 1)[0]
    assert 'UiLanguage.Text("粤语增强", "Cantonese")' in render
    assert "CallAsync" not in render and "ResetCantoneseResult" not in render
    assert "CantoneseJyutpingBox.Text" not in render and "CantoneseIpaBox.Text" not in render
    assert all(len(text) <= 24 for text in re.findall(r'UiLanguage.Text\("([^"\n]+)"', code))


def test_prepare_is_only_called_from_explicit_click() -> None:
    code = (DESKTOP / "MainPage.Cantonese.cs").read_text(encoding="utf-8-sig")
    handler = code.split("private async void CantonesePrepare_Click", 1)[1].split("private async void CantoneseRun_Click", 1)[0]
    assert 'service.CallAsync("cantonese4_prepare", parameters, active.Token)' in handler
    assert code.count('"cantonese4_prepare"') == 1
    assert code.count("CantonesePrepare_Click") == 1
    assert "ModelProxyBox.Text" in handler
    page = (DESKTOP / "MainPage.xaml.cs").read_text(encoding="utf-8-sig")
    assert "cantonese4_prepare" not in page and "CantonesePrepare_Click" not in page


def test_cantonese_tab_scopes_top_actions_without_changing_startup_defaults() -> None:
    root = ET.parse(DESKTOP / "MainPage.xaml").getroot()
    names = {node.attrib[NAME]: node for node in root.iter() if NAME in node.attrib}
    assert names["WorkspaceTabs"].attrib["SelectionChanged"] == "WorkspaceTabs_SelectionChanged"
    assert names["RunButton"].attrib["Click"] == "Recognize_Click"
    for name in ("RunButton", "AutoVsqxBox", "WholeMode", "ClipMode", "CancelButton", "BusyRing"):
        assert names[name].attrib.get("Visibility", "Visible") == "Visible"
