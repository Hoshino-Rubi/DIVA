"""Localization coverage and UI wiring; intentionally independent of panel layout."""
from __future__ import annotations

import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET


DESKTOP = Path(__file__).resolve().parents[1] / "apps/Any2Xsampa.Desktop"
NAME = "{http://schemas.microsoft.com/winfx/2006/xaml}Name"
CHINESE = re.compile(r"[\u3400-\u9fff]")
EVENTS = {
    "Loaded", "Click", "Checked", "Unchecked", "SelectionChanged",
    "ValueChanged", "SizeChanged", "PointerPressed", "DragOver", "Drop",
}
ORIGINAL_PAGE_NAMES = set("""
ActivityLog AnalyzeScoreButton AudioPanel AudioPathBox AudioStatus AutoDownloadBox
AutoVsqxBox BusyRing CancelButton ClipMode DeviceBox DraftStatus
DurationBox EnvironmentActions ExportScoreButton GpuStatus ImportModelsButton
ImportNotesButton ImportScoreButton InstallModelsButton LoopBox ModelBox
ModelDownloadProgress ModelProxyBox ModelSetupStatus ModelStatus NoteRuntimeStatus
NoteSnapshotBox PageLayout PhoneList Player ResultStatus ReviewColumn ReviewPanel
RunButton ScoreList ScoreStatus ScoreTempo StartBox
StatusBar TaskActions UseDraftBox Waveform WholeMode WholeProgress
WorkspacePanels WorkspaceScroll
""".split())
ORIGINAL_PAGE_HANDLERS = set("""
AnalyzeScore_Click Audio_DragOver Audio_Drop AutoDownload_Changed
BrowseNoteModel_Click Browse_Click Cancel_Click CheckUpdate_Click Doctor_Click
ExportDiagnostics_Click ExportIpa_Click ExportScore_Click ExportXSampa_Click
ImportModels_Click ImportNotes_Click ImportScore_Click InstallModels_Click
Model_Changed OpenOutput_Click Page_Loaded Phone_Selected Play_Click Range_Changed
Recognize_Click Replay_Click Score_Selected Stop_Click Unload_Click
Waveform_PointerPressed Waveform_SizeChanged
""".split())


def test_existing_control_names_and_handlers_are_preserved() -> None:
    page = ET.parse(DESKTOP / "MainPage.xaml").getroot()
    names = {node.attrib[NAME] for node in page.iter() if NAME in node.attrib}
    handlers = {
        value for node in page.iter() for key, value in node.attrib.items()
        if key in EVENTS
    }
    assert not (ORIGINAL_PAGE_NAMES - names), sorted(ORIGINAL_PAGE_NAMES - names)
    assert not (ORIGINAL_PAGE_HANDLERS - handlers), sorted(ORIGINAL_PAGE_HANDLERS - handlers)
    window = ET.parse(DESKTOP / "MainWindow.xaml").getroot()
    assert {"AppTitleBar", "RootFrame"} <= {
        node.attrib[NAME] for node in window.iter() if NAME in node.attrib
    }


def test_attached_chinese_ui_sources_have_english_dictionary_entries() -> None:
    dictionary = DESKTOP / "Localization/StaticText.en.json"
    translations = json.loads(dictionary.read_text(encoding="utf-8-sig"))
    assert isinstance(translations, dict), dictionary

    sources: set[str] = set()
    for path in DESKTOP.glob("*.xaml"):
        for node in ET.parse(path).getroot().iter():
            for key, value in node.attrib.items():
                attached_property = key.rsplit("}", 1)[-1]
                if attached_property.startswith("UiText.") and CHINESE.search(value):
                    sources.add(value)
    assert sources, "No attached Chinese UiText sources found; coverage would be empty"
    missing = sorted(source for source in sources if source not in translations)
    assert not missing, f"Chinese UiText sources missing English translations: {missing}"
    invalid = {
        source: translations[source] for source in sources
        if not isinstance(translations[source], str)
        or not translations[source].strip()
    }
    # Language selector autonyms (e.g. 中文) may intentionally remain native.
    assert not invalid, f"English UI dictionary entries must be nonempty strings: {invalid}"
