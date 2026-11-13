from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET


MAIN_PAGE = Path(__file__).resolve().parents[1] / "apps/Any2Xsampa.Desktop/MainPage.xaml"
UI = "{http://schemas.microsoft.com/winfx/2006/xaml/presentation}"
NAME = "{http://schemas.microsoft.com/winfx/2006/xaml}Name"


def layout():
    root = ET.parse(MAIN_PAGE).getroot()
    names = {node.attrib[NAME]: node for node in root.iter() if NAME in node.attrib}
    parents = {child: parent for parent in root.iter() for child in parent}
    return root, names, parents


def ancestors(node, parents):
    while node in parents:
        node = parents[node]
        yield node


def test_real_inference_stays_visible_outside_scroll_region() -> None:
    root, names, parents = layout()
    rows = names["PageLayout"].find(f"{UI}Grid.RowDefinitions")
    assert [r.attrib["Height"] for r in rows] == ["Auto", "Auto", "*", "Auto"]
    assert names["RunButton"].attrib["Click"] == "Recognize_Click"
    assert names["CancelButton"].attrib["Click"] == "Cancel_Click"
    for name in ("RunButton", "CancelButton"):
        chain = list(ancestors(names[name], parents))
        assert names["TaskActions"] in chain
        assert not any(node.tag in {f"{UI}ScrollViewer", f"{UI}Pivot", f"{UI}PivotItem"} for node in chain)
        top_level = next(node for node in chain if parents.get(node) is names["PageLayout"])
        assert top_level.attrib.get("Grid.Row", "0") == "1"
    assert names["WholeMode"].attrib["IsChecked"] == "True"
    assert any(
        node.tag == f"{UI}KeyboardAccelerator"
        and node.attrib.get("Key") == "Enter"
        and node.attrib.get("Modifiers") == "Control"
        for node in names["RunButton"].iter()
    )


def test_four_tabs_keep_configuration_and_scrolling_independent() -> None:
    root, names, parents = layout()
    pivot, = root.iter(f"{UI}Pivot")
    assert parents[pivot] is names["PageLayout"]
    assert pivot.attrib["Grid.Row"] == "2"
    tabs = pivot.findall(f"{UI}PivotItem")
    assert len(tabs) == 4
    groups = (
        ("AudioPathBox", "ModelBox", "DeviceBox", "StartBox", "DurationBox", "Player", "Waveform", "PhoneList"),
        ("NoteSnapshotBox", "ScoreTempo", "AnalyzeScoreButton", "ImportNotesButton", "ImportScoreButton", "ExportScoreButton", "ScoreList"),
        ("ModelSetupStatus", "ModelProxyBox", "AutoDownloadBox", "InstallModelsButton", "ImportModelsButton", "EnvironmentActions", "ActivityLog"),
        ("CantoneseModelButton", "CantonesePrepareButton", "CantoneseRunButton", "CantoneseOpenButton", "CantoneseJyutpingBox", "CantoneseIpaBox"),
    )
    for tab, controls in zip(tabs, groups):
        assert any(node.tag == f"{UI}ScrollViewer" for node in tab.iter())
        assert not any(node.tag == f"{UI}ScrollViewer" for node in ancestors(tab, parents))
        for name in controls:
            assert tab in ancestors(names[name], parents), name

    def scroll_parent(name):
        return next(node for node in ancestors(names[name], parents) if node.tag == f"{UI}ScrollViewer")

    scrolls = [scroll_parent(name) for name in ("AudioPathBox", "NoteSnapshotBox", "ModelProxyBox", "EnvironmentActions", "ActivityLog")]
    assert len(set(scrolls)) == len(scrolls)
    for scroll in scrolls:
        assert scroll.attrib["VerticalScrollBarVisibility"] == "Auto"
        assert scroll.attrib["HorizontalScrollBarVisibility"] == "Disabled"
    assert scroll_parent("AudioPathBox") is names["WorkspaceScroll"]


def test_result_lists_fill_their_own_panels_and_scroll_independently() -> None:
    _, names, parents = layout()
    for name in ("PhoneList", "ScoreList"):
        node = names[name]
        assert node.attrib["ScrollViewer.VerticalScrollBarVisibility"] == "Auto"
        # Wide mode stretches the lists; only the narrow visual state limits phones.
        assert "Height" not in node.attrib
        scrolls = [parent for parent in ancestors(node, parents) if parent.tag == f"{UI}ScrollViewer"]
        assert scrolls == ([names["ReviewViewport"]] if name == "PhoneList" else [])
        panel = parents[node]
        assert panel.tag == f"{UI}Grid"
        rows = panel.find(f"{UI}Grid.RowDefinitions")
        assert rows[int(node.attrib.get("Grid.Row", "0"))].attrib["Height"] == "*"
    viewport = names["ReviewViewport"]
    assert parents[names["ReviewPanel"]] is viewport
    assert viewport.attrib["VerticalScrollBarVisibility"] == "Disabled"
    assert viewport.attrib["HorizontalScrollBarVisibility"] == "Disabled"
    assert viewport.attrib["HorizontalScrollMode"] == "Disabled"
    assert not any(parent.tag == f"{UI}ScrollViewer" for parent in ancestors(viewport, parents))


def test_narrow_window_stacks_review_under_audio() -> None:
    root, names, parents = layout()
    assert parents[names["WorkspaceScroll"]] is names["WorkspacePanels"]
    assert parents[names["ReviewViewport"]] is names["WorkspacePanels"]
    assert parents[names["ReviewPanel"]] is names["ReviewViewport"]
    assert names["WorkspaceScroll"].attrib.get("Grid.Row", "0") == "0"
    assert names["ReviewViewport"].attrib["Grid.Row"] == "1"
    assert names["ReviewColumn"].attrib["Width"] == "0"
    rows = names["WorkspacePanels"].find(f"{UI}Grid.RowDefinitions")
    assert [row.attrib["Height"] for row in rows] == ["*", "*"]
    narrow = names["StackedPanels"].find(f"{UI}VisualState.StateTriggers/{UI}AdaptiveTrigger")
    wide = names["SideBySidePanels"].find(f"{UI}VisualState.StateTriggers/{UI}AdaptiveTrigger")
    assert float(narrow.attrib["MinWindowWidth"]) == 0
    assert float(wide.attrib["MinWindowWidth"]) == 800
    narrow_setters = {
        node.attrib["Target"]: node.attrib["Value"]
        for node in names["StackedPanels"].iter(f"{UI}Setter")
    }
    # Keep the list usable without letting review headers/actions consume its height;
    # overflowing review content scrolls separately from audio configuration.
    assert narrow_setters["ReviewViewport.VerticalScrollBarVisibility"] == "Auto"
    assert float(narrow_setters["PhoneList.Height"]) == 160
    setters = {
        node.attrib["Target"]: node.attrib["Value"]
        for node in names["SideBySidePanels"].iter(f"{UI}Setter")
    }
    assert setters["ReviewViewport.(Grid.Row)"] == "0"
    assert setters["ReviewViewport.(Grid.Column)"] == "1"
    assert setters["ReviewColumn.Width"] == "*"
    assert setters["SecondWorkspaceRow.Height"] == "0"


def test_model_controls_stack_until_the_wide_control_breakpoint() -> None:
    _, names, _ = layout()
    assert names["ModelBox"].attrib["Grid.ColumnSpan"] == "2"
    assert names["DeviceBox"].attrib["Grid.Row"] == "1"
    wide = names["WideControls"].find(f"{UI}VisualState.StateTriggers/{UI}AdaptiveTrigger")
    assert float(wide.attrib["MinWindowWidth"]) == 1200
    setters = {
        node.attrib["Target"]: node.attrib["Value"]
        for node in names["WideControls"].iter(f"{UI}Setter")
    }
    assert setters["ModelBox.(Grid.ColumnSpan)"] == "1"
    assert setters["DeviceBox.(Grid.Row)"] == "0"
    assert setters["DeviceBox.(Grid.Column)"] == "1"


def test_player_transport_controls_have_room_to_expand() -> None:
    _, names, parents = layout()
    player = names["Player"]
    assert player.attrib["AreTransportControlsEnabled"] == "True"
    assert float(player.attrib["MinHeight"]) >= 104
    assert "Height" not in player.attrib
    assert "MaxHeight" not in player.attrib
    assert names["WorkspaceScroll"] in ancestors(player, parents)
