from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET


MAIN_PAGE = Path(__file__).resolve().parents[1] / "apps/Any2Xsampa.Desktop/MainPage.xaml"
UI = "{http://schemas.microsoft.com/winfx/2006/xaml/presentation}"
NAME = "{http://schemas.microsoft.com/winfx/2006/xaml}Name"


def test_real_inference_stays_visible_outside_scroll_region() -> None:
    root = ET.parse(MAIN_PAGE).getroot()
    names = {node.attrib[NAME]: node for node in root.iter() if NAME in node.attrib}
    parents = {child: parent for parent in root.iter() for child in parent}

    def inside(child, ancestor) -> bool:
        while child in parents:
            child = parents[child]
            if child is ancestor:
                return True
        return False

    rows = names["PageLayout"].find(f"{UI}Grid.RowDefinitions")
    assert [r.attrib["Height"] for r in rows] == ["Auto", "Auto", "*", "Auto"]
    scroll = names["WorkspaceScroll"]
    assert scroll.attrib["VerticalScrollBarVisibility"] == "Auto"
    assert names["RunButton"].attrib["Click"] == "Recognize_Click"
    assert names["CancelButton"].attrib["Click"] == "Cancel_Click"
    assert not inside(names["RunButton"], scroll)
    assert inside(names["RunButton"], names["TaskActions"])
    assert names["TaskActions"].attrib.get("Grid.Row") is None
    assert names["TaskActions"] in list(parents[names["TaskActions"]])
    assert not inside(names["CancelButton"], scroll)
    for name in ("AudioPathBox", "StartBox", "DurationBox", "Player", "Waveform", "PhoneList"):
        assert inside(names[name], scroll)
    assert names["WholeMode"].attrib["IsChecked"] == "True"
    assert names["PhoneList"].attrib["Height"] == "320"


def test_narrow_window_stacks_review_under_audio() -> None:
    root = ET.parse(MAIN_PAGE).getroot()
    names = {node.attrib[NAME]: node for node in root.iter() if NAME in node.attrib}
    assert names["ReviewPanel"].attrib["Grid.Row"] == "1"
    assert names["ReviewColumn"].attrib["Width"] == "0"
    setters = [node.attrib["Target"] for node in root.iter() if node.tag == f"{UI}Setter"]
    assert "ReviewPanel.(Grid.Row)" in setters
    assert "ReviewColumn.Width" in setters
