"""VSQ4 XML serialization regressions; these do not verify a native editor GUI."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO
from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET

import pytest

from any2xsampa.local.common import read, write
from any2xsampa.local.score_tasks import export_saved_score
from any2xsampa.vocaloid import vsqx
from any2xsampa.vocaloid.vsqx_alignment import parse_vsqx_timeline


def _legacy_xml(prefix: str = "ns0", *, pre_measure: int = 0) -> bytes:
    """Two notes suffice to reproduce the prefixed prefab serialization defect."""
    xml = f'''<?xml version="1.0" encoding="utf-8"?>
<{prefix}:vsq4 xmlns:{prefix}="{vsqx.VSQ4_NS}"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
 xsi:schemaLocation="{vsqx.VSQ4_NS} vsq4.xsd" id="project&amp;one">
  <{prefix}:vender>Yamaha corporation</{prefix}:vender>
  <{prefix}:version>4.0.0.3</{prefix}:version>
  <{prefix}:masterTrack>
    <{prefix}:resolution>480</{prefix}:resolution>
    <{prefix}:preMeasure>{pre_measure}</{prefix}:preMeasure>
    <{prefix}:timeSig><{prefix}:m>0</{prefix}:m><{prefix}:nu>4</{prefix}:nu><{prefix}:de>4</{prefix}:de></{prefix}:timeSig>
    <{prefix}:tempo><{prefix}:t>0</{prefix}:t><{prefix}:v>12000</{prefix}:v></{prefix}:tempo>
  </{prefix}:masterTrack>
  <{prefix}:vsTrack>
    <{prefix}:tNo>0</{prefix}:tNo>
    <{prefix}:name>轨道 &amp; &lt;track&gt;</{prefix}:name>
    <{prefix}:vsPart>
      <{prefix}:t>1920</{prefix}:t><{prefix}:playTime>1200</{prefix}:playTime>
      <{prefix}:note id="first">
        <{prefix}:t>0</{prefix}:t><{prefix}:dur>360</{prefix}:dur><{prefix}:n>60</{prefix}:n>
        <{prefix}:v>64</{prefix}:v><{prefix}:y><![CDATA[歌词&<> ns0:note rootXML:note]]></{prefix}:y>
        <{prefix}:p lock="1">a</{prefix}:p>
        <{prefix}:nStyle><{prefix}:v id="accent">50</{prefix}:v></{prefix}:nStyle>
      </{prefix}:note>
      <{prefix}:note id="second">
        <{prefix}:t>720</{prefix}:t><{prefix}:dur>480</{prefix}:dur><{prefix}:n>67</{prefix}:n>
        <{prefix}:v>80</{prefix}:v><{prefix}:y>二 &amp; &lt;三&gt;</{prefix}:y>
        <{prefix}:p lock="0">i</{prefix}:p>
        <{prefix}:nStyle><{prefix}:v id="bendDep">8</{prefix}:v></{prefix}:nStyle>
      </{prefix}:note>
      <{prefix}:plane>0</{prefix}:plane>
    </{prefix}:vsPart>
  </{prefix}:vsTrack>
</{prefix}:vsq4>'''
    return xml.encode("utf-8")


def _signature(node: ET.Element) -> tuple:
    """Compare expanded names, attributes, values, whitespace and child order."""
    return (
        node.tag, dict(node.attrib), node.text, node.tail,
        tuple(_signature(child) for child in node),
    )


def _assert_native_names(raw: bytes) -> None:
    document = minidom.parseString(raw)
    try:
        root = document.documentElement
        assert root.tagName == "vsq4"
        assert root.namespaceURI == vsqx.VSQ4_NS
        assert root.getAttribute("xmlns") == vsqx.VSQ4_NS
        assert len(document.getElementsByTagName("note")) == 2
        vsq_nodes = document.getElementsByTagNameNS(vsqx.VSQ4_NS, "*")
        assert all(node.prefix is None for node in vsq_nodes)
    finally:
        document.unlink()


@pytest.mark.parametrize(
    ("prefix", "destination_kind"),
    [("ns0", "binary"), ("rootXML", "path"), ("ns0", "str")],
)
def test_writer_uses_default_namespace_without_changing_xml_semantics(
    tmp_path: Path, prefix: str, destination_kind: str,
) -> None:
    legacy = _legacy_xml(prefix)
    tree = ET.ElementTree(ET.fromstring(legacy))
    expected = _signature(tree.getroot())
    output = tmp_path / "歌词 导出.vsqx"
    if destination_kind == "binary":
        with BytesIO() as handle:
            vsqx.write_vsqx(tree, handle)
            raw = handle.getvalue()
    else:
        vsqx.write_vsqx(tree, str(output) if destination_kind == "str" else output)
        raw = output.read_bytes()

    _assert_native_names(raw)
    assert _signature(ET.fromstring(raw)) == expected


def test_writer_does_not_mutate_the_callers_tree() -> None:
    tree = ET.ElementTree(ET.fromstring(_legacy_xml("rootXML")))
    root = tree.getroot()
    expected = _signature(deepcopy(root))
    original_nodes = tuple(root.iter())
    with BytesIO() as handle:
        vsqx.write_vsqx(tree, handle)

    assert tree.getroot() is root
    assert tuple(root.iter()) == original_nodes
    assert _signature(root) == expected


@pytest.mark.parametrize("pre_measure", [0, 1])
@pytest.mark.parametrize("score_source", [
    "audio_game", "audio_pyin_draft", "supplied_vsqx_phone_draft",
])
def test_saved_score_reserializes_legacy_xml_and_binds_actual_export_hash(
    tmp_path: Path, pre_measure: int, score_source: str,
) -> None:
    source = tmp_path / "旧 ns0 草稿.vsqx"
    source.write_bytes(_legacy_xml(pre_measure=pre_measure))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    score_path = tmp_path / "score.json"
    write(score_path, {"source": score_source})
    score_hash = hashlib.sha256(score_path.read_bytes()).hexdigest()
    delivery_path = tmp_path / "DELIVERY.json"
    write(delivery_path, {
        "score_path": str(score_path), "score_sha256": score_hash,
        "vsqx_path": str(source), "vsqx_sha256": source_hash,
        "source": score_source, "note_count": 2,
        "placeholder_note_count": 0, "model_forwards": 0,
    })
    inputs = {path: path.read_bytes() for path in (source, score_path, delivery_path)}
    output = tmp_path / "export" / "新 默认命名空间.vsqx"

    result = export_saved_score({
        "score_path": str(score_path), "score_sha256": score_hash,
        "output": str(output),
    })

    raw = output.read_bytes()
    _assert_native_names(raw)
    expected = ET.fromstring(inputs[source])
    ns = "{" + vsqx.VSQ4_NS + "}"
    if pre_measure == 0 and score_source in {"audio_game", "audio_pyin_draft"}:
        expected.find(ns + "masterTrack/" + ns + "preMeasure").text = "1"
        expected.find(".//" + ns + "vsPart/" + ns + "t").text = "3840"
    assert _signature(ET.fromstring(raw)) == _signature(expected)
    assert raw != inputs[source]
    for path, original in inputs.items():
        assert path.read_bytes() == original
    actual_output_hash = hashlib.sha256(raw).hexdigest()
    assert actual_output_hash != source_hash
    proof = read(output.with_name(output.name + ".provenance.json"))
    assert proof["export_path"] == str(output)
    assert proof["export_sha256"] == actual_output_hash
    assert proof["vsqx_path"] == str(source)
    assert proof["vsqx_sha256"] == source_hash
    assert proof["score_sha256"] == score_hash
    assert proof["note_count"] == result["note_count"] == 2
    assert Path(result["output"]) == output
    assert result["model_forwards"] == 0


_CDATA_FIELDS = (
    "vender", "version", "id", "name", "seqName", "comment", "y", "p",
    "content", "filePath",
)


def _string_field_tree(value: str) -> ET.ElementTree:
    tree = ET.ElementTree(ET.fromstring(_legacy_xml()))
    root = tree.getroot()
    ns = "{" + vsqx.VSQ4_NS + "}"
    master = root.find(ns + "masterTrack")
    track = root.find(ns + "vsTrack")
    part = track.find(ns + "vsPart")
    plugin = ET.SubElement(part, ns + "sPlug")
    aux = ET.SubElement(root, ns + "aux")
    wav_part = ET.SubElement(ET.SubElement(root, ns + "monoTrack"), ns + "wavPart")
    for parent, name in (
        (master, "seqName"), (track, "comment"), (plugin, "id"),
        (aux, "content"), (wav_part, "filePath"),
    ):
        ET.SubElement(parent, ns + name)
    for name in _CDATA_FIELDS:
        for node in root.iter(ns + name):
            node.text = value
    return tree


def test_native_cdata_string_fields_and_text_numeric_fields() -> None:
    tree = _string_field_tree("中文 & <value> ns0:note rootXML:note")
    expected = _signature(tree.getroot())
    with BytesIO() as handle:
        vsqx.write_vsqx(tree, handle)
        raw = handle.getvalue()
    assert _signature(ET.fromstring(raw)) == expected
    document = minidom.parseString(raw)
    try:
        for name in _CDATA_FIELDS:
            nodes = document.getElementsByTagName(name)
            assert nodes, name
            for node in nodes:
                assert [child.nodeType for child in node.childNodes] == [
                    node.CDATA_SECTION_NODE,
                ], f"{name} must use CDATA, not escaped TEXT"
        for name in ("resolution", "preMeasure", "tNo", "t", "dur", "n", "v", "plane", "playTime"):
            nodes = document.getElementsByTagName(name)
            assert nodes, name
            for node in nodes:
                assert [child.nodeType for child in node.childNodes] == [
                    node.TEXT_NODE,
                ], f"{name} must remain numeric TEXT"
    finally:
        document.unlink()


@pytest.mark.parametrize("empty", [None, ""])
def test_native_cdata_empty_comment_is_not_self_closed(empty: str | None) -> None:
    tree = _string_field_tree("ordinary text")
    comment = tree.find(".//{" + vsqx.VSQ4_NS + "}comment")
    comment.text = empty
    expected = _signature(ET.fromstring(ET.tostring(tree.getroot())))
    with BytesIO() as handle:
        vsqx.write_vsqx(tree, handle)
        raw = handle.getvalue()

    assert b"<comment><![CDATA[]]></comment>" in raw
    assert _signature(ET.fromstring(raw)) == expected
    assert comment.text == empty


def test_native_cdata_splits_terminators_without_changing_text() -> None:
    value = "]]>歌词&<> ns0:note ]]> rootXML:note]]>"
    tree = _string_field_tree(value)
    expected = _signature(tree.getroot())
    with BytesIO() as handle:
        vsqx.write_vsqx(tree, handle)
        raw = handle.getvalue()

    assert _signature(ET.fromstring(raw)) == expected
    assert _signature(tree.getroot()) == expected
    document = minidom.parseString(raw)
    try:
        for name in _CDATA_FIELDS:
            for node in document.getElementsByTagName(name):
                assert len(node.childNodes) >= 2, f"{name} must split CDATA terminators"
                assert all(child.nodeType == node.CDATA_SECTION_NODE for child in node.childNodes)
                assert "".join(child.data for child in node.childNodes) == value
    finally:
        document.unlink()


def test_native_cdata_xml_declaration_is_utf8_standalone_no() -> None:
    with BytesIO() as handle:
        vsqx.write_vsqx(_string_field_tree("歌词"), handle)
        raw = handle.getvalue()
    assert raw.startswith(b"<?xml ")
    document = minidom.parseString(raw)
    try:
        assert document.version == "1.0"
        assert document.encoding.upper() == "UTF-8"
        assert document.standalone is False
    finally:
        document.unlink()


def _preroll_tree(
    numerator: int, denominator: int = 4, resolution: int = 480,
    *, pre_measure: int = 0,
) -> ET.ElementTree:
    tree = ET.ElementTree(ET.fromstring(_legacy_xml(pre_measure=pre_measure)))
    root = tree.getroot()
    ns = "{" + vsqx.VSQ4_NS + "}"
    master = root.find(ns + "masterTrack")
    master.find(ns + "resolution").text = str(resolution)
    master.remove(master.find(ns + "timeSig"))
    for measure, nu, de in ((0, numerator, denominator), (2, 5, 8)):
        signature = ET.SubElement(master, ns + "timeSig")
        for name, value in (("m", measure), ("nu", nu), ("de", de)):
            ET.SubElement(signature, ns + name).text = str(value)
    for tick, value in ((480, 9000), (960, 15000)):
        tempo = ET.SubElement(master, ns + "tempo")
        ET.SubElement(tempo, ns + "t").text = str(tick)
        ET.SubElement(tempo, ns + "v").text = str(value)
    track = root.find(ns + "vsTrack")
    part = track.find(ns + "vsPart")
    part.find(ns + "t").text = "0"
    second_part = deepcopy(part)
    second_part.find(ns + "t").text = "2400"
    track.append(second_part)
    second_track = deepcopy(track)
    second_track.find(ns + "tNo").text = "1"
    second_track.remove(second_track.findall(ns + "vsPart")[1])
    second_track.find(ns + "vsPart/" + ns + "t").text = "4320"
    root.append(second_track)
    for track_name in ("monoTrack", "stTrack"):
        audio_track = ET.SubElement(root, ns + track_name)
        for tick in (0, 240):
            audio_part = ET.SubElement(audio_track, ns + "wavPart")
            ET.SubElement(audio_part, ns + "t").text = str(tick)
            ET.SubElement(audio_part, ns + "playTime").text = "4800"
            ET.SubElement(audio_part, ns + "filePath").text = "音频 & <mix>.wav"
    return tree


@pytest.mark.parametrize(
    ("numerator", "resolution", "bar_ticks"),
    [(4, 480, 1920), (3, 480, 1440), (3, 960, 2880)],
)
def test_zero_preroll_moves_only_absolute_project_positions(
    numerator: int, resolution: int, bar_ticks: int,
) -> None:
    tree = _preroll_tree(numerator, resolution=resolution)
    root = tree.getroot()
    original_nodes = tuple(root.iter())
    before = _signature(root)
    expected = deepcopy(root)
    ns = "{" + vsqx.VSQ4_NS + "}"
    master = expected.find(ns + "masterTrack")
    master.find(ns + "preMeasure").text = "1"
    for track_name, part_name in (
        ("vsTrack", "vsPart"), ("monoTrack", "wavPart"), ("stTrack", "wavPart"),
    ):
        for part in expected.findall(ns + track_name + "/" + ns + part_name):
            tick = part.find(ns + "t")
            tick.text = str(int(tick.text) + bar_ticks)
    for tempo in master.findall(ns + "tempo"):
        tick = tempo.find(ns + "t")
        if int(tick.text) != 0:
            tick.text = str(int(tick.text) + bar_ticks)
    for signature in master.findall(ns + "timeSig"):
        measure = signature.find(ns + "m")
        if int(measure.text) != 0:
            measure.text = str(int(measure.text) + 1)

    with BytesIO() as handle:
        vsqx.write_vsqx(tree, handle, repair_zero_preroll=True)
        actual = ET.fromstring(handle.getvalue())

    assert tree.getroot() is root
    assert tuple(root.iter()) == original_nodes
    assert _signature(root) == before
    expected_notes = [_signature(note) for note in root.iter(ns + "note")]
    assert len(expected_notes) == 6
    assert [_signature(note) for note in actual.iter(ns + "note")] == expected_notes
    assert _signature(actual) == _signature(expected)


@pytest.mark.parametrize("pre_measure", [1, 2])
def test_existing_positive_preroll_is_not_migrated(pre_measure: int) -> None:
    tree = _preroll_tree(3, pre_measure=pre_measure)
    expected = _signature(tree.getroot())
    with BytesIO() as handle:
        vsqx.write_vsqx(tree, handle, repair_zero_preroll=True)
        actual = ET.fromstring(handle.getvalue())

    assert _signature(actual) == expected
    assert _signature(tree.getroot()) == expected


@pytest.mark.parametrize("numerator", [4, 3])
def test_zero_preroll_keeps_parsed_note_wallclock_times(
    tmp_path: Path, numerator: int,
) -> None:
    tree = _preroll_tree(numerator)
    source = tmp_path / "zero-preroll.vsqx"
    tree.write(source, encoding="utf-8", xml_declaration=True)
    before = parse_vsqx_timeline(source)
    output = tmp_path / "native-preroll.vsqx"
    vsqx.write_vsqx(tree, output, repair_zero_preroll=True)
    after = parse_vsqx_timeline(output)

    ns = "{" + vsqx.VSQ4_NS + "}"
    assert ET.parse(output).findtext(ns + "masterTrack/" + ns + "preMeasure") == "1"
    assert before["note_count"] == after["note_count"] == 6
    assert before["notes"][0]["start_sec"] == 0
    # The parser reports project time; only the audio-relative comparison
    # subtracts the added bar, at the unchanged initial tempo of 120 BPM.
    preroll_seconds = numerator * 60 / 120
    assert after["notes"][0]["start_tick"] == numerator * 480
    assert after["notes"][0]["start_sec"] == pytest.approx(preroll_seconds)
    for original, restored in zip(before["notes"], after["notes"], strict=True):
        assert restored["start_sec"] - preroll_seconds == pytest.approx(original["start_sec"], abs=1e-6)
        assert restored["end_sec"] - preroll_seconds == pytest.approx(original["end_sec"], abs=1e-6)
        for field in ("duration_ticks", "pitch", "lyric", "phonemes"):
            assert restored[field] == original[field]
