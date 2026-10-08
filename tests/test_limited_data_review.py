from __future__ import annotations

import copy
from html.parser import HTMLParser

from tools.build_limited_data_review import centered_range, choose_review_items, render_html


def fixture() -> tuple[dict, dict]:
    song = {
        "song_id": "freedom-flower", "display_name": "<script>song</script>", "split": "train",
        "original_audio": {"duration_sec": 100}, "rendered_audio": {"duration_sec": 110},
        "notes": [{"note_index": 1, "note_id": "n1", "start_sec": 4, "end_sec": 5, "unresolved_tokens": ["uo"], "raw_v4c_tokens": ["k", "uo"], "raw_lyric": "<script>lyric</script>"}],
    }
    match = {
        "clips": [{
            "example_id": "freedom-flower-original-01", "ambiguous": True, "relative_cost_gap": 0.01,
            "requested_original_start_sec": 20, "requested_original_end_sec": 38,
            "candidates": [{"rendered_start_sec": 18, "rendered_end_sec": 36}],
        }],
    }
    return {"songs": [song, {**song, "song_id": "king-of-karaoke", "split": "test"}]}, {"freedom-flower": match}


def test_review_selection_is_bounded_pending_and_excludes_test() -> None:
    reference, matches = fixture()
    before = copy.deepcopy(reference)
    items = choose_review_items(reference, matches)
    assert len(items) == 2
    assert all(item["song_id"] == "freedom-flower" and item["review_status"] == "pending" for item in items)
    assert items[0]["answer_scope"] == "section_identity_only_not_phone_transcription"
    assert items[1]["answer_scope"] == "engine_reference_semantics_only_not_original_truth"
    assert reference == before


def test_preview_bounds_stay_inside_short_sources() -> None:
    assert centered_range(-4, 12, 6) == (0, 6)
    assert centered_range(200, 12, 100) == (88, 100)


def test_review_html_escapes_untrusted_labels_and_has_no_scripts() -> None:
    reference, matches = fixture()
    items = choose_review_items(reference, matches)
    for item in items:
        item["previews"] = [{
            "role": "rendered", "audio_file": "R01-rendered.wav", "waveform_file": "R01-rendered.png",
            "source_crop": {"effective_start_sec": 4, "effective_end_sec": 10},
        }]
    document = render_html(items)
    assert "<script>" not in document
    assert "&lt;script&gt;" in document
    assert "2 \u9879" in document
    assert "media-src 'self' file:" in document

    class Tags(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.audio_count = 0
            self.image_count = 0

        def handle_starttag(self, tag, attrs) -> None:
            if tag == "audio":
                self.audio_count += 1
                assert dict(attrs)["src"].startswith("media/")
            if tag == "img":
                self.image_count += 1

    parser = Tags()
    parser.feed(document)
    assert parser.audio_count == parser.image_count == 2
