"""VOCALOID-oriented export helpers."""

from any2xsampa.vocaloid.draft import (
    DEFAULT_SINGER_TARGET,
    VOCAL_TRACK_DRAFT_SCHEMA_VERSION,
    VocalNoteDraft,
    VocalTrackDraft,
    VocalTrackDraftTrack,
    load_vocal_track_draft,
)
from any2xsampa.vocaloid.phoneme_policy import (
    KNOWN_SILENT_OR_UNSAFE_SEQUENCES,
    KNOWN_SILENT_OR_UNSAFE_TOKENS,
    KNOWN_UNSAFE_SUFFIXES,
    assert_miku_v4c_note_phonemes_playable,
    collect_miku_v4c_policy_issues,
    note_phoneme_sequence,
    validate_miku_v4c_note_phonemes,
)
from any2xsampa.vocaloid.phg2 import (
    PHG2_SCHEMA_VERSION,
    PHG2ParseError,
    parse_phg2,
    parse_phg2_bytes,
    parse_phg2_path,
)
from any2xsampa.vocaloid.phone_segments import draft_from_phone_segments
from any2xsampa.vocaloid.miku_phone_segments import draft_miku_v4c_from_phone_segments
from any2xsampa.vocaloid.evidence_fusion import EvidenceFusionError, build_pitch_skeleton_payload
from any2xsampa.vocaloid.vsqx import VSQ4_NS, build_vsqx_tree, export_vsqx_file, export_vsqx_from_template

__all__ = [
    "DEFAULT_SINGER_TARGET",
    "PHG2_SCHEMA_VERSION",
    "PHG2ParseError",
    "VOCAL_TRACK_DRAFT_SCHEMA_VERSION",
    "VocalNoteDraft",
    "VocalTrackDraft",
    "VocalTrackDraftTrack",
    "VSQ4_NS",
    "build_vsqx_tree",
    "export_vsqx_file",
    "export_vsqx_from_template",
    "draft_from_phone_segments",
    "draft_miku_v4c_from_phone_segments",
    "EvidenceFusionError",
    "build_pitch_skeleton_payload",
    "KNOWN_SILENT_OR_UNSAFE_SEQUENCES",
    "KNOWN_SILENT_OR_UNSAFE_TOKENS",
    "KNOWN_UNSAFE_SUFFIXES",
    "assert_miku_v4c_note_phonemes_playable",
    "collect_miku_v4c_policy_issues",
    "note_phoneme_sequence",
    "validate_miku_v4c_note_phonemes",
    "load_vocal_track_draft",
    "parse_phg2",
    "parse_phg2_bytes",
    "parse_phg2_path",
]
