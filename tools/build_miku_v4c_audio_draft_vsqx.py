"""Audio-product wrapper for the recovered Miku V4C draft lane."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from any2xsampa.adapters.forced_aligner import (
    FixtureForcedPhoneAligner,
    ForcedPhoneAlignerError,
    compare_forced_and_ctc_alignment,
)
from any2xsampa.adapters.forced_alignment_acceptance import (
    QUALIFIED_MODEL_ROUTE,
    ForcedPhoneAcceptanceError,
    load_forced_phone_acceptance,
)
from any2xsampa.adapters.mfa_forced_aligner import MfaForcedPhoneAligner
from any2xsampa.adapters.recognizer import (
    ConfigurableUniversalPhoneRecognizer,
    FixtureUniversalPhoneRecognizer,
    RecognizerConfig,
)
from any2xsampa.asr import (
    ASRTranscriptError,
    Qwen3ASRAdapter,
    Qwen3ASRLoadError,
    Qwen3ASRRequest,
    dumps_transcript_json,
    validate_transcript_payload,
)
from any2xsampa.cantonese.g2p import (
    CantoneseG2PError,
    PyCantoneseG2PAdapter,
    ReviewedLexiconCantoneseG2PAdapter,
    resolve_lyrics_timeline,
)
from any2xsampa.lyrics.timeline import (
    LyricsTimelineError,
    canonical_timeline_sha256,
    load_lyrics_timeline,
    sha256_file,
    timeline_from_asr_transcript,
)
from any2xsampa.music import (
    PitchNoteDependencyError,
    analyze_audio_pitch_notes,
)
from any2xsampa.pipeline.normalizer import ConfigurablePhoneNormalizer
from tools.build_miku_v4c_draft_vsqx import run_draft_vsqx

SCHEMA_VERSION = "miku_v4c_audio_draft_vsqx_run_v1"


def run_audio_draft_vsqx(
    *,
    audio: Path,
    template: Path,
    output_root: Path,
    lyrics: str | None = None,
    jyutping_override: str | None = None,
    asr_transcript: Path | None = None,
    fake_asr: bool = False,
    asr_model: str | None = None,
    language: str | None = "Chinese",
    context: str = "",
    timestamps: bool = False,
    vocal_stem: Path | None = None,
    lyrics_timeline: Path | None = None,
    forced_phone_alignment: Path | None = None,
    forced_phone_acceptance: Path | None = None,
    mfa_alignment: Path | None = None,
    mfa_alignment_manifest: Path | None = None,
    mfa_dictionary: Path | None = None,
    forced_model_hash: str | None = None,
    forced_model_revision: str | None = None,
    ctc_phone_segments: Path | None = None,
    g2p_lexicon: Path | None = None,
    strict_product: bool = False,
    pitch_analysis: Path | None = None,
    auto_pitch: bool = False,
    phone_segments: Path | None = None,
    auto_phones: bool = False,
    recognizer_model: Path | None = None,
    recognizer_device: str = "cuda",
    pitches: list[int] | None = None,
    tempo: float = 128.0,
    prefix: str = "miku_v4c_audio_draft",
    pretty: bool = False,
) -> tuple[int, dict[str, Any]]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if not _valid_prefix(prefix):
        safe_summary = output_root / "miku_v4c_audio_draft_invalid_prefix_summary.json"
        return _write_failure(
            safe_summary,
            {"audio_product_summary": str(safe_summary)},
            "invalid_arguments",
            ["prefix must be a simple filename component"],
        )
    product_summary_path = output_root / f"{prefix}_audio_product_summary.json"
    asr_path = output_root / f"{prefix}_asr_transcript.json"
    artifacts = {
        "audio_product_summary": str(product_summary_path),
        "draft": str(output_root / f"{prefix}.json"),
        "vsqx": str(output_root / f"{prefix}.vsqx"),
        "draft_summary": str(output_root / f"{prefix}_summary.json"),
        "asr_transcript": str(asr_path),
        "pitch_analysis": str(output_root / f"{prefix}_pitch_analysis.json"),
        "phone_segments": str(output_root / f"{prefix}_phone_segments.json"),
        "ctc_phone_segments": str(output_root / f"{prefix}_ctc_phone_segments.json"),
        "lyrics_timeline": str(output_root / f"{prefix}_lyrics_timeline.json"),
        "g2p": str(output_root / f"{prefix}_g2p.json"),
        "forced_phone_alignment": str(output_root / f"{prefix}_forced_phone_alignment.json"),
        "ctc_forced_comparison": str(output_root / f"{prefix}_ctc_forced_comparison.json"),
        "provenance": str(output_root / f"{prefix}_provenance.json"),
    }
    if not Path(audio).is_file():
        return _write_failure(
            product_summary_path,
            artifacts,
            "missing_audio",
            [f"missing audio: {audio}"],
        )
    if mfa_alignment is not None and forced_phone_alignment is not None:
        return _write_failure(
            product_summary_path,
            artifacts,
            "invalid_arguments",
            ["--mfa-alignment and --forced-phone-alignment are mutually exclusive"],
        )
    if forced_phone_acceptance is not None and forced_phone_alignment is None and mfa_alignment is None:
        return _write_failure(
            product_summary_path,
            artifacts,
            "invalid_arguments",
            ["--forced-phone-acceptance requires --forced-phone-alignment or --mfa-alignment"],
        )
    if mfa_alignment_manifest is not None and mfa_alignment is None:
        return _write_failure(
            product_summary_path,
            artifacts,
            "invalid_arguments",
            ["--mfa-alignment-manifest requires --mfa-alignment"],
        )
    if mfa_dictionary is not None and mfa_alignment is None:
        return _write_failure(
            product_summary_path,
            artifacts,
            "invalid_arguments",
            ["--mfa-dictionary requires --mfa-alignment"],
        )
    if (forced_model_hash or forced_model_revision) and mfa_alignment is None:
        return _write_failure(
            product_summary_path,
            artifacts,
            "invalid_arguments",
            ["--forced-model-hash/--forced-model-revision are only valid with --mfa-alignment"],
        )
    if bool(forced_model_hash) != bool(forced_model_revision):
        return _write_failure(
            product_summary_path,
            artifacts,
            "invalid_forced_phone_evidence",
            ["MFA alignment requires both --forced-model-hash and --forced-model-revision"],
        )
    if mfa_alignment is not None and mfa_alignment_manifest is None:
        return _write_failure(
            product_summary_path,
            artifacts,
            "invalid_forced_phone_evidence",
            ["MFA product alignment requires an explicit --mfa-alignment-manifest"],
        )
    strict_requested = bool(
        strict_product
        or lyrics_timeline is not None
        or forced_phone_alignment is not None
        or forced_phone_acceptance is not None
        or mfa_alignment is not None
    )
    if strict_requested and (vocal_stem is None or not Path(vocal_stem).is_file()):
        return _write_failure(
            product_summary_path,
            artifacts,
            "missing_vocal_stem_evidence",
            [f"strict product mode requires an explicit readable vocal stem: {vocal_stem}"],
        )
    asr_payload: dict[str, Any] | None = None
    lyric_source = "user_lyrics" if lyrics and lyrics.strip() else None
    selected_lyrics = lyrics.strip() if lyrics and lyrics.strip() else ""
    if asr_transcript is not None:
        try:
            asr_payload = validate_transcript_payload(
                json.loads(Path(asr_transcript).read_text(encoding="utf-8-sig"))
            )
        except (OSError, json.JSONDecodeError, ASRTranscriptError) as exc:
            return _write_failure(
                product_summary_path,
                artifacts,
                "asr_transcript_error",
                [str(exc)],
            )
        asr_payload["source_transcript_sha256"] = _sha256_file(Path(asr_transcript))
        asr_path.write_text(dumps_transcript_json(asr_payload), encoding="utf-8")
        if not selected_lyrics:
            selected_lyrics = str(asr_payload.get("text", "")).strip()
            lyric_source = "external_asr_transcript"
    elif fake_asr:
        fake_text = str(context).strip()
        if not selected_lyrics:
            selected_lyrics = fake_text
            lyric_source = "qwen3_asr_text_evidence"
        try:
            asr_payload = validate_transcript_payload(
                {
                    "schema_version": "asr_transcript_v1",
                    "text": fake_text,
                    "model_output": False,
                    "segments": [],
                    "provenance": {
                        "truth_layer": "asr_evidence_not_phone_truth",
                        "adapter": "fixture",
                        "timestamps_requested": bool(timestamps),
                    },
                }
            )
        except ASRTranscriptError as exc:
            return _write_failure(
                product_summary_path,
                artifacts,
                "asr_transcript_error",
                [str(exc)],
            )
        asr_path.write_text(dumps_transcript_json(asr_payload), encoding="utf-8")
    elif _optional_text(asr_model):
        try:
            model_ref = _optional_text(asr_model)
            adapter = Qwen3ASRAdapter(model_name_or_path=model_ref)
            asr_payload = adapter.transcribe(
                Qwen3ASRRequest(
                    audio_path=Path(audio),
                    model=model_ref,
                    language=language,
                    context=context or "",
                    timestamps=bool(timestamps),
                )
            )
        except Qwen3ASRLoadError as exc:
            return _write_failure(
                product_summary_path,
                artifacts,
                "asr_model_unavailable",
                [str(exc)],
            )
        except ASRTranscriptError as exc:
            return _write_failure(
                product_summary_path,
                artifacts,
                "asr_transcript_error",
                [str(exc)],
            )
        asr_payload = validate_transcript_payload(_mark_model_output(asr_payload))
        asr_path.write_text(dumps_transcript_json(asr_payload), encoding="utf-8")
        if not selected_lyrics:
            selected_lyrics = str(asr_payload.get("text", "")).strip()
            lyric_source = "qwen3_asr_text_evidence"
    if not selected_lyrics and not (strict_requested and lyrics_timeline is not None):
        return _write_failure(
            product_summary_path,
            artifacts,
            "missing_lyrics_evidence",
            ["no lyrics or ASR evidence was provided"],
        )
    timeline = None
    resolved_items = []
    g2p_summary: dict[str, Any] = {"provided": False}
    if strict_requested:
        try:
            if lyrics_timeline is not None:
                timeline = load_lyrics_timeline(Path(lyrics_timeline), audio_path=Path(audio))
            elif asr_payload is not None:
                timeline_source = "external_asr" if asr_transcript is not None else "qwen3_asr"
                timeline = timeline_from_asr_transcript(asr_payload, audio_path=Path(audio), source=timeline_source)
            else:
                raise LyricsTimelineError(
                    "missing_lyrics_timing_evidence",
                    "strict product mode requires lyrics_timeline_v1 or timestamped explicit ASR output",
                )
        except LyricsTimelineError as exc:
            return _write_failure(product_summary_path, artifacts, exc.error_kind, [str(exc)])
        if selected_lyrics and "".join(selected_lyrics.split()) != timeline.text:
            return _write_failure(
                product_summary_path,
                artifacts,
                "lyrics_timeline_text_mismatch",
                ["selected lyrics do not match the timed lyric items"],
            )
        selected_lyrics = timeline.text
        lyric_source = timeline.source
        Path(artifacts["lyrics_timeline"]).write_text(
            json.dumps(timeline.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        try:
            g2p_adapter = (
                ReviewedLexiconCantoneseG2PAdapter(g2p_lexicon)
                if g2p_lexicon is not None
                else PyCantoneseG2PAdapter.for_timeline(timeline)
            )
            resolved_items, g2p_summary = resolve_lyrics_timeline(timeline, adapter=g2p_adapter)
        except (CantoneseG2PError, OSError, ValueError) as exc:
            kind = exc.error_kind if isinstance(exc, CantoneseG2PError) else "unsupported_lyrics_evidence"
            details = exc.details if isinstance(exc, CantoneseG2PError) else {}
            return _write_failure(product_summary_path, artifacts, kind, [str(exc)], details=details)
        g2p_payload = {
            "schema_version": "cantonese_g2p_evidence_v1",
            **g2p_summary,
            "audio_sha256": timeline.audio_sha256,
            "lyrics_timeline_sha256": canonical_timeline_sha256(timeline),
            "items": [item.to_dict() for item in resolved_items],
        }
        Path(artifacts["g2p"]).write_text(
            json.dumps(g2p_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        g2p_summary.update(
            {
                "audio_sha256": timeline.audio_sha256,
                "lyrics_timeline_sha256": canonical_timeline_sha256(timeline),
                "artifact_path": artifacts["g2p"],
                "artifact_sha256": sha256_file(Path(artifacts["g2p"])),
            }
        )
        jyutping_override = " ".join(item.candidate.jyutping for item in resolved_items)
    selected_pitch_path = Path(pitch_analysis) if pitch_analysis else None
    if auto_pitch:
        pitch_audio = Path(vocal_stem or audio)
        try:
            pitch_payload = analyze_audio_pitch_notes(pitch_audio)
        except PitchNoteDependencyError as exc:
            return _write_failure(product_summary_path, artifacts, "pitch_dependency_unavailable", [str(exc)])
        except (OSError, RuntimeError, ValueError) as exc:
            return _write_failure(product_summary_path, artifacts, "pitch_analysis_failure", [str(exc)])
        selected_pitch_path = Path(artifacts["pitch_analysis"])
        selected_pitch_path.write_text(
            json.dumps(pitch_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    selected_phone_segments = None
    ctc_segments = None
    phone_summary: dict[str, Any] = {"provided": False}
    ctc_summary: dict[str, Any] = {"provided": False, "role": "auxiliary_only"}
    forced_summary: dict[str, Any] = {"provided": False}
    if strict_requested:
        if forced_phone_alignment is None and mfa_alignment is None:
            return _write_failure(
                product_summary_path,
                artifacts,
                "missing_phone_evidence",
                ["strict product mode requires explicit forced phone alignment evidence"],
            )
        assert timeline is not None and vocal_stem is not None
        try:
            if mfa_alignment is not None:
                if not forced_model_hash or not forced_model_revision:
                    raise ForcedPhoneAlignerError(
                        "invalid_forced_phone_evidence",
                        "MFA alignment requires explicit forced_model_hash and forced_model_revision",
                    )
                if mfa_dictionary is None:
                    raise ForcedPhoneAlignerError(
                        "invalid_forced_phone_evidence",
                        "MFA product alignment requires an explicit --mfa-dictionary",
                    )
                try:
                    mfa_aligner = MfaForcedPhoneAligner(
                        mfa_alignment,
                        model_hash=forced_model_hash,
                        model_revision=forced_model_revision,
                        manifest=mfa_alignment_manifest,
                        trusted=True,
                        dictionary_path=mfa_dictionary,
                    )
                except TypeError as exc:
                    # Keep the CLI failure contract stable while an older
                    # adapter is present in an installed environment.
                    raise ForcedPhoneAlignerError(
                        "invalid_forced_phone_evidence",
                        "MFA adapter does not support trusted alignment manifests",
                    ) from exc
                forced_result = mfa_aligner.align(Path(vocal_stem), timeline, resolved_items)
            else:
                assert forced_phone_alignment is not None
                forced_result = FixtureForcedPhoneAligner(forced_phone_alignment).align(
                    Path(vocal_stem), timeline, resolved_items
                )
            if forced_phone_acceptance is None:
                raise ForcedPhoneAcceptanceError(
                    "strict product mode requires detached forced-phone acceptance evidence "
                    "for a qualified Cantonese-singing model or approved human phone review",
                    details={
                        "qualified_model_route": False,
                        "human_review_route": False,
                    },
                )
            alignment_input = Path(mfa_alignment or forced_phone_alignment)
            acceptance = load_forced_phone_acceptance(
                Path(forced_phone_acceptance),
                alignment_path=alignment_input,
                vocal_stem=Path(vocal_stem),
                model_hash=str(forced_result.metadata.get("model_hash") or ""),
                segments=forced_result.segments,
            )
            if mfa_alignment is not None and acceptance.get("route") == QUALIFIED_MODEL_ROUTE:
                raise ForcedPhoneAcceptanceError(
                    "automatic MFA alignment and manifest evidence cannot qualify the qualified "
                    "Cantonese-singing model route; use a genuinely human-reviewed alignment "
                    "with detached human acceptance",
                    details={
                        "qualified_model_route": False,
                        "human_review_route": False,
                    },
                )
        except (ForcedPhoneAlignerError, ForcedPhoneAcceptanceError) as exc:
            return _write_failure(
                product_summary_path, artifacts, exc.error_kind, [str(exc)], details=exc.details
            )
        selected_phone_segments = list(forced_result.segments)
        forced_summary = {
            "provided": True,
            "source": "mfa_json" if mfa_alignment is not None else "forced_phone_alignment_v1",
            "path": str(mfa_alignment or forced_phone_alignment),
            "acceptance": acceptance,
            **dict(forced_result.metadata),
        }
        Path(artifacts["forced_phone_alignment"]).write_text(
            json.dumps(forced_result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        forced_summary.update(
            {
                "source_artifact_sha256": _sha256_file(Path(mfa_alignment or forced_phone_alignment)),
                "artifact_path": artifacts["forced_phone_alignment"],
                "artifact_sha256": _sha256_file(Path(artifacts["forced_phone_alignment"])),
                "model_revision": forced_result.metadata.get("model", {}).get("revision")
                if isinstance(forced_result.metadata.get("model"), Mapping)
                else None,
            }
        )
        ctc_path = ctc_phone_segments or phone_segments
        if ctc_path is not None:
            try:
                ctc_segments = FixtureUniversalPhoneRecognizer(Path(ctc_path)).recognize(Path(vocal_stem))
            except (OSError, ValueError) as exc:
                return _write_failure(product_summary_path, artifacts, "invalid_phone_evidence", [str(exc)])
            ctc_summary.update({"provided": True, "source": "external_ctc_phone_segments", "path": str(ctc_path)})
        elif auto_phones:
            try:
                recognizer = ConfigurableUniversalPhoneRecognizer(
                    RecognizerConfig(
                        mode="real",
                        real_backend="onnx-espeak",
                        model_path=Path(recognizer_model) if recognizer_model else None,
                        device=recognizer_device,
                        allow_stub_fallback=False,
                    )
                )
                ctc_segments = recognizer.recognize(Path(vocal_stem))
            except (OSError, RuntimeError, ValueError) as exc:
                return _write_failure(product_summary_path, artifacts, "phone_recognizer_failure", [str(exc)])
            ctc_summary.update({"provided": True, "source": "onnx-espeak", "device": recognizer_device})
            _merge_recognizer_diagnostics(ctc_summary, ctc_segments)
    elif phone_segments is not None:
        try:
            recognizer = FixtureUniversalPhoneRecognizer(Path(phone_segments))
            selected_phone_segments = recognizer.recognize(Path(audio))
        except (OSError, ValueError) as exc:
            return _write_failure(product_summary_path, artifacts, "invalid_phone_evidence", [str(exc)])
        phone_summary.update({"provided": True, "source": "external_phone_segments", "path": str(phone_segments)})
    elif auto_phones:
        try:
            recognizer = ConfigurableUniversalPhoneRecognizer(
                RecognizerConfig(
                    mode="real",
                    real_backend="onnx-espeak",
                    model_path=Path(recognizer_model) if recognizer_model else None,
                    device=recognizer_device,
                    allow_stub_fallback=False,
                )
            )
            selected_phone_segments = recognizer.recognize(Path(vocal_stem or audio))
        except (OSError, RuntimeError, ValueError) as exc:
            return _write_failure(product_summary_path, artifacts, "phone_recognizer_failure", [str(exc)])
        phone_summary.update({"provided": True, "source": "onnx-espeak", "device": recognizer_device})
        _merge_recognizer_diagnostics(phone_summary, selected_phone_segments)
    if strict_requested:
        phone_summary = {"provided": True, "source": "forced_phone_alignment", **forced_summary}
        if ctc_segments:
            ConfigurablePhoneNormalizer(SRC_ROOT / "any2xsampa" / "config" / "ipa_aliases.json").normalize(ctc_segments)
            ctc_payload = {
                "schema_version": "phone_segment_evidence_v1",
                "audio_path": str(vocal_stem),
                "audio_sha256": sha256_file(Path(vocal_stem)),
                "segments": [segment.to_dict() for segment in ctc_segments],
            }
            Path(artifacts["ctc_phone_segments"]).write_text(
                json.dumps(ctc_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            comparison = compare_forced_and_ctc_alignment(selected_phone_segments or [], ctc_segments)
            Path(artifacts["ctc_forced_comparison"]).write_text(
                json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            ctc_summary.update(
                {
                    "segment_count": len(ctc_segments),
                    "comparison": comparison,
                    "comparison_schema_version": "ctc_forced_alignment_comparison_v1",
                    "comparison_artifact_path": artifacts["ctc_forced_comparison"],
                    "comparison_artifact_sha256": _sha256_file(Path(artifacts["ctc_forced_comparison"])),
                }
            )
    if selected_phone_segments and not strict_requested:
        ConfigurablePhoneNormalizer(SRC_ROOT / "any2xsampa" / "config" / "ipa_aliases.json").normalize(
            selected_phone_segments
        )
    if selected_phone_segments:
        phone_payload = {
            "schema_version": "phone_segment_evidence_v1",
            "audio_path": str(vocal_stem or audio),
            "audio_sha256": _sha256_file(Path(vocal_stem or audio)),
            "segments": [segment.to_dict() for segment in selected_phone_segments],
        }
        Path(artifacts["phone_segments"]).write_text(
            json.dumps(phone_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        phone_summary.update(
            {
                "segment_count": len(selected_phone_segments),
                "sha256": _sha256_file(Path(artifacts["phone_segments"])),
                "artifact_path": artifacts["phone_segments"],
                "artifact_sha256": _sha256_file(Path(artifacts["phone_segments"])),
            }
        )
    product_evidence_requested = bool(strict_requested or auto_pitch or auto_phones or phone_segments is not None)
    rc, draft_summary = run_draft_vsqx(
        lyrics=selected_lyrics,
        template=Path(template),
        output_root=output_root,
        jyutping_override=jyutping_override,
        pitch_analysis=selected_pitch_path,
        pitches=pitches,
        tempo=tempo,
        prefix=prefix,
        pretty=pretty,
        phone_segments=selected_phone_segments,
        lyric_source=lyric_source or "unknown_lyric_evidence",
        require_evidence_gate=product_evidence_requested,
    )
    if rc != 0:
        # A failed gate/export must never leave a stale or provisional VSQX
        # from an earlier invocation at the advertised artifact path.
        _remove_product_outputs(artifacts)
    pitch_summary = dict(draft_summary.get("pitch_evidence", {}))
    pitch_audio_path = Path(vocal_stem or audio)
    pitch_summary.update(
        {
            "pitch_audio": str(pitch_audio_path),
            "pitch_audio_sha256": _sha256_file(pitch_audio_path),
            "pitch_audio_role": "vocal_stem_or_pitch_dedicated_audio" if vocal_stem else "full_mix_audio",
        }
    )
    if selected_pitch_path is not None and selected_pitch_path.is_file():
        pitch_summary["pitch_artifact_sha256"] = _sha256_file(selected_pitch_path)
    result = {
        "schema_version": SCHEMA_VERSION,
        "ok": rc == 0 and bool(draft_summary.get("ok")),
        "error_kind": draft_summary.get("error_kind"),
        "errors": list(draft_summary.get("errors", [])),
        "lyrics": selected_lyrics,
        "lyric_source": lyric_source,
        "model_output": False,
        "training_data": False,
        "user_jyutping_required": False,
        "manual_vocaloid_import_required": True,
        "asr_evidence": {
            "provided": asr_payload is not None,
            "model_output": _model_output_marker(asr_payload),
            "timestamps_requested": _timestamp_marker(asr_payload, fallback=timestamps),
            "truth_layer": (asr_payload.get("provenance", {}).get("truth_layer") if asr_payload else None),
        },
        "pitch_evidence": pitch_summary,
        "phone_evidence": phone_summary,
        "forced_alignment_evidence": forced_summary,
        "ctc_evidence": ctc_summary,
        "lyrics_timeline_evidence": {
            "provided": timeline is not None,
            "source": timeline.source if timeline else None,
            "item_count": len(timeline.items) if timeline else 0,
            "audio_sha256": timeline.audio_sha256 if timeline else None,
            "timeline_sha256": canonical_timeline_sha256(timeline) if timeline else None,
        },
        "g2p_evidence": g2p_summary,
        "product_mode": "forced_alignment_product" if strict_requested else "legacy_draft",
        "acceptance_boundary": {"prefer_vocal_stem_for_pitch": True, "phones_ipa_remain_truth": True},
        "draft_vsqx": draft_summary,
        "artifacts": {**artifacts, "draft": draft_summary.get("artifacts", {}).get("draft", artifacts["draft"]), "vsqx": draft_summary.get("artifacts", {}).get("vsqx", artifacts["vsqx"])},
    }
    provenance = {
        "schema_version": "miku_v4c_audio_evidence_provenance_v1",
        "audio": {"path": str(audio), "sha256": _sha256_file(Path(audio))},
        "vocal_stem": (
            {"path": str(vocal_stem), "sha256": sha256_file(Path(vocal_stem))}
            if vocal_stem is not None and Path(vocal_stem).is_file() else None
        ),
        "product_mode": "forced_alignment_product" if strict_requested else "legacy_draft",
        "lyric_source": lyric_source,
        "asr_evidence": result["asr_evidence"],
        "pitch_evidence": pitch_summary,
        "phone_evidence": phone_summary,
        "forced_alignment_evidence": forced_summary,
        "ctc_evidence": ctc_summary,
        "lyrics_timeline_evidence": result["lyrics_timeline_evidence"],
        "g2p_evidence": g2p_summary,
        "evidence_source_priority": ["user_lyrics", "external_asr", "qwen3_asr"],
        "evidence_fusion": draft_summary.get("evidence_fusion"),
        "manual_vocaloid4_import_required": True,
        "manual_vocaloid4_playback_required": True,
        "manual_acceptance_status": "pending_user_validation",
        "xsampa_role": "export_only_not_internal_truth",
    }
    if result["ok"]:
        Path(artifacts["provenance"]).write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    product_summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rc, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lyrics")
    parser.add_argument("--jyutping-override")
    parser.add_argument("--asr-transcript", type=Path)
    parser.add_argument("--asr-model")
    parser.add_argument("--fake-asr", action="store_true")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--context", default="")
    parser.add_argument("--timestamps", action="store_true")
    parser.add_argument("--vocal-stem", type=Path)
    parser.add_argument("--lyrics-timeline", type=Path)
    parser.add_argument("--forced-phone-alignment", type=Path)
    parser.add_argument("--forced-phone-acceptance", type=Path)
    parser.add_argument("--mfa-alignment", type=Path)
    parser.add_argument("--mfa-alignment-manifest", type=Path)
    parser.add_argument("--mfa-dictionary", type=Path)
    parser.add_argument("--forced-model-hash")
    parser.add_argument("--forced-model-revision")
    parser.add_argument("--ctc-phone-segments", type=Path)
    parser.add_argument("--g2p-lexicon", type=Path)
    parser.add_argument("--strict-product", action="store_true")
    parser.add_argument("--pitch-analysis", type=Path)
    parser.add_argument("--auto-pitch", action="store_true")
    parser.add_argument("--phone-segments", type=Path)
    parser.add_argument("--auto-phones", action="store_true")
    parser.add_argument("--recognizer-model", type=Path)
    parser.add_argument("--recognizer-device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--pitches")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--prefix", default="miku_v4c_audio_draft")
    args = parser.parse_args(argv)
    pitches = [int(value) for value in args.pitches.split(",")] if args.pitches else None
    rc, result = run_audio_draft_vsqx(
        audio=args.audio,
        template=args.template,
        output_root=args.output_root,
        lyrics=args.lyrics,
        jyutping_override=args.jyutping_override,
        asr_transcript=args.asr_transcript,
        fake_asr=args.fake_asr,
        asr_model=args.asr_model,
        language=args.language,
        context=args.context,
        timestamps=args.timestamps,
        vocal_stem=args.vocal_stem,
        lyrics_timeline=args.lyrics_timeline,
        forced_phone_alignment=args.forced_phone_alignment,
        forced_phone_acceptance=args.forced_phone_acceptance,
        mfa_alignment=args.mfa_alignment,
        mfa_alignment_manifest=args.mfa_alignment_manifest,
        mfa_dictionary=args.mfa_dictionary,
        forced_model_hash=args.forced_model_hash,
        forced_model_revision=args.forced_model_revision,
        ctc_phone_segments=args.ctc_phone_segments,
        g2p_lexicon=args.g2p_lexicon,
        strict_product=args.strict_product,
        pitch_analysis=args.pitch_analysis,
        auto_pitch=args.auto_pitch,
        phone_segments=args.phone_segments,
        auto_phones=args.auto_phones,
        recognizer_model=args.recognizer_model,
        recognizer_device=args.recognizer_device,
        pitches=pitches,
        prefix=args.prefix,
        pretty=args.pretty,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return rc


def _write_failure(
    product_summary_path: Path,
    artifacts: Mapping[str, str],
    error_kind: str,
    errors: list[str],
    *,
    details: Mapping[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    _remove_product_outputs(artifacts)
    result = {
        "schema_version": SCHEMA_VERSION,
        "ok": False,
        "error_kind": error_kind,
        "errors": list(errors),
        "artifacts": dict(artifacts),
    }
    if details:
        result["details"] = dict(details)
    product_summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 2, result


def _remove_product_outputs(artifacts: Mapping[str, str]) -> None:
    """Remove product artifacts that could make a failed run look successful."""

    for raw_path in artifacts.values():
        if not raw_path:
            continue
        try:
            Path(raw_path).unlink(missing_ok=True)
        except OSError:
            # Preserve the structured failure; a stale output is still visible
            # through the artifact path for callers to diagnose.
            continue


def _merge_recognizer_diagnostics(summary: dict[str, Any], segments: object) -> None:
    """Expose model/runtime hashes emitted by the real recognizer adapter."""

    if not isinstance(segments, list):
        return
    for segment in segments:
        metadata = getattr(segment, "metadata", None)
        diagnostics = metadata.get("adapter_diagnostics") if isinstance(metadata, Mapping) else None
        if not isinstance(diagnostics, Mapping):
            continue
        for field in (
            "model_sha256",
            "vocab_sha256",
            "manifest_path",
            "manifest_verified",
            "providers",
            "session_providers",
            "provider_fallback_reason",
        ):
            if field in diagnostics and diagnostics[field] is not None:
                summary.setdefault(field, diagnostics[field])
        break


def _mark_model_output(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["model_output"] = True
    model = result.get("model")
    if isinstance(model, Mapping):
        model_payload = dict(model)
        model_payload["model_output"] = True
        result["model"] = model_payload
    return result


def _model_output_marker(payload: Mapping[str, Any] | None) -> bool:
    if not payload:
        return False
    if "model_output" in payload:
        return bool(payload.get("model_output"))
    model = payload.get("model")
    return bool(model.get("model_output", False)) if isinstance(model, Mapping) else False


def _timestamp_marker(payload: Mapping[str, Any] | None, *, fallback: bool) -> bool:
    if not payload:
        return bool(fallback)
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping) and "timestamps_requested" in provenance:
        return bool(provenance.get("timestamps_requested"))
    return bool(fallback)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _valid_prefix(value: object) -> bool:
    text = str(value or "").strip()
    return bool(text) and Path(text).name == text and text not in {".", ".."}


def _valid_prefix(value: object) -> bool:
    text = str(value or "").strip()
    return bool(text) and Path(text).name == text and text not in {".", ".."}


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
