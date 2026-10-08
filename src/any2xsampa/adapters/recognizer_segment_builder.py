from __future__ import annotations

from typing import Any, Callable

from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange


AcousticEvidenceAnnotator = Callable[
    [dict[str, Any], list[PhoneCandidate], str, float, float],
    dict[str, Any],
]
CandidateInferrer = Callable[[dict[str, Any]], list[PhoneCandidate]]
FeatureProvider = Callable[[str], dict[str, str]]


def annotate_segment_acoustic_evidence(
    metadata: dict[str, Any],
    candidates: list[PhoneCandidate],
    *,
    source_mode: str,
    start_sec: float,
    end_sec: float,
    annotate_acoustic_evidence: AcousticEvidenceAnnotator,
) -> dict[str, Any]:
    return annotate_acoustic_evidence(
        metadata=metadata,
        candidates=candidates,
        source_mode=source_mode,
        start_sec=start_sec,
        end_sec=end_sec,
    )


def build_segment_from_payload(
    segment_payload: dict[str, Any],
    *,
    source_mode: str,
    index: int,
    infer_candidates_from_metadata: CandidateInferrer,
    default_articulatory_features: FeatureProvider,
    annotate_acoustic_evidence: AcousticEvidenceAnnotator,
) -> PhoneSegment:
    candidates_payload = list(segment_payload.get("candidate_phones", []))
    if not candidates_payload:
        inferred_candidates = infer_candidates_from_metadata(dict(segment_payload.get("metadata", {})))
        candidates_payload = [
            {"symbol": candidate.symbol, "confidence": candidate.confidence}
            for candidate in inferred_candidates
        ]
    if not candidates_payload:
        raise ValueError(f"Recognizer segment {index} is missing candidate_phones.")

    candidates = [
        PhoneCandidate(symbol=str(candidate["symbol"]), confidence=float(candidate["confidence"]))
        for candidate in candidates_payload
    ]
    candidates.sort(key=lambda candidate: -candidate.confidence)
    top_candidate = candidates[0]
    metadata = dict(segment_payload.get("metadata", {}))
    metadata["source_mode"] = source_mode
    metadata["raw_candidates"] = [candidate.symbol for candidate in candidates]
    metadata.setdefault("normalization_notes", [])
    metadata.setdefault("articulatory_features", default_articulatory_features(top_candidate.symbol))
    start_sec = float(segment_payload["start_sec"])
    end_sec = float(segment_payload["end_sec"])
    metadata = annotate_segment_acoustic_evidence(
        metadata,
        candidates,
        source_mode=source_mode,
        start_sec=start_sec,
        end_sec=end_sec,
        annotate_acoustic_evidence=annotate_acoustic_evidence,
    )
    confidence_value = segment_payload.get("confidence", top_candidate.confidence)
    if confidence_value is None or isinstance(confidence_value, bool):
        confidence_value = top_candidate.confidence

    return PhoneSegment(
        time_range=TimeRange(start_sec=start_sec, end_sec=end_sec),
        candidate_phones=candidates,
        confidence=float(confidence_value),
        metadata=metadata,
    )


def build_segments_from_payload(
    payload: dict[str, Any],
    *,
    source_mode: str,
    infer_candidates_from_metadata: CandidateInferrer,
    default_articulatory_features: FeatureProvider,
    annotate_acoustic_evidence: AcousticEvidenceAnnotator,
) -> list[PhoneSegment]:
    segments_payload = payload.get("segments")
    if not isinstance(segments_payload, list) or not segments_payload:
        raise ValueError("Recognizer payload must contain a non-empty 'segments' list.")
    return [
        build_segment_from_payload(
            segment_payload,
            source_mode=source_mode,
            index=index,
            infer_candidates_from_metadata=infer_candidates_from_metadata,
            default_articulatory_features=default_articulatory_features,
            annotate_acoustic_evidence=annotate_acoustic_evidence,
        )
        for index, segment_payload in enumerate(segments_payload)
    ]
