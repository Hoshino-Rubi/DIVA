from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from any2xsampa.adapters.recognizer_support import (
    _normalize_symbol_scores_frame,
    _safe_mean,
    _softmax_symbol_scores,
    _sorted_symbol_scores,
)
from any2xsampa.core.models import PhoneCandidate


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalized_path_text(value: object) -> str | None:
    if value in {None, ""}:
        return None
    return str(value).replace("\\", "/").rstrip("/")


def _shares_external_sibling_root(value: object, *, repo_root: Path) -> bool:
    """Identify an absolute lineage path under the repo's enclosing workspace."""
    lineage_text = _normalized_path_text(value)
    repo_text = _normalized_path_text(str(repo_root))
    if not lineage_text or not repo_text:
        return False
    lineage_parts = [part.casefold() for part in lineage_text.split("/") if part]
    repo_parts = [part.casefold() for part in repo_text.split("/") if part]
    common_length = 0
    for lineage_part, repo_part in zip(lineage_parts, repo_parts):
        if lineage_part != repo_part:
            break
        common_length += 1
    return (
        common_length >= 2
        and common_length < len(lineage_parts)
        and common_length < len(repo_parts)
    )


def _path_within_repo(value: object, *, repo_root: Path) -> bool | None:
    if value in {None, ""}:
        return None
    try:
        resolved = Path(str(value)).resolve(strict=False)
        resolved.relative_to(repo_root)
        return True
    except Exception:
        return False


def _resolve_payload_path_now(value: object, *, repo_root: Path) -> Path | None:
    if value in {None, ""}:
        return None
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve(strict=False)


def _file_sha1_now(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wav_duration_sec_now(path: Path | None) -> float | None:
    if path is None or not path.exists() or path.suffix.lower() != ".wav":
        return None
    try:
        import wave

        with wave.open(str(path), "rb") as handle:
            frame_rate = handle.getframerate()
            if frame_rate <= 0:
                return None
            return round(handle.getnframes() / frame_rate, 3)
    except Exception:
        return None


def _snapshot_id_from_local_path(value: object) -> str | None:
    normalized = _normalized_path_text(value)
    if not normalized:
        return None
    marker = "/snapshots/"
    if marker not in normalized:
        return None
    snapshot_suffix = normalized.split(marker, 1)[1]
    snapshot_id = snapshot_suffix.split("/", 1)[0].strip()
    return snapshot_id or None


def _prediction_candidate_payload(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    posterior_candidates = prediction.get("posterior_candidates")
    if isinstance(posterior_candidates, list) and posterior_candidates:
        return posterior_candidates
    candidates = prediction.get("candidates")
    if isinstance(candidates, list) and candidates:
        payload: list[dict[str, Any]] = []
        for candidate in candidates:
            if isinstance(candidate, PhoneCandidate):
                payload.append({"symbol": candidate.symbol, "confidence": candidate.confidence})
            elif isinstance(candidate, dict):
                symbol = str(candidate.get("symbol", "")).strip()
                if not symbol:
                    continue
                payload.append(
                    {
                        "symbol": symbol,
                        "confidence": float(candidate.get("confidence", 0.0)),
                    }
                )
        if payload:
            return payload
    logit_candidates = prediction.get("logit_candidates")
    if isinstance(logit_candidates, list) and logit_candidates:
        return logit_candidates
    return []


def _prediction_symbol_scores(prediction: dict[str, Any]) -> dict[str, float]:
    candidate_payload = _prediction_candidate_payload(prediction)
    score_key = "confidence"
    if candidate_payload and isinstance(candidate_payload[0], dict) and "logit" in candidate_payload[0]:
        score_key = "logit"
    scores = _normalize_symbol_scores_frame(candidate_payload, score_key=score_key)
    if score_key == "logit" and scores:
        return _softmax_symbol_scores(scores)
    return scores


def prediction_candidate_payload(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    return list(_prediction_candidate_payload(prediction))


def prediction_symbol_scores(prediction: dict[str, Any]) -> dict[str, float]:
    return dict(_prediction_symbol_scores(prediction))


def _grouping_symbol_runs(symbols: list[str | None]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    current_symbol: str | None = None
    start_index = 0
    for index, symbol in enumerate(symbols):
        if current_symbol is None:
            current_symbol = symbol
            start_index = index
            continue
        if symbol != current_symbol:
            runs.append(
                {
                    "symbol": current_symbol,
                    "start_index": start_index,
                    "end_index": index - 1,
                    "frame_count": index - start_index,
                }
            )
            current_symbol = symbol
            start_index = index
    if current_symbol is not None:
        runs.append(
            {
                "symbol": current_symbol,
                "start_index": start_index,
                "end_index": len(symbols) - 1,
                "frame_count": len(symbols) - start_index,
            }
        )
    return runs


def run_summary_from_predictions(
    predictions: list[dict[str, Any]],
    symbols: list[str | None],
) -> list[dict[str, Any]]:
    runs = _grouping_symbol_runs(symbols)
    summary: list[dict[str, Any]] = []
    for run in runs:
        start_index = int(run["start_index"])
        end_index = int(run["end_index"])
        start_sec = predictions[start_index].get("start_sec")
        end_sec = predictions[end_index].get("end_sec")
        duration_sec = None
        if isinstance(start_sec, (int, float)) and isinstance(end_sec, (int, float)):
            duration_sec = round(max(0.0, float(end_sec) - float(start_sec)), 3)
        summary.append(
            {
                "symbol": run.get("symbol"),
                "frame_count": int(run["frame_count"]),
                "start_index": start_index,
                "end_index": end_index,
                "start_sec": round(float(start_sec), 3) if isinstance(start_sec, (int, float)) else None,
                "end_sec": round(float(end_sec), 3) if isinstance(end_sec, (int, float)) else None,
                "duration_sec": duration_sec,
            }
        )
    return summary


def prediction_raw_label_candidates(
    prediction: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_labels = prediction.get("raw_top_labels")
    if not isinstance(raw_labels, list):
        return []
    candidates: list[dict[str, Any]] = []
    for item in raw_labels:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        if not label:
            continue
        candidate: dict[str, Any] = {"label": label}
        posterior = item.get("posterior")
        if isinstance(posterior, (int, float)):
            candidate["posterior"] = round(float(posterior), 4)
        logit = item.get("logit")
        if isinstance(logit, (int, float)):
            candidate["logit"] = round(float(logit), 4)
        mapped_symbol = item.get("mapped_symbol")
        if isinstance(mapped_symbol, str) and mapped_symbol:
            candidate["mapped_symbol"] = mapped_symbol
        candidates.append(candidate)
    return candidates


def prediction_mapped_frame_candidates(
    prediction: dict[str, Any],
) -> list[dict[str, Any]]:
    mapped_candidates = prediction.get("mapped_frame_candidates")
    if not isinstance(mapped_candidates, list):
        return []
    normalized_candidates: list[dict[str, Any]] = []
    for item in mapped_candidates:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).strip()
        if not symbol:
            continue
        candidate: dict[str, Any] = {"symbol": symbol}
        posterior = item.get("posterior")
        if isinstance(posterior, (int, float)):
            candidate["posterior"] = round(float(posterior), 6)
        raw_posterior_mass = item.get("raw_posterior_mass")
        if isinstance(raw_posterior_mass, (int, float)):
            candidate["raw_posterior_mass"] = round(float(raw_posterior_mass), 6)
        logit = item.get("logit")
        if isinstance(logit, (int, float)):
            candidate["logit"] = round(float(logit), 6)
        source_labels_payload = item.get("source_labels")
        if isinstance(source_labels_payload, list):
            source_labels: list[dict[str, Any]] = []
            for source_item in source_labels_payload:
                if not isinstance(source_item, dict):
                    continue
                label = str(source_item.get("label", "")).strip()
                if not label:
                    continue
                source_label: dict[str, Any] = {"label": label}
                source_posterior = source_item.get("posterior")
                if isinstance(source_posterior, (int, float)):
                    source_label["posterior"] = round(float(source_posterior), 6)
                source_logit = source_item.get("logit")
                if isinstance(source_logit, (int, float)):
                    source_label["logit"] = round(float(source_logit), 6)
                source_labels.append(source_label)
            if source_labels:
                candidate["source_labels"] = source_labels
        normalized_candidates.append(candidate)
    return normalized_candidates


def prediction_records_from_posterior_payload(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    segments_payload = payload.get("segments")
    if not isinstance(segments_payload, list) or not segments_payload:
        return [], "posterior-json"

    resolved_backend_name = "posterior-json"
    producer_payload = payload.get("producer")
    if isinstance(producer_payload, dict):
        resolved_backend_name = str(producer_payload.get("backend_kind", "")) or resolved_backend_name

    predictions: list[dict[str, Any]] = []
    for source_segment_index, segment in enumerate(segments_payload):
        if not isinstance(segment, dict):
            continue
        segment_start = float(segment.get("start_sec", 0.0))
        segment_end = float(segment.get("end_sec", segment_start))
        metadata = segment.get("metadata") if isinstance(segment.get("metadata"), dict) else {}
        producer_raw = (
            metadata.get("producer_raw_model_output")
            if isinstance(metadata.get("producer_raw_model_output"), dict)
            else {}
        )
        raw_label_trace_payload = segment.get("raw_label_trace")
        if not isinstance(raw_label_trace_payload, dict):
            raw_label_trace_payload = (
                metadata.get("raw_label_trace")
                if isinstance(metadata.get("raw_label_trace"), dict)
                else {}
            )
        if not isinstance(raw_label_trace_payload, dict):
            raw_label_trace_payload = {}
        if not raw_label_trace_payload:
            raw_label_trace_payload = (
                producer_raw.get("raw_label_trace")
                if isinstance(producer_raw.get("raw_label_trace"), dict)
                else {}
            )
        mapped_frame_trace_payload = segment.get("mapped_frame_trace")
        if not isinstance(mapped_frame_trace_payload, dict):
            mapped_frame_trace_payload = (
                metadata.get("mapped_frame_trace")
                if isinstance(metadata.get("mapped_frame_trace"), dict)
                else {}
            )
        if not mapped_frame_trace_payload:
            mapped_frame_trace_payload = (
                producer_raw.get("mapped_frame_trace")
                if isinstance(producer_raw.get("mapped_frame_trace"), dict)
                else {}
            )
        trace_payload = segment.get("posterior_trace")
        trace_map = trace_payload if isinstance(trace_payload, dict) else {}
        logit_trace_payload = segment.get("logit_trace")
        logit_trace_map = logit_trace_payload if isinstance(logit_trace_payload, dict) else {}
        frames_payload = trace_map.get("frames") if isinstance(trace_map.get("frames"), list) else None
        logit_frames_payload = (
            logit_trace_map.get("frames") if isinstance(logit_trace_map.get("frames"), list) else None
        )
        raw_label_frames_payload = (
            raw_label_trace_payload.get("frames")
            if isinstance(raw_label_trace_payload.get("frames"), list)
            else None
        )
        mapped_frame_frames_payload = (
            mapped_frame_trace_payload.get("frames")
            if isinstance(mapped_frame_trace_payload.get("frames"), list)
            else None
        )
        if not frames_payload:
            frames_payload = [{"candidates": segment.get("candidate_phones", [])}]
        frame_count = max(1, len(frames_payload))
        span = max(0.0, segment_end - segment_start)
        for frame_index, frame in enumerate(frames_payload):
            if frame_count == 1:
                frame_start = segment_start
                frame_end = segment_end
            else:
                frame_start = segment_start + (span * (frame_index / frame_count))
                frame_end = segment_start + (span * ((frame_index + 1) / frame_count))
            frame_map = frame if isinstance(frame, dict) else {"candidates": frame}
            candidate_payload = frame_map.get("candidates") if isinstance(frame_map.get("candidates"), list) else []
            scores = _normalize_symbol_scores_frame(candidate_payload, score_key="confidence")
            if not scores:
                scores = _normalize_symbol_scores_frame(candidate_payload, score_key="logit")
                if scores:
                    scores = _softmax_symbol_scores(scores)
            ordered = _sorted_symbol_scores(scores)
            logit_candidates: list[dict[str, Any]] = []
            if isinstance(logit_frames_payload, list) and frame_index < len(logit_frames_payload):
                logit_frame = logit_frames_payload[frame_index]
                if isinstance(logit_frame, dict) and isinstance(logit_frame.get("candidates"), list):
                    logit_candidates = [
                        dict(item)
                        for item in logit_frame.get("candidates", [])
                        if isinstance(item, dict)
                    ]
            raw_top_labels: list[dict[str, Any]] = []
            if isinstance(raw_label_frames_payload, list) and frame_index < len(raw_label_frames_payload):
                raw_label_frame = raw_label_frames_payload[frame_index]
                if isinstance(raw_label_frame, dict) and isinstance(raw_label_frame.get("candidates"), list):
                    raw_top_labels = [
                        dict(item)
                        for item in raw_label_frame.get("candidates", [])
                        if isinstance(item, dict)
                    ]
            mapped_frame_candidates: list[dict[str, Any]] = []
            if isinstance(mapped_frame_frames_payload, list) and frame_index < len(mapped_frame_frames_payload):
                mapped_frame = mapped_frame_frames_payload[frame_index]
                if isinstance(mapped_frame, dict) and isinstance(mapped_frame.get("candidates"), list):
                    mapped_frame_candidates = [
                        dict(item)
                        for item in mapped_frame.get("candidates", [])
                        if isinstance(item, dict)
                    ]
            predictions.append(
                {
                    "start_sec": round(frame_start, 4),
                    "end_sec": round(frame_end, 4),
                    "source_segment_index": source_segment_index,
                    "source_frame_index": frame_index,
                    "candidates": [
                        PhoneCandidate(symbol=symbol, confidence=round(score, 3))
                        for symbol, score in ordered[:4]
                    ],
                    "posterior_candidates": [
                        {"symbol": symbol, "confidence": round(score, 3)}
                        for symbol, score in ordered[:4]
                    ],
                    "logit_candidates": logit_candidates,
                    "raw_top_labels": raw_top_labels,
                    "mapped_frame_candidates": mapped_frame_candidates,
                }
            )
    return predictions, resolved_backend_name


def prediction_indices_for_reference_window(
    predictions: list[dict[str, Any]],
    *,
    start_sec: float,
    end_sec: float,
    selection_mode: str = "overlap_or_anchor",
) -> list[int]:
    if not predictions:
        return []
    if selection_mode not in {"overlap_or_anchor", "center_inside_window"}:
        raise ValueError(f"Unsupported reference window selection_mode: {selection_mode}")
    selected: list[int] = []
    target_center = start_sec + ((end_sec - start_sec) / 2.0)
    best_index = 0
    best_overlap = -1.0
    best_center_distance = float("inf")
    for index, prediction in enumerate(predictions):
        prediction_start = float(prediction.get("start_sec", 0.0))
        prediction_end = float(prediction.get("end_sec", prediction_start))
        overlap = max(0.0, min(prediction_end, end_sec) - max(prediction_start, start_sec))
        center = prediction_start + ((prediction_end - prediction_start) / 2.0)
        center_distance = abs(center - target_center)
        if selection_mode == "center_inside_window":
            if start_sec <= center < end_sec or (
                index == len(predictions) - 1 and start_sec <= center <= end_sec
            ):
                selected.append(index)
        elif overlap > 0.0 or (prediction_start <= target_center <= prediction_end):
            selected.append(index)
        if overlap > best_overlap or (overlap == best_overlap and center_distance < best_center_distance):
            best_index = index
            best_overlap = overlap
            best_center_distance = center_distance
    return selected or [best_index]


def producer_runtime_contract_from_posterior_payload(payload: dict[str, Any]) -> dict[str, Any]:
    producer_payload = payload.get("producer") if isinstance(payload.get("producer"), dict) else {}
    runtime_payload = (
        producer_payload.get("onnx_espeak_runtime")
        if isinstance(producer_payload.get("onnx_espeak_runtime"), dict)
        else {}
    )
    runtime_path = runtime_payload.get("runtime_path") if isinstance(runtime_payload.get("runtime_path"), dict) else {}
    external_worker = (
        runtime_payload.get("external_worker")
        if isinstance(runtime_payload.get("external_worker"), dict)
        else {}
    )
    model_identity: dict[str, Any] = {}
    segments_payload = payload.get("segments") if isinstance(payload.get("segments"), list) else []
    serialized_score_hint_note_ids: list[str] = []
    serialized_producer_context_match_note_ids: list[str] = []
    serialized_generation_path_evidence: list[str] = []
    serialized_mapped_frame_lineage_evidence: list[str] = []
    serialized_runtime_lineage_evidence: list[str] = []
    serialized_runtime_signature_evidence: list[str] = []
    serialized_environment_signature_evidence: list[str] = []
    serialized_environment_dependency_version_lineage_evidence: list[str] = []
    serialized_mapped_frame_trace_present = False
    serialized_posterior_trace_present = False

    def _append_unique(target: list[str], label: str) -> None:
        if label not in target:
            target.append(label)

    def _append_unique_note_id(target: list[str], value: object) -> None:
        if value in {None, ""}:
            return
        note_id = str(value)
        if note_id not in target:
            target.append(note_id)

    for segment in segments_payload:
        if not isinstance(segment, dict):
            continue
        metadata = segment.get("metadata") if isinstance(segment.get("metadata"), dict) else {}
        score_hint = metadata.get("score_hint") if isinstance(metadata.get("score_hint"), dict) else {}
        producer_context = (
            metadata.get("producer_context") if isinstance(metadata.get("producer_context"), dict) else {}
        )
        _append_unique_note_id(serialized_score_hint_note_ids, score_hint.get("normalized_note_id"))
        _append_unique_note_id(serialized_producer_context_match_note_ids, producer_context.get("matched_note_id"))
        producer_raw = (
            metadata.get("producer_raw_model_output")
            if isinstance(metadata.get("producer_raw_model_output"), dict)
            else {}
        )
        segment_mapped_frame_trace = (
            segment.get("mapped_frame_trace")
            if isinstance(segment.get("mapped_frame_trace"), dict)
            else {}
        )
        metadata_mapped_frame_trace = (
            metadata.get("mapped_frame_trace")
            if isinstance(metadata.get("mapped_frame_trace"), dict)
            else {}
        )
        producer_mapped_frame_trace = (
            producer_raw.get("mapped_frame_trace")
            if isinstance(producer_raw.get("mapped_frame_trace"), dict)
            else {}
        )
        segment_posterior_trace = (
            segment.get("posterior_trace")
            if isinstance(segment.get("posterior_trace"), dict)
            else {}
        )
        if metadata.get("producer_name") not in {None, ""}:
            _append_unique(serialized_generation_path_evidence, "segment.metadata.producer_name")
        if metadata.get("producer_backend_kind") not in {None, ""}:
            _append_unique(serialized_generation_path_evidence, "segment.metadata.producer_backend_kind")
        if isinstance(metadata.get("producer_context"), dict) and metadata.get("producer_context"):
            _append_unique(serialized_generation_path_evidence, "segment.metadata.producer_context")
        if isinstance(metadata.get("producer_contract_layers"), dict) and metadata.get("producer_contract_layers"):
            _append_unique(serialized_generation_path_evidence, "segment.metadata.producer_contract_layers")
        if isinstance(metadata.get("producer_generated_fields"), list) and metadata.get("producer_generated_fields"):
            _append_unique(serialized_generation_path_evidence, "segment.metadata.producer_generated_fields")
        if producer_raw:
            _append_unique(serialized_generation_path_evidence, "segment.metadata.producer_raw_model_output")
        if isinstance(segment_posterior_trace.get("frames"), list) and segment_posterior_trace.get("frames"):
            serialized_posterior_trace_present = True
            _append_unique(serialized_mapped_frame_lineage_evidence, "segment.posterior_trace")
        if isinstance(segment_mapped_frame_trace.get("frames"), list) and segment_mapped_frame_trace.get("frames"):
            serialized_mapped_frame_trace_present = True
            _append_unique(serialized_mapped_frame_lineage_evidence, "segment.mapped_frame_trace")
        if isinstance(metadata_mapped_frame_trace.get("frames"), list) and metadata_mapped_frame_trace.get("frames"):
            serialized_mapped_frame_trace_present = True
            _append_unique(serialized_mapped_frame_lineage_evidence, "segment.metadata.mapped_frame_trace")
        if isinstance(producer_mapped_frame_trace.get("frames"), list) and producer_mapped_frame_trace.get("frames"):
            serialized_mapped_frame_trace_present = True
            _append_unique(
                serialized_mapped_frame_lineage_evidence,
                "segment.metadata.producer_raw_model_output.mapped_frame_trace",
            )
        for key in ("raw_model_output_kind", "model_backend", "model_repo", "model_file", "vocab_repo"):
            if key in producer_raw and producer_raw.get(key) not in {None, ""}:
                model_identity[key] = producer_raw.get(key)
        if model_identity:
            break

    for key in (
        "name",
        "version",
        "audio_path",
        "note_context_path",
        "note_context_format",
        "dispatch",
        "backend_registry_entry",
        "backend_capability",
        "contract_layers",
        "derived_fields",
        "onnx_espeak_runtime",
        "raw_model_output",
    ):
        value = producer_payload.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, (dict, list)) and not value:
            continue
        _append_unique(serialized_generation_path_evidence, f"producer.{key}")

    current_repo_root = _repo_root().resolve(strict=False)
    current_repo_root_text = _normalized_path_text(str(current_repo_root)) or ""
    in_process_payload = runtime_payload.get("in_process") if isinstance(runtime_payload.get("in_process"), dict) else {}
    primary_bundle_layout = (
        runtime_path.get("primary_bundle_layout")
        if isinstance(runtime_path.get("primary_bundle_layout"), dict)
        else {}
    )
    primary_bundle_status = runtime_path.get("primary_bundle_status")
    primary_bundle_root = primary_bundle_layout.get("runtime_root")
    primary_bundle_root_text = _normalized_path_text(primary_bundle_root)
    primary_bundle_worker_entrypoint = primary_bundle_layout.get("worker_entrypoint")
    primary_bundle_worker_entrypoint_text = _normalized_path_text(primary_bundle_worker_entrypoint)
    primary_bundle_manifest_path = primary_bundle_layout.get("manifest_path")
    runtime_repo_root = runtime_path.get("repo_root")
    runtime_repo_root_text = _normalized_path_text(runtime_repo_root)
    in_process_command = in_process_payload.get("command") if isinstance(in_process_payload.get("command"), list) else None
    actual_in_process_runtime = runtime_payload.get("runtime_status") == "in_process_ready"
    selected_runtime_root = external_worker.get("selected_runtime_root") or runtime_path.get("selected_runtime_root")
    selected_runtime_root_text = _normalized_path_text(selected_runtime_root)
    selected_worker_entrypoint = (
        external_worker.get("selected_worker_entrypoint")
        or runtime_path.get("selected_worker_entrypoint")
    )
    selected_worker_entrypoint_text = _normalized_path_text(selected_worker_entrypoint)
    selected_command = external_worker.get("selected_command") or runtime_path.get("selected_command")
    if selected_command in (None, "") and isinstance(in_process_command, list) and in_process_command:
        selected_command = in_process_command
    runtime_python_version = in_process_payload.get("python_version") or external_worker.get("python_version")
    bundle_status = runtime_payload.get("bundle_status")
    runtime_status = runtime_payload.get("runtime_status")

    for label, present in (
        ("runtime_path.repo_root", runtime_repo_root_text),
        ("runtime_path.selected_runtime_root", selected_runtime_root_text),
        ("runtime_path.selected_worker_entrypoint", selected_worker_entrypoint_text),
        ("runtime_path.primary_bundle_layout.runtime_root", primary_bundle_root_text),
        ("runtime_path.primary_bundle_layout.worker_entrypoint", primary_bundle_worker_entrypoint_text),
        ("runtime_path.primary_bundle_layout.manifest_path", primary_bundle_manifest_path),
        ("runtime_path.selected_command", selected_command if isinstance(selected_command, list) and selected_command else None),
        ("runtime.in_process.command", in_process_command if isinstance(in_process_command, list) and in_process_command else None),
        ("runtime.in_process.python_version", runtime_python_version),
        ("runtime.bundle_status", bundle_status),
        ("runtime.runtime_status", runtime_status),
        ("runtime_path.primary_bundle_status", primary_bundle_status),
    ):
        if present is not None and present != "":
            _append_unique(serialized_runtime_lineage_evidence, label)

    for label, present in (
        ("runtime_path.repo_root", runtime_repo_root_text),
        ("runtime.in_process.command", in_process_command if isinstance(in_process_command, list) and in_process_command else None),
        ("runtime.python_version", runtime_python_version),
        ("runtime_path.primary_bundle_status", primary_bundle_status),
        ("runtime_path.primary_bundle_layout.runtime_root", primary_bundle_root_text),
        ("runtime_path.primary_bundle_layout.worker_entrypoint", primary_bundle_worker_entrypoint_text),
        ("runtime_path.primary_bundle_layout.manifest_path", primary_bundle_manifest_path),
        ("runtime_path.selected_command", selected_command if isinstance(selected_command, list) and selected_command else None),
        ("runtime_path.selected_runtime_root", selected_runtime_root_text),
        ("runtime_path.selected_worker_entrypoint", selected_worker_entrypoint_text),
    ):
        if present is not None and present != "":
            _append_unique(serialized_runtime_signature_evidence, label)

    serialized_runtime_lineage_present = bool(serialized_runtime_lineage_evidence)
    lineage_path_texts = [
        text
        for text in (runtime_repo_root_text, selected_runtime_root_text, selected_worker_entrypoint_text)
        if text
    ]
    if not serialized_runtime_lineage_present:
        serialized_runtime_lineage_kind = "runtime_lineage_missing"
    elif any(text.startswith("/work/repo") for text in lineage_path_texts):
        serialized_runtime_lineage_kind = "historical_linux_repo_runtime_lineage"
    elif any(
        _path_within_repo(text, repo_root=current_repo_root) is True
        for text in lineage_path_texts
    ):
        serialized_runtime_lineage_kind = "repo_local_current_runtime_lineage"
    elif any(
        _shares_external_sibling_root(text, repo_root=current_repo_root)
        for text in lineage_path_texts
    ):
        serialized_runtime_lineage_kind = "repo_external_sibling_runtime_lineage"
    else:
        serialized_runtime_lineage_kind = "external_runtime_lineage"

    serialized_mapped_frame_lineage_present = serialized_mapped_frame_trace_present
    if serialized_mapped_frame_lineage_present:
        serialized_mapped_frame_lineage_kind = "persisted_mapped_frame_trace_lineage"
    elif (
        serialized_runtime_lineage_kind == "historical_linux_repo_runtime_lineage"
        and serialized_posterior_trace_present
    ):
        serialized_mapped_frame_lineage_kind = "historical_exact_mapped_frame_lineage_unrecoverable_from_payload"
        _append_unique(serialized_mapped_frame_lineage_evidence, "historical_linux_repo_runtime_lineage")
        _append_unique(serialized_mapped_frame_lineage_evidence, "posterior_trace_top_k_only")
    elif serialized_posterior_trace_present:
        serialized_mapped_frame_lineage_kind = "posterior_trace_top_k_only_lineage"
        _append_unique(serialized_mapped_frame_lineage_evidence, "posterior_trace_top_k_only")
    else:
        serialized_mapped_frame_lineage_kind = "mapped_frame_lineage_missing"

    serialized_runtime_signature_present = bool(serialized_runtime_signature_evidence)
    if not serialized_runtime_signature_present:
        serialized_runtime_signature_kind = "runtime_signature_missing"
    elif actual_in_process_runtime:
        serialized_runtime_signature_kind = "in_process_bundle_runtime_signature"
    elif isinstance(selected_command, list) and selected_command:
        serialized_runtime_signature_kind = "selected_worker_runtime_signature"
    else:
        serialized_runtime_signature_kind = "path_only_runtime_signature"

    model_resources = (
        runtime_payload.get("model_resources")
        if isinstance(runtime_payload.get("model_resources"), dict)
        else {}
    )
    environment_checked_via = model_resources.get("checked_via")
    environment_checked_command = (
        model_resources.get("checked_with_command")
        if isinstance(model_resources.get("checked_with_command"), list)
        else None
    )
    in_process_dependencies = (
        in_process_payload.get("dependencies")
        if isinstance(in_process_payload.get("dependencies"), dict)
        else {}
    )
    external_worker_dependencies = (
        external_worker.get("dependencies")
        if isinstance(external_worker.get("dependencies"), dict)
        else {}
    )
    environment_dependency_source = None
    environment_python_version = None
    environment_dependencies: dict[str, Any] = {}
    if actual_in_process_runtime and in_process_dependencies:
        environment_dependency_source = "in_process"
        environment_python_version = in_process_payload.get("python_version")
        environment_dependencies = in_process_dependencies
    elif bool(external_worker.get("ready", False)) and external_worker_dependencies:
        environment_dependency_source = "external_worker"
        environment_python_version = external_worker.get("python_version")
        environment_dependencies = external_worker_dependencies
    elif in_process_dependencies:
        environment_dependency_source = "in_process_probe"
        environment_python_version = in_process_payload.get("python_version")
        environment_dependencies = in_process_dependencies
    elif external_worker_dependencies:
        environment_dependency_source = "external_worker_probe"
        environment_python_version = external_worker.get("python_version")
        environment_dependencies = external_worker_dependencies

    external_worker_candidates = (
        external_worker.get("candidates") if isinstance(external_worker.get("candidates"), list) else []
    )
    selected_external_worker_candidate = next(
        (
            candidate
            for candidate in external_worker_candidates
            if isinstance(candidate, dict)
            and bool(candidate.get("ready", False))
        ),
        None,
    )
    if not isinstance(selected_external_worker_candidate, dict):
        selected_external_worker_candidate = next(
            (
                candidate
                for candidate in external_worker_candidates
                if isinstance(candidate, dict)
                and candidate.get("command") == selected_command
            ),
            None,
        )

    if environment_checked_via == "external_worker" and isinstance(selected_external_worker_candidate, dict):
        selected_candidate_dependencies = (
            selected_external_worker_candidate.get("dependencies")
            if isinstance(selected_external_worker_candidate.get("dependencies"), dict)
            else {}
        )
        if selected_candidate_dependencies:
            environment_dependencies = selected_candidate_dependencies
            environment_dependency_source = "external_worker"
            environment_python_version = (
                selected_external_worker_candidate.get("python_version")
                or external_worker.get("python_version")
            )
    elif not environment_dependencies and isinstance(selected_external_worker_candidate, dict):
        selected_candidate_dependencies = (
            selected_external_worker_candidate.get("dependencies")
            if isinstance(selected_external_worker_candidate.get("dependencies"), dict)
            else {}
        )
        if selected_candidate_dependencies:
            environment_dependencies = selected_candidate_dependencies
            if environment_dependency_source is None:
                environment_dependency_source = "external_worker"
            if environment_python_version in {None, ""}:
                environment_python_version = selected_external_worker_candidate.get("python_version")

    if environment_dependency_source is None and environment_checked_via == "external_worker":
        environment_dependency_source = "external_worker"
        if environment_python_version in {None, ""}:
            environment_python_version = external_worker.get("python_version") or (
                selected_external_worker_candidate.get("python_version")
                if isinstance(selected_external_worker_candidate, dict)
                else None
            )
    elif environment_dependency_source is None and environment_checked_via == "in_process":
        environment_dependency_source = "in_process"
        if environment_python_version in {None, ""}:
            environment_python_version = in_process_payload.get("python_version")

    serialized_environment_dependency_availability: dict[str, bool] = {}
    serialized_environment_dependency_versions: dict[str, str | None] = {}
    for dependency_name, dependency_payload in environment_dependencies.items():
        if not isinstance(dependency_payload, dict):
            continue
        normalized_name = str(dependency_name)
        serialized_environment_dependency_availability[normalized_name] = bool(
            dependency_payload.get("available", False)
        )
        version_value = dependency_payload.get("version")
        serialized_environment_dependency_versions[normalized_name] = (
            str(version_value) if version_value not in {None, ""} else None
        )

    serialized_environment_dependency_count = len(serialized_environment_dependency_availability)
    serialized_environment_dependency_available_count = sum(
        1 for value in serialized_environment_dependency_availability.values() if value
    )
    serialized_environment_dependency_versioned_count = sum(
        1 for value in serialized_environment_dependency_versions.values() if value is not None
    )
    serialized_environment_dependencies_missing_versions = sorted(
        dependency_name
        for dependency_name, available in serialized_environment_dependency_availability.items()
        if available and serialized_environment_dependency_versions.get(dependency_name) is None
    )

    environment_local_files = (
        model_resources.get("local_files") if isinstance(model_resources.get("local_files"), list) else []
    )
    environment_resource_roots: list[str] = []
    serialized_environment_resource_snapshot_ids: dict[str, str | None] = {}
    for resource_payload in environment_local_files:
        if not isinstance(resource_payload, dict):
            continue
        local_path = _normalized_path_text(resource_payload.get("local_path"))
        if not local_path:
            continue
        resource_name = str(resource_payload.get("resource", "")).strip()
        if resource_name and resource_name not in serialized_environment_resource_snapshot_ids:
            serialized_environment_resource_snapshot_ids[resource_name] = _snapshot_id_from_local_path(local_path)
        if "/snapshots/" in local_path:
            local_root = local_path.split("/snapshots/", 1)[0]
        else:
            local_root = _normalized_path_text(str(Path(local_path).parent)) or local_path
        if local_root not in environment_resource_roots:
            environment_resource_roots.append(local_root)

    if environment_dependency_source is not None:
        _append_unique(
            serialized_environment_signature_evidence,
            f"runtime.{environment_dependency_source}.dependencies",
        )
        _append_unique(
            serialized_environment_dependency_version_lineage_evidence,
            f"runtime.{environment_dependency_source}.dependencies",
        )
    if environment_python_version not in {None, ""}:
        _append_unique(
            serialized_environment_signature_evidence,
            f"runtime.{environment_dependency_source or 'probe'}.python_version",
        )
        _append_unique(
            serialized_environment_dependency_version_lineage_evidence,
            f"runtime.{environment_dependency_source or 'probe'}.python_version",
        )
    if environment_checked_via not in {None, ""}:
        _append_unique(serialized_environment_signature_evidence, "runtime.model_resources.checked_via")
        _append_unique(
            serialized_environment_dependency_version_lineage_evidence,
            "runtime.model_resources.checked_via",
        )
    if environment_checked_command:
        _append_unique(serialized_environment_signature_evidence, "runtime.model_resources.checked_with_command")
        _append_unique(
            serialized_environment_dependency_version_lineage_evidence,
            "runtime.model_resources.checked_with_command",
        )
    if environment_local_files:
        _append_unique(serialized_environment_signature_evidence, "runtime.model_resources.local_files")
        _append_unique(
            serialized_environment_dependency_version_lineage_evidence,
            "runtime.model_resources.local_files",
        )
    if any(value is not None for value in serialized_environment_resource_snapshot_ids.values()):
        _append_unique(
            serialized_environment_signature_evidence,
            "runtime.model_resources.local_files.snapshot_ids",
        )
        _append_unique(
            serialized_environment_dependency_version_lineage_evidence,
            "runtime.model_resources.local_files.snapshot_ids",
        )
    if primary_bundle_layout.get("cache_dir") not in {None, ""}:
        _append_unique(serialized_environment_signature_evidence, "runtime_path.primary_bundle_layout.cache_dir")
    if primary_bundle_layout.get("model_dir") not in {None, ""}:
        _append_unique(serialized_environment_signature_evidence, "runtime_path.primary_bundle_layout.model_dir")

    serialized_environment_signature_present = bool(serialized_environment_signature_evidence)
    serialized_environment_dependency_versions_complete = (
        bool(serialized_environment_dependency_versions)
        and all(version is not None for version in serialized_environment_dependency_versions.values())
        and environment_python_version not in {None, ""}
        and bool(environment_checked_command)
        and bool(environment_resource_roots)
    )
    normalized_environment_signature_source = environment_dependency_source
    if normalized_environment_signature_source == "in_process_probe":
        normalized_environment_signature_source = "in_process"
    elif normalized_environment_signature_source == "external_worker_probe":
        normalized_environment_signature_source = "external_worker"

    if not serialized_environment_signature_present:
        serialized_environment_signature_kind = "environment_signature_missing"
    elif normalized_environment_signature_source == "in_process":
        serialized_environment_signature_kind = "in_process_dependency_probe_environment_signature"
    elif normalized_environment_signature_source == "external_worker":
        serialized_environment_signature_kind = "external_worker_dependency_probe_environment_signature"
    else:
        serialized_environment_signature_kind = "dependency_probe_environment_signature"

    if not serialized_environment_signature_present:
        serialized_environment_dependency_probe_completeness_kind = "dependency_probe_missing"
    elif (
        serialized_environment_dependency_count > 0
        and serialized_environment_dependency_versioned_count == serialized_environment_dependency_count
        and environment_python_version not in {None, ""}
    ):
        serialized_environment_dependency_probe_completeness_kind = "versioned_dependency_probe"
    elif serialized_environment_dependency_available_count > 0:
        serialized_environment_dependency_probe_completeness_kind = "availability_only_dependency_probe"
    elif serialized_environment_dependency_count > 0:
        serialized_environment_dependency_probe_completeness_kind = "unavailable_dependency_probe"
    else:
        serialized_environment_dependency_probe_completeness_kind = "dependency_probe_without_dependency_rows"

    if serialized_environment_dependency_versions_complete:
        serialized_environment_dependency_version_lineage_kind = "versioned_dependency_probe_lineage"
    elif (
        serialized_runtime_lineage_kind == "historical_linux_repo_runtime_lineage"
        and serialized_environment_dependency_available_count > 0
        and serialized_environment_dependency_versioned_count == 0
    ):
        serialized_environment_dependency_version_lineage_kind = (
            "historical_exact_dependency_versions_unrecoverable_from_payload"
        )
        _append_unique(
            serialized_environment_dependency_version_lineage_evidence,
            "historical_linux_repo_runtime_lineage",
        )
        _append_unique(
            serialized_environment_dependency_version_lineage_evidence,
            "dependencies_available_without_versions",
        )
    elif (
        serialized_environment_dependency_available_count > 0
        and serialized_environment_dependency_versioned_count == 0
    ):
        serialized_environment_dependency_version_lineage_kind = (
            "availability_only_dependency_probe_lineage"
        )
        _append_unique(
            serialized_environment_dependency_version_lineage_evidence,
            "dependencies_available_without_versions",
        )
    elif serialized_environment_dependency_versioned_count > 0:
        serialized_environment_dependency_version_lineage_kind = "partial_dependency_version_lineage"
        _append_unique(
            serialized_environment_dependency_version_lineage_evidence,
            "partial_dependency_versions_present",
        )
    elif serialized_environment_dependency_count > 0:
        serialized_environment_dependency_version_lineage_kind = "dependency_version_lineage_missing"
    else:
        serialized_environment_dependency_version_lineage_kind = "dependency_version_lineage_not_probed"

    resolved_audio_path_now = _resolve_payload_path_now(producer_payload.get("audio_path"), repo_root=current_repo_root)
    resolved_note_context_path_now = _resolve_payload_path_now(
        producer_payload.get("note_context_path"),
        repo_root=current_repo_root,
    )
    serialized_total_end_sec = round(
        max((float(segment.get("end_sec", 0.0)) for segment in segments_payload if isinstance(segment, dict)), default=0.0),
        3,
    )
    serialized_score_hint_segment_count = sum(
        1
        for segment in segments_payload
        if isinstance(segment, dict)
        and isinstance((segment.get("metadata") or {}).get("score_hint"), dict)
    )
    serialized_producer_context_match_segment_count = sum(
        1
        for segment in segments_payload
        if isinstance(segment, dict)
        and isinstance((segment.get("metadata") or {}).get("producer_context"), dict)
        and ((segment.get("metadata") or {}).get("producer_context") or {}).get("matched_note_id") not in {None, ""}
    )
    reference_segment_contract_selection_mode = producer_payload.get("reference_segment_contract_selection_mode")
    if serialized_score_hint_segment_count > 0 and serialized_score_hint_segment_count == len(segments_payload):
        serialized_payload_contract_kind = "score_hint_enriched_runtime_payload"
    elif serialized_score_hint_segment_count > 0:
        serialized_payload_contract_kind = "mixed_runtime_payload"
    else:
        serialized_payload_contract_kind = "raw_producer_payload"

    if producer_payload.get("raw_projection_from_enriched_runtime_payload") is True:
        serialized_payload_origin_kind = "projected_raw_from_enriched_runtime_payload"
    elif producer_payload.get("reference_segment_contract_same_segmentation_projection") is True:
        serialized_payload_origin_kind = "reference_segment_same_segmentation_projection_payload"
    elif producer_payload.get("regenerated_with_reference_segments") is True:
        serialized_payload_origin_kind = "reference_segment_canonicalized_payload"
    elif producer_payload.get("regenerated_with_grouping_repair") is True:
        serialized_payload_origin_kind = "grouping_repaired_replay_payload"
    elif serialized_payload_contract_kind == "score_hint_enriched_runtime_payload":
        serialized_payload_origin_kind = "enriched_runtime_payload"
    elif serialized_payload_contract_kind == "mixed_runtime_payload":
        serialized_payload_origin_kind = "mixed_runtime_payload"
    else:
        serialized_payload_origin_kind = "native_raw_producer_payload"

    serialized_generation_path_present = bool(serialized_generation_path_evidence)
    if producer_payload.get("raw_projection_from_enriched_runtime_payload") is True:
        serialized_generation_path_kind = "projected_from_enriched_runtime_generation_path"
    elif producer_payload.get("reference_segment_contract_same_segmentation_projection") is True:
        serialized_generation_path_kind = "reference_segment_same_segmentation_projection_generation_path"
    elif producer_payload.get("regenerated_with_reference_segments") is True:
        serialized_generation_path_kind = "reference_segment_canonicalized_generation_path"
    elif producer_payload.get("regenerated_with_grouping_repair") is True:
        serialized_generation_path_kind = "grouping_repaired_replay_generation_path"
    elif not serialized_generation_path_present:
        serialized_generation_path_kind = "generation_path_missing"
    elif serialized_payload_contract_kind == "score_hint_enriched_runtime_payload":
        serialized_generation_path_kind = "runtime_enriched_generation_path"
    elif serialized_payload_contract_kind == "mixed_runtime_payload":
        serialized_generation_path_kind = "mixed_runtime_generation_path"
    else:
        serialized_generation_path_kind = "native_raw_producer_generation_path"

    resolved_audio_duration_sec_now = _wav_duration_sec_now(resolved_audio_path_now)
    serialized_coverage_ratio_against_resolved_audio_now = None
    if resolved_audio_duration_sec_now and resolved_audio_duration_sec_now > 0.0:
        serialized_coverage_ratio_against_resolved_audio_now = round(
            serialized_total_end_sec / resolved_audio_duration_sec_now,
            3,
        )

    return {
        "backend_kind": producer_payload.get("backend_kind"),
        "requested_backend_kind": producer_payload.get("requested_backend_kind"),
        "current_repo_root": str(current_repo_root),
        "audio_path": producer_payload.get("audio_path"),
        "note_context_path": producer_payload.get("note_context_path"),
        "note_context_format": producer_payload.get("note_context_format"),
        "lyrics_guided": producer_payload.get("lyrics_guided"),
        "dispatch": {
            "requested_backend_kind": (producer_payload.get("dispatch") or {}).get("requested_backend_kind")
            if isinstance(producer_payload.get("dispatch"), dict)
            else None,
            "resolved_backend_kind": (producer_payload.get("dispatch") or {}).get("resolved_backend_kind")
            if isinstance(producer_payload.get("dispatch"), dict)
            else None,
            "auto_fallback_used": (producer_payload.get("dispatch") or {}).get("auto_fallback_used")
            if isinstance(producer_payload.get("dispatch"), dict)
            else None,
        },
        "onnx_runtime": {
            "bundle_status": runtime_payload.get("bundle_status"),
            "runtime_status": runtime_payload.get("runtime_status"),
            "ready": runtime_payload.get("ready"),
            "model_resources_status": (runtime_payload.get("model_resources") or {}).get("status")
            if isinstance(runtime_payload.get("model_resources"), dict)
            else None,
            "selected_source": external_worker.get("selected_source") or runtime_path.get("selected_runtime_source"),
            "selected_command": selected_command,
            "selected_python_path": external_worker.get("selected_python_path") or runtime_path.get("selected_python_path"),
            "selected_runtime_root": selected_runtime_root,
            "selected_worker_entrypoint": selected_worker_entrypoint,
            "selected_bundle_status": runtime_path.get("selected_bundle_status") or primary_bundle_status,
            "selected_runtime_root_within_repo": _path_within_repo(
                selected_runtime_root,
                repo_root=current_repo_root,
            ),
            "selected_worker_entrypoint_within_repo": _path_within_repo(
                selected_worker_entrypoint,
                repo_root=current_repo_root,
            ),
        },
        "model_identity": model_identity,
        "payload_signature": {
            "segment_count": len(segments_payload),
            "total_end_sec": serialized_total_end_sec,
            "serialized_total_end_sec": serialized_total_end_sec,
            "serialized_segment_count": len(segments_payload),
            "serialized_payload_contract_kind": serialized_payload_contract_kind,
            "serialized_payload_origin_kind": serialized_payload_origin_kind,
            "serialized_generation_path_kind": serialized_generation_path_kind,
            "serialized_generation_path_present": serialized_generation_path_present,
            "serialized_generation_path_evidence": serialized_generation_path_evidence,
            "serialized_mapped_frame_lineage_kind": serialized_mapped_frame_lineage_kind,
            "serialized_mapped_frame_lineage_present": serialized_mapped_frame_lineage_present,
            "serialized_mapped_frame_lineage_evidence": serialized_mapped_frame_lineage_evidence,
            "reference_segment_contract_selection_mode": reference_segment_contract_selection_mode,
            "serialized_runtime_lineage_kind": serialized_runtime_lineage_kind,
            "serialized_runtime_lineage_present": serialized_runtime_lineage_present,
            "serialized_runtime_lineage_evidence": serialized_runtime_lineage_evidence,
            "serialized_runtime_signature_kind": serialized_runtime_signature_kind,
            "serialized_runtime_signature_present": serialized_runtime_signature_present,
            "serialized_runtime_signature_evidence": serialized_runtime_signature_evidence,
            "serialized_environment_signature_kind": serialized_environment_signature_kind,
            "serialized_environment_signature_present": serialized_environment_signature_present,
            "serialized_environment_signature_evidence": serialized_environment_signature_evidence,
            "serialized_environment_checked_via": environment_checked_via,
            "serialized_environment_checked_command": environment_checked_command,
            "serialized_environment_python_version": environment_python_version,
            "serialized_environment_dependency_availability": serialized_environment_dependency_availability,
            "serialized_environment_dependency_versions": serialized_environment_dependency_versions,
            "serialized_environment_dependency_count": serialized_environment_dependency_count,
            "serialized_environment_dependency_available_count": serialized_environment_dependency_available_count,
            "serialized_environment_dependency_versioned_count": serialized_environment_dependency_versioned_count,
            "serialized_environment_dependencies_missing_versions": serialized_environment_dependencies_missing_versions,
            "serialized_environment_dependency_probe_completeness_kind": serialized_environment_dependency_probe_completeness_kind,
            "serialized_environment_dependency_versions_complete": serialized_environment_dependency_versions_complete,
            "serialized_environment_dependency_version_lineage_kind": serialized_environment_dependency_version_lineage_kind,
            "serialized_environment_dependency_version_lineage_evidence": serialized_environment_dependency_version_lineage_evidence,
            "serialized_environment_resource_roots": environment_resource_roots,
            "serialized_environment_resource_snapshot_ids": serialized_environment_resource_snapshot_ids,
            "serialized_runtime_repo_root": runtime_repo_root,
            "serialized_runtime_python_version": runtime_python_version,
            "serialized_runtime_selected_command": selected_command,
            "serialized_runtime_selected_runtime_root": selected_runtime_root,
            "serialized_runtime_selected_worker_entrypoint": selected_worker_entrypoint,
            "serialized_runtime_in_process_command": in_process_command,
            "serialized_runtime_primary_bundle_status": primary_bundle_status,
            "serialized_runtime_primary_bundle_root": primary_bundle_root,
            "serialized_runtime_primary_bundle_worker_entrypoint": primary_bundle_worker_entrypoint,
            "serialized_runtime_primary_bundle_manifest_path": primary_bundle_manifest_path,
            "serialized_runtime_repo_root_within_current_repo": (
                runtime_repo_root_text.startswith(current_repo_root_text)
                if runtime_repo_root_text and current_repo_root_text
                else None
            ),
            "serialized_runtime_selected_runtime_root_within_current_repo": _path_within_repo(
                selected_runtime_root,
                repo_root=current_repo_root,
            ),
            "serialized_runtime_selected_worker_entrypoint_within_current_repo": _path_within_repo(
                selected_worker_entrypoint,
                repo_root=current_repo_root,
            ),
            "serialized_score_hint_segment_count": serialized_score_hint_segment_count,
            "serialized_score_hint_note_ids": serialized_score_hint_note_ids,
            "serialized_producer_context_match_segment_count": serialized_producer_context_match_segment_count,
            "serialized_producer_context_match_note_ids": serialized_producer_context_match_note_ids,
            "resolved_audio_path_now": str(resolved_audio_path_now) if resolved_audio_path_now is not None else None,
            "resolved_audio_exists_now": bool(resolved_audio_path_now and resolved_audio_path_now.exists()),
            "resolved_audio_sha1_now": _file_sha1_now(resolved_audio_path_now),
            "resolved_audio_duration_sec_now": resolved_audio_duration_sec_now,
            "serialized_coverage_ratio_against_resolved_audio_now": serialized_coverage_ratio_against_resolved_audio_now,
            "resolved_note_context_path_now": str(resolved_note_context_path_now) if resolved_note_context_path_now is not None else None,
            "resolved_note_context_exists_now": bool(
                resolved_note_context_path_now and resolved_note_context_path_now.exists()
            ),
            "resolved_note_context_sha1_now": _file_sha1_now(resolved_note_context_path_now),
            "audio_path_within_repo": _path_within_repo(
                producer_payload.get("audio_path"),
                repo_root=current_repo_root,
            ),
            "note_context_path_within_repo": _path_within_repo(
                producer_payload.get("note_context_path"),
                repo_root=current_repo_root,
            ),
        },
    }


def frame_evidence_summary_from_posterior_payload(
    payload: dict[str, Any],
    *,
    anchor_start_sec: float | None = None,
    anchor_end_sec: float | None = None,
    top_k: int = 2,
) -> dict[str, Any]:
    predictions, backend_name = prediction_records_from_posterior_payload(payload)

    def _compact_margin_example(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "frame_index": row.get("frame_index", row.get("global_index")),
            "global_index": row.get("global_index"),
            "source_segment_index": row.get("source_segment_index"),
            "source_frame_index": row.get("source_frame_index"),
            "start_sec": row.get("start_sec"),
            "end_sec": row.get("end_sec"),
            "top_symbol": row.get("top_symbol"),
            "runner_up_symbol": row.get("runner_up_symbol"),
            "margin": row.get("margin"),
            "candidate_source_kind": row.get("candidate_source_kind"),
        }

    def _near_tie_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
        near_tie_margin_threshold = 0.01
        weak_margin_threshold = 0.03
        normalized_rows = [row for row in rows if isinstance(row, dict)]
        for index, row in enumerate(normalized_rows):
            row["structural_instability_candidate"] = bool(
                index in {0, len(normalized_rows) - 1}
                or (
                    index > 0
                    and row.get("top_symbol") != normalized_rows[index - 1].get("top_symbol")
                )
                or (
                    index + 1 < len(normalized_rows)
                    and row.get("top_symbol") != normalized_rows[index + 1].get("top_symbol")
                )
            )
        ordered_rows = sorted(
            normalized_rows,
            key=lambda row: (
                float(row.get("margin", 0.0) or 0.0),
                int(row.get("global_index", -1) or -1),
            ),
        )
        near_tie_rows = [
            row
            for row in normalized_rows
            if float(row.get("margin", 0.0) or 0.0) <= near_tie_margin_threshold
        ]
        weak_margin_rows = [
            row
            for row in normalized_rows
            if float(row.get("margin", 0.0) or 0.0) <= weak_margin_threshold
        ]
        structural_instability_rows = [
            row
            for row in weak_margin_rows
            if bool(row.get("structural_instability_candidate"))
        ]
        return {
            "near_tie_margin_threshold": near_tie_margin_threshold,
            "weak_margin_threshold": weak_margin_threshold,
            "frame_count": len(normalized_rows),
            "near_tie_present": bool(near_tie_rows),
            "weak_margin_present": bool(weak_margin_rows),
            "near_tie_frame_count": len(near_tie_rows),
            "weak_margin_frame_count": len(weak_margin_rows),
            "structural_instability_candidate_count": len(structural_instability_rows),
            "structural_instability_candidate_present": bool(structural_instability_rows),
            "lowest_margin": (
                round(float(ordered_rows[0].get("margin", 0.0) or 0.0), 4)
                if ordered_rows
                else None
            ),
            "candidate_source_kinds": sorted(
                {
                    str(row.get("candidate_source_kind"))
                    for row in normalized_rows
                    if isinstance(row.get("candidate_source_kind"), str)
                }
            ),
            "lowest_margin_examples": [
                _compact_margin_example(row)
                for row in ordered_rows[:5]
            ],
            "structural_instability_examples": [
                _compact_margin_example(row)
                for row in structural_instability_rows[:5]
            ],
        }

    if not predictions:
        return {
            "backend_kind": backend_name,
            "frame_count": 0,
            "full_top_symbol_sequence": [],
            "full_run_symbols": [],
            "full_run_lengths": [],
            "anchor_window_frame_count": 0,
            "anchor_window_top_symbol_sequence": [],
            "anchor_window_run_symbols": [],
            "anchor_window_run_lengths": [],
            "anchor_window_frames": [],
            "near_tie_audit": {
                "full": _near_tie_audit([]),
                "anchor_window": _near_tie_audit([]),
                "target_neighborhood": _near_tie_audit([]),
            },
        }

    def _top_k_summary(prediction: dict[str, Any]) -> dict[str, Any]:
        mapped_frame_candidates = prediction_mapped_frame_candidates(prediction)
        candidate_source_kind = "mapped_frame_trace" if mapped_frame_candidates else "posterior_trace_top_k"
        posterior_map = _normalize_symbol_scores_frame(mapped_frame_candidates, score_key="posterior")
        logit_map = _normalize_symbol_scores_frame(mapped_frame_candidates, score_key="logit")
        if not posterior_map:
            posterior_map = _normalize_symbol_scores_frame(
                prediction.get("posterior_candidates"),
                score_key="confidence",
            )
        if not logit_map:
            logit_map = _normalize_symbol_scores_frame(prediction.get("logit_candidates"), score_key="logit")
        if not posterior_map and logit_map:
            posterior_map = _softmax_symbol_scores(logit_map)
        ordered_posteriors = _sorted_symbol_scores(posterior_map)
        ordered_logits = _sorted_symbol_scores(logit_map)
        top_symbol = None
        top_confidence = 0.0
        runner_up_symbol = None
        runner_up_confidence = 0.0
        if ordered_posteriors:
            top_symbol = ordered_posteriors[0][0]
            top_confidence = float(ordered_posteriors[0][1])
            if len(ordered_posteriors) > 1:
                runner_up_symbol = ordered_posteriors[1][0]
                runner_up_confidence = float(ordered_posteriors[1][1])
        elif ordered_logits:
            top_symbol = ordered_logits[0][0]
        return {
            "top_symbol": top_symbol,
            "top_confidence": round(top_confidence, 4),
            "runner_up_symbol": runner_up_symbol,
            "runner_up_confidence": round(runner_up_confidence, 4),
            "margin": round(max(0.0, top_confidence - runner_up_confidence), 4),
            "top_k_posteriors": [
                {"symbol": symbol, "confidence": round(value, 4)}
                for symbol, value in ordered_posteriors[:top_k]
            ],
            "top_k_logits": [
                {"symbol": symbol, "logit": round(value, 4)}
                for symbol, value in ordered_logits[:top_k]
            ],
            "candidate_source_kind": candidate_source_kind,
            "mapped_candidates": mapped_frame_candidates,
        }

    frame_rows: list[dict[str, Any]] = []
    full_symbols: list[str | None] = []
    for global_index, prediction in enumerate(predictions):
        summary = _top_k_summary(prediction)
        frame_rows.append(
            {
                "frame_index": global_index,
                "global_index": global_index,
                "source_segment_index": int(prediction.get("source_segment_index", 0)),
                "source_frame_index": int(prediction.get("source_frame_index", 0)),
                "start_sec": round(float(prediction.get("start_sec", 0.0)), 4),
                "end_sec": round(float(prediction.get("end_sec", 0.0)), 4),
                **summary,
            }
        )
        full_symbols.append(summary["top_symbol"])

    full_runs = run_summary_from_predictions(predictions, full_symbols)
    frame_hops = [
        max(0.0, float(frame_rows[index + 1]["start_sec"]) - float(frame_rows[index]["start_sec"]))
        for index in range(len(frame_rows) - 1)
    ]
    anchor_indices = list(range(len(frame_rows)))
    if isinstance(anchor_start_sec, (int, float)) and isinstance(anchor_end_sec, (int, float)):
        anchor_indices = prediction_indices_for_reference_window(
            predictions,
            start_sec=float(anchor_start_sec),
            end_sec=float(anchor_end_sec),
        )
    anchor_rows = [frame_rows[index] for index in anchor_indices]
    anchor_symbols = [row.get("top_symbol") for row in anchor_rows]
    anchor_runs = run_summary_from_predictions([predictions[index] for index in anchor_indices], anchor_symbols)
    neighborhood_indices = list(anchor_indices)
    if neighborhood_indices:
        start_index = max(0, min(neighborhood_indices) - 1)
        end_index = min(len(frame_rows) - 1, max(neighborhood_indices) + 1)
        neighborhood_indices = list(range(start_index, end_index + 1))
    neighborhood_rows = [frame_rows[index] for index in neighborhood_indices]
    neighborhood_symbols = [row.get("top_symbol") for row in neighborhood_rows]
    neighborhood_runs = run_summary_from_predictions(
        [predictions[index] for index in neighborhood_indices],
        neighborhood_symbols,
    )

    return {
        "backend_kind": backend_name,
        "frame_count": len(frame_rows),
        "frame_hop_sec": round(_safe_mean(frame_hops, 0.0), 4) if frame_hops else None,
        "full_frames": frame_rows,
        "full_top_symbol_sequence": [symbol for symbol in full_symbols if symbol is not None],
        "full_run_symbols": [run.get("symbol") for run in full_runs],
        "full_run_lengths": [run.get("frame_count") for run in full_runs],
        "full_runs": full_runs,
        "anchor_window_start_sec": round(float(anchor_start_sec), 4) if isinstance(anchor_start_sec, (int, float)) else None,
        "anchor_window_end_sec": round(float(anchor_end_sec), 4) if isinstance(anchor_end_sec, (int, float)) else None,
        "anchor_window_frame_count": len(anchor_rows),
        "anchor_window_top_symbol_sequence": [symbol for symbol in anchor_symbols if symbol is not None],
        "anchor_window_run_symbols": [run.get("symbol") for run in anchor_runs],
        "anchor_window_run_lengths": [run.get("frame_count") for run in anchor_runs],
        "anchor_window_frames": anchor_rows,
        "target_neighborhood_frame_count": len(neighborhood_rows),
        "target_neighborhood_top_symbol_sequence": [symbol for symbol in neighborhood_symbols if symbol is not None],
        "target_neighborhood_run_symbols": [run.get("symbol") for run in neighborhood_runs],
        "target_neighborhood_run_lengths": [run.get("frame_count") for run in neighborhood_runs],
        "target_neighborhood_frames": neighborhood_rows,
        "near_tie_audit": {
            "full": _near_tie_audit(frame_rows),
            "anchor_window": _near_tie_audit(anchor_rows),
            "target_neighborhood": _near_tie_audit(neighborhood_rows),
        },
    }


__all__ = [
    "frame_evidence_summary_from_posterior_payload",
    "prediction_candidate_payload",
    "prediction_indices_for_reference_window",
    "prediction_mapped_frame_candidates",
    "prediction_raw_label_candidates",
    "prediction_records_from_posterior_payload",
    "prediction_symbol_scores",
    "producer_runtime_contract_from_posterior_payload",
    "run_summary_from_predictions",
]
