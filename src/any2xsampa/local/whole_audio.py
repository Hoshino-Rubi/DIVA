"""Whole-file planning and evidence stitching. No model calls or label conditions."""
from __future__ import annotations
import copy
import math
from pathlib import Path
from .common import check_cancel, read, sha, write

CHUNK_SECONDS = 20
OVERLAP_SECONDS = 2
STEP_SECONDS = CHUNK_SECONDS - OVERLAP_SECONDS
POLICY = "overlap_midpoint_frame_ownership_v1"

def plan_chunks(info: dict) -> list[dict]:
    rate, total = info["sample_rate"], info["frames"]
    if not isinstance(rate, int) or rate <= 0 or total < math.ceil(.1*rate):
        raise ValueError("Audio must contain at least 0.1 seconds")
    chunks = []
    first = 0
    while first < total:
        last = min(total, first+CHUNK_SECONDS*rate)
        chunks.append({"index": len(chunks), "start_sec": first/rate, "end_sec": last/rate,
                       "source_start_sample": first, "source_end_sample_exclusive": last})
        if last == total:
            break
        first += STEP_SECONDS*rate
        if total-first < math.ceil(.1*rate):
            raise ValueError("Final model window would be shorter than 0.1 seconds")
    for i, chunk in enumerate(chunks):
        chunk["keep_start_sec"] = 0.0 if i == 0 else (chunks[i-1]["end_sec"]+chunk["start_sec"])/2
        chunk["keep_end_sec"] = total/rate if i == len(chunks)-1 else (chunk["end_sec"]+chunks[i+1]["start_sec"])/2
    return chunks

def whole_input(audio: Path) -> dict:
    from .common import audio_info
    info = audio_info(audio)
    return {**info, "sha256": sha(audio), "start_sec": 0.0, "end_sec": info["duration"],
            "source_start_sample": 0, "source_end_sample_exclusive": info["frames"],
            "scope": "whole_file", "audio_copy": False}

def verify_chunk_result(record: dict, spec: dict, info: dict) -> dict:
    file = Path(record["result_path"])
    if sha(file) != record["result_sha256"]:
        raise ValueError("Whole-file chunk result hash mismatch")
    result = read(file)
    source = result["input"]
    if result["mode"] != "real" or source["sha256"] != info["sha256"]:
        raise ValueError("Chunk audio source/mode differs from the whole-file request")
    if source["source_start_sample"] != spec["source_start_sample"] or source["source_end_sample_exclusive"] != spec["source_end_sample_exclusive"]:
        raise ValueError("Chunk sample interval differs from frozen plan")
    if result["time_basis"] != "original_audio":
        raise ValueError("Chunk emissions must use original-audio time")
    return result

def stitch_candidate(chunks: list[dict], results: list[dict], info: dict, cancel) -> tuple[list, dict]:
    """Keep one raw logit row per global frame, then perform CTC collapse ONCE.

    This preserves repeats separated by blank and joins a continuous emission
    spanning a chunk seam. It never deletes matching phone strings heuristically.
    """
    import numpy as np
    from safetensors.numpy import load_file
    from any2xsampa.adapters.phoneticxeus import emission_payload
    from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange
    kept = []
    ranges = []
    identity = vocabulary = geometry = None
    next_frame = 0
    model_samples = (info["frames"]*16000+info["sample_rate"]-1)//info["sample_rate"]
    expected_frames = None
    source_ids = []
    for chunk, result in zip(chunks, results, strict=True):
        check_cancel(cancel)
        path = Path(result["posterior_path"])
        if sha(path) != result["posterior_sha256"]:
            raise ValueError("Chunk posterior changed")
        p = read(path)
        raw = p["raw_evidence"]
        evidence = (path.parent/raw["file"]).resolve()
        vocab_file = (path.parent/raw["vocabulary_file"]).resolve()
        if not evidence.is_relative_to(path.parent.resolve()) or not vocab_file.is_relative_to(path.parent.resolve()):
            raise ValueError("Chunk evidence must stay adjacent to its posterior")
        current_identity = p["producer"]["model_metadata"]
        if sha(evidence) != raw["sha256"] or sha(vocab_file) != current_identity["vocabulary_sha256"]:
            raise ValueError("Chunk raw evidence/vocabulary hash mismatch")
        current_vocab = read(vocab_file)
        g = p["decoding"]["geometry"]
        crop = p["producer"]["crop"]
        if p["producer"]["input_audio"]["sha256"] != info["sha256"] or crop["source_start_sample"] != chunk["source_start_sample"] or crop["source_end_sample_exclusive"] != chunk["source_end_sample_exclusive"]:
            raise ValueError("Posterior source interval mismatch")
        if identity is None:
            identity, vocabulary = current_identity, current_vocab
            geometry = {"stride_samples": g["stride_samples"], "receptive_field_samples": g["receptive_field_samples"]}
        if identity != current_identity or vocabulary != current_vocab or any(g[k] != v for k,v in geometry.items()):
            raise ValueError("Incompatible model/vocabulary/timebase across chunks")
        logits = load_file(str(evidence))["frame_logits"]
        if logits.dtype != np.float32 or logits.shape != (g["frame_count"], len(vocabulary)) or not np.isfinite(logits).all():
            raise ValueError("Invalid full-vocabulary logits")
        stride = geometry["stride_samples"]
        frame_offset = crop["effective_start_sec"]*16000/stride
        if not math.isclose(frame_offset, round(frame_offset), rel_tol=0, abs_tol=1e-6):
            raise ValueError("Chunk starts are not aligned to the global frame grid")
        if expected_frames is None:
            expected_frames = (model_samples-geometry["receptive_field_samples"])//geometry["stride_samples"]+1
        ids = np.arange(logits.shape[0])+round(frame_offset)
        centers = (ids*stride+(geometry["receptive_field_samples"]-1)/2)/16000
        mask = (centers >= chunk["keep_start_sec"]) & (centers < chunk["keep_end_sec"]) & (ids < expected_frames)
        selected = ids[mask]
        if selected.size == 0 or int(selected[0]) != next_frame or np.any(np.diff(selected) != 1):
            raise ValueError("Missing/duplicated global CTC frames at chunk seam")
        next_frame = int(selected[-1])+1
        source_ids.extend(selected.tolist())
        kept.append(logits[mask])
        ranges.append({"chunk": chunk["index"], "global_frame_start": int(selected[0]), "global_frame_end_exclusive": next_frame})
    if next_frame != expected_frames or source_ids != list(range(expected_frames)):
        raise ValueError("Whole-file global frame coverage is incomplete")
    values = np.concatenate(kept, axis=0)
    crop = {"effective_start_sec": 0.0, "effective_end_sec": info["duration"]}
    p = emission_payload(values, vocabulary, input_audio=info, crop=crop, model_metadata=identity,
                         geometry={**geometry, "frame_count": expected_frames}, runtime={"inference": "serial_window_stitch"})
    segments = []
    for s in p["segments"]:
        phone = s["raw_label_trace"]["label"]
        segments.append(PhoneSegment(TimeRange(s["start_sec"],s["end_sec"]),
                        [PhoneCandidate(**c) for c in s["candidate_phones"]],s["confidence"],normalized_ipa=phone,
                        metadata={"source_mode": "real", "backend_kind": "phoneticxeus", "raw_label_trace": s["raw_label_trace"],
                                  "posterior_summary": s["posterior_summary"], "boundary_uncertainty": s["boundary_uncertainty"],
                                  "whole_file_policy": POLICY, "human_reference": False}).to_dict())
    return segments, {"algorithm": "global_frame_ownership_then_greedy_ctc", "frame_count": expected_frames,
                      "full_label_count": len(vocabulary), "ranges": ranges, "blank_frames": p["decoding"]["blank_frame_count"]}

def stitch_onnx(chunks: list[dict], results: list[dict], info: dict, cancel) -> tuple[list, dict]:
    from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange
    from any2xsampa.pipeline.normalizer import ConfigurablePhoneNormalizer
    from .common import ROOT
    frames = []
    for chunk, result in zip(chunks, results, strict=True):
        check_cancel(cancel)
        file = Path(result["frame_evidence_path"])
        if sha(file) != result["frame_evidence_sha256"]:
            raise ValueError("ONNX frame evidence hash mismatch")
        raw = read(file)
        if raw.get("schema_version") != "local_onnx_frames_v1":
            raise ValueError("Expected blank-preserving ONNX frame evidence")
        for frame in raw["frames"]:
            start = chunk["start_sec"]+frame["start_index"]/16000
            end = chunk["start_sec"]+frame["end_index"]/16000
            if chunk["keep_start_sec"] <= (start+end)/2 < chunk["keep_end_sec"]:
                frames.append({**frame,"start_sec":start,"end_sec":end})
    segments, run = [], []
    def flush():
        if not run or not run[0]["symbol"]:
            return
        symbol = run[0]["symbol"]
        totals = {}
        for frame in run:
            for candidate in frame["candidates"]:
                totals[candidate["symbol"]] = totals.get(candidate["symbol"],0)+candidate["confidence"]/len(run)
        ordered = sorted(totals,key=lambda k:(-totals[k],k))[:5]
        ordered = [symbol]+[k for k in ordered if k != symbol]
        candidates = [PhoneCandidate(k,totals.get(k,0)) for k in ordered[:5]]
        segments.append(PhoneSegment(TimeRange(run[0]["start_sec"],run[-1]["end_sec"]),candidates,candidates[0].confidence,
                                     metadata={"source_mode":"real","backend_kind":"onnx-espeak","whole_file_policy":POLICY}))
    for frame in frames:
        if run and frame["symbol"] != run[-1]["symbol"]:
            flush();run=[]
        run.append(frame)
    flush()
    segments = ConfigurablePhoneNormalizer(ROOT/'src/any2xsampa/config/ipa_aliases.json').normalize(segments)
    return [s.to_dict() for s in segments], {"algorithm":"owned_mapped_frames_then_ctc", "frame_count":len(frames),
                                            "blank_frames":sum(f["symbol"] is None for f in frames)}

def merge_results(manifest: dict, cancel) -> dict:
    if manifest.get("schema_version") != "local_whole_evidence_v1" or manifest.get("policy") != POLICY:
        raise ValueError("Unsupported whole-file evidence policy")
    info, chunks = manifest["input"], manifest["chunks"]
    if chunks != plan_chunks(info) or len(manifest["completed"]) != len(chunks):
        raise ValueError("Whole-file plan/coverage is incomplete")
    results = [verify_chunk_result(r,c,info) for r,c in zip(manifest["completed"],chunks,strict=True)]
    for result in results:
        if result["package_sha256"] != manifest["package_sha256"] or result["model_id"] != manifest["model_id"] or result["device"] != manifest["device"]:
            raise ValueError("Whole-file chunk model identity changed")
    if manifest["backend"] == "guarded-phoneticxeus":
        segments, summary = stitch_candidate(chunks,results,info,cancel)
    elif manifest["backend"] == "onnx-espeak":
        segments, summary = stitch_onnx(chunks,results,info,cancel)
    else:
        raise ValueError("No whole-file evidence merger for this backend")
    return {"segments":segments,"time_basis":"original_audio","scope":"whole_file", "merge_summary":summary,
            "model_id":manifest["model_id"],"package_sha256":manifest["package_sha256"],"device":manifest["device"],
            "quality":results[0].get("quality"),"chunk_count":len(chunks),"load_count":results[-1].get("load_count"),
            "boundary_note":"Overlapping-context machine inference; seam phonetic correctness is not human-verified"}
