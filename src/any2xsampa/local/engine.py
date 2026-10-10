"""Shared local orchestration. Machine evidence and manual drafts stay separate."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import array
import copy
from pathlib import Path
import sys
import threading
import time
import uuid
from .common import ROOT, Cancelled, audio_info, check_cancel, crop, read, request, sha, write
from .process import WorkerClient
from .storage import state_dir
from .registry import doctor, models, package
from any2xsampa.exporters.ipa_xsampa import IpaXSampaConverter

class LocalEngine:
    def __init__(self):
        self.worker = None
        self.worker_key = None

    def close(self):
        if self.worker is not None:
            self.worker.close()
        self.worker = self.worker_key = None

    def execute(self, action: str, params: dict, cancel: threading.Event, progress):
        check_cancel(cancel)
        if action in {"model_setup_status", "install_models"}:
            from .model_install import status, install
            return status() if action == "model_setup_status" else install(params, cancel, progress)
        if action == "models":
            return {"models": models()}
        if action == "note_runtime":
            from .game_score_tasks import runtime_config
            try:
                directory, python = runtime_config(params.get("snapshot"))
                return {"available": True, "snapshot": str(directory), "python": python, "device": "cpu", "model_loaded": False}
            except (ValueError, OSError) as exc:
                return {"available": False, "reason": str(exc), "model_loaded": False}
        if action == "doctor":
            return doctor()
        if action == "unload":
            self.close()
            return {"unloaded": True}
        if action == "inspect_audio":
            return self.inspect_audio(Path(params["audio"]), cancel)
        if action == "draft":
            return self.draft(params)
        if action in {"build_score", "import_score", "export_score"}:
            from .score_tasks import build_score, import_score, export_saved_score
            if action == "build_score":
                return build_score(params, cancel, progress)
            if action == "import_score":
                return import_score(params, cancel, progress)
            return export_saved_score(params)
        if action == "export":
            return self.export(params)
        if action == "export_vsqx":
            if not params.get("template_path"):
                from .game_score_tasks import export_saved_notes
                if params.get("use_draft") or params.get("apply_phone_suggestions"):
                    raise ValueError("These options require an explicit template")
                clean = {k: v for k, v in params.items() if k not in {"template_path", "use_draft", "apply_phone_suggestions"}}
                return export_saved_notes(clean)
            if any(params.get(k) is not None for k in ("notes_path", "phones_path", "tempo")):
                raise ValueError("Template export cannot be combined with generated-note options")
            from .vsqx_export import export_vsqx_from_result
            return export_vsqx_from_result(
                result_path=params["result_path"],
                template_path=params["template_path"],
                output_path=params["output_path"],
                result_sha256=params.get("result_sha256"),
                use_draft=bool(params.get("use_draft", False)),
                apply_phone_suggestions=bool(params.get("apply_phone_suggestions", False)),
            )
        if action == "preview_xsampa":
            return self.preview_xsampa(params)
        if action == "recognize-whole":
            from .whole_task import run_whole
            return run_whole(self, params, cancel, progress)
        if action == "replay" and "fixture" in params and read(Path(params["fixture"])).get("scope") == "whole_file":
            from .whole_task import replay_whole
            return replay_whole(self, params, cancel, progress)
        if action not in {"recognize", "replay"}:
            raise ValueError("Unknown local command: " + action)
        started = time.monotonic()
        if action == "recognize":
            params = request(params)
        else:
            if set(params) - {"fixture", "audio", "start", "duration", "output_dir"} or "fixture" not in params:
                raise ValueError("replay requires an explicit fixture; no model inference")
            source = read(Path(params["fixture"]))
            if source.get("schema_version") == "local_task_result_v1":
                params = {**params, "audio": source["input"]["path"], "start": source["input"]["start_sec"],
                          "duration": source["input"]["end_sec"]-source["input"]["start_sec"]}
            elif source.get("producer", {}).get("backend_kind") == "phoneticxeus":
                p = source["producer"]
                if not (params.get("audio") or p["input_audio"].get("path")):
                    raise ValueError("Bound posterior replay requires --audio (its source metadata has no path)")
                params = {**params, "audio": params.get("audio") or p["input_audio"].get("path"),
                          "start": p["crop"]["requested_start_sec"],
                          "duration": p["crop"]["requested_end_sec"]-p["crop"]["requested_start_sec"]}
            elif "audio" not in params:
                raise ValueError("Generic fixture replay requires --audio")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = Path(params.get("output_dir") or state_dir(ROOT) / "runs" / ("local-task-"+stamp+"-"+uuid.uuid4().hex[:8])).resolve()
        output.mkdir(parents=True, exist_ok=False)
        write(output / "request.json", {"command": action, "params": params, "at_utc": stamp})
        try:
            progress({"stage": "validating", "output_dir": str(output)})
            info = crop(Path(params["audio"]), params.get("start", 0), params.get("duration", 8), output / "input-crop.wav")
            check_cancel(cancel)
            if action == "recognize":
                spec = package(params["model"])
                key = (spec["package_sha256"], params["device"])
                if self.worker is None or self.worker_key != key or self.worker.process.poll() is not None:
                    self.close()
                    self.worker = WorkerClient(spec["python"], state_dir(ROOT) / "worker-logs" / (uuid.uuid4().hex + ".log"))
                    self.worker_key = key
                artifact = self.worker.call("recognize", {**params, "_output": str(output)}, cancel, progress)
                evidence_path = Path(artifact["evidence_path"])
                if evidence_path.resolve() != output / "runtime-result.json" or sha(evidence_path) != artifact["evidence_sha256"]:
                    raise ValueError("Worker evidence path/hash mismatch")
                evidence = read(evidence_path)
                mode = "real"
            else:
                evidence = self.replay(params, info)
                mode = "replay"
            check_cancel(cancel)
            if sha(Path(params["audio"])) != info["sha256"]:
                raise ValueError("Audio changed during inference")
            progress({"stage": "saving"})
            result = self.save_result(output, mode, info, evidence)
            write(output / "EXIT.json", {"status": "completed", "elapsed_seconds": time.monotonic()-started,
                                        "model_forwards": 1 if action == "recognize" else 0})
            return {"result_path": str(output / "result.json"), "output_dir": str(output),
                    "mode": mode, "phone_count": len(result["segments"]),
                    "model_id": evidence.get("model_id"), "load_count": evidence.get("load_count")}
        except BaseException as e:
            if action == "recognize":
                self.close()
            write(output / "EXIT.json", {"status": "cancelled" if isinstance(e, Cancelled) else "failed",
                                        "error": str(e), "elapsed_seconds": time.monotonic()-started, "retried": False})
            raise

    @staticmethod
    def replay(params, info):
        file = Path(params["fixture"])
        payload = read(file)
        if payload.get("schema_version") == "local_task_result_v1":
            if info["sha256"] != payload["input"]["sha256"] or info["crop_sha256"] != payload["input"]["crop_sha256"]:
                raise ValueError("Replay audio hash mismatch")
            from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange
            for s in payload["segments"]:
                PhoneSegment(TimeRange(s["start_sec"], s["end_sec"]),
                             [PhoneCandidate(**c) for c in s["candidate_phones"]], s["confidence"])
            return {"segments": copy.deepcopy(payload["segments"]), "time_basis": "original_audio",
                    "source_result_sha256": sha(file), "source_result_path": str(file.resolve()),
                    "original_mode": payload["mode"], "model_id": payload.get("model_id")}
        if payload.get("producer", {}).get("backend_kind") == "phoneticxeus":
            from any2xsampa.adapters.phoneticxeus import PhoneticXeusPosteriorRecognizer
            segments = PhoneticXeusPosteriorRecognizer(file, fixture=True).recognize(Path(params["audio"]))
            for s in segments:
                s.normalized_ipa = s.metadata["raw_label_trace"]["label"]
            basis = "original_audio"
        else:
            from any2xsampa.adapters.recognizer import ConfigurableUniversalPhoneRecognizer, RecognizerConfig
            from any2xsampa.pipeline.normalizer import ConfigurablePhoneNormalizer
            model = ConfigurableUniversalPhoneRecognizer(RecognizerConfig(mode="fixture", fixture_path=file, allow_stub_fallback=False))
            segments = ConfigurablePhoneNormalizer(ROOT / "src/any2xsampa/config/ipa_aliases.json").normalize(model.recognize(Path(params["audio"])))
            basis = "crop"
        return {"segments": [s.to_dict() for s in segments], "time_basis": basis,
                "fixture_path": str(file.resolve()), "fixture_sha256": sha(file)}

    @staticmethod
    def save_result(output, mode, info, evidence):
        segments = copy.deepcopy(evidence["segments"])
        projection = IpaXSampaConverter().project([s["normalized_ipa"] for s in segments])
        for s, xsampa in zip(segments, projection["xsampa"], strict=True):
            if evidence["time_basis"] == "crop":
                s["start_sec"] += info["start_sec"]
                s["end_sec"] += info["start_sec"]
            s["xsampa"] = xsampa
        if any(s["start_sec"] < info["start_sec"]-1e-5 or s["end_sec"] > info["end_sec"]+1e-5 for s in segments):
            raise ValueError("Phone emissions fall outside the selected audio; refusing misleading playback")
        value = {**evidence, "schema_version": "local_task_result_v1", "mode": mode, "input": info,
                 "time_basis": "original_audio", "segments": segments, "timings_are_phone_durations": False,
                 "default_model_changed": False, "human_reference": False,
                 "unmapped": list(projection["unmapped_label_counts"]),
                 "xsampa_export": {k:v for k,v in projection.items() if k != "xsampa"}}
        write(output / "result.json", value)
        (output / "phones.ipa.txt").write_text(" ".join(s["normalized_ipa"] for s in segments)+"\n", encoding="utf-8")
        (output / "phones.xsampa.txt").write_text(" ".join(s["xsampa"] or "[UNMAPPED:"+s["normalized_ipa"]+"]" for s in segments)+"\n", encoding="utf-8")
        return value

    @staticmethod
    def preview_xsampa(params):
        file = Path(params["result_path"])
        digest = sha(file)
        if params.get("result_sha256") != digest:
            raise ValueError("Export preview belongs to a different result version")
        result = read(file)
        if result.get("schema_version") != "local_task_result_v1" or sha(file) != digest:
            raise ValueError("Invalid or changed local task result")
        projection = IpaXSampaConverter().project([s["normalized_ipa"] for s in result["segments"]])
        return {**projection, "result_sha256": digest, "source_modified": False,
                "model_forwards": 0}

    @staticmethod
    def draft(params):
        file = Path(params["result_path"]).resolve()
        result = read(file)
        if result.get("schema_version") != "local_task_result_v1":
            raise ValueError("Expected local task result")
        if params.get("result_sha256") != sha(file):
            raise ValueError("Draft belongs to a different result version")
        rows = params["edits"]
        if not isinstance(rows, list) or len(rows) != len(result["segments"]):
            raise ValueError("Draft must preserve the machine segment count")
        for i, row in enumerate(rows):
            if row.get("index") != i or not isinstance(row.get("ipa"), str) or not row["ipa"].strip():
                raise ValueError("Invalid manual phone edit")
        value = {"schema_version": "local_review_draft_v1", "result_sha256": sha(file), "edits": rows,
                 "source": "user_edit_not_acoustic_ground_truth"}
        write(file.parent / "review-draft.json", value, replace=True)
        return {"draft_path": str(file.parent / "review-draft.json"), "saved": True}

    @staticmethod
    def export(params):
        result_path = Path(params["result_path"])
        result = read(result_path)
        if result.get("schema_version") != "local_task_result_v1":
            raise ValueError("Expected local task result")
        phones = [s["normalized_ipa"] for s in result["segments"]]
        if params.get("use_draft", False):
            draft = read(result_path.parent / "review-draft.json")
            if draft["result_sha256"] != sha(result_path):
                raise ValueError("Draft/source mismatch")
            if len(draft["edits"]) != len(phones) or any(x["index"] != i for i,x in enumerate(draft["edits"])):
                raise ValueError("Invalid draft indexing")
            phones = [x["ipa"] for x in draft["edits"]]
        fmt = params.get("format", "xsampa")
        if fmt not in {"ipa", "xsampa"}:
            raise ValueError("Export format must be ipa or xsampa")
        projection = IpaXSampaConverter().project(phones)
        exported = [x or "[UNMAPPED:"+p+"]" for p,x in zip(phones,projection["xsampa"],strict=True)]
        text = " ".join(phones if fmt == "ipa" else exported)+"\n"
        output = Path(params["output"])
        with output.open("x", encoding="utf-8") as f:
            f.write(text)
        write(output.with_name(output.name+".provenance.json"), {"result_sha256": sha(result_path),
              "manual_draft": bool(params.get("use_draft")), "format": fmt, "output_sha256": sha(output),
              "xsampa_export": {k:v for k,v in projection.items() if k != "xsampa"}})
        return {"output": str(output.resolve()), "unmapped_phone_count": projection["unmapped_phone_count"]}

    @staticmethod
    def inspect_audio(file: Path, cancel):
        import wave
        info = audio_info(file)
        peaks = []
        with wave.open(str(file), "rb") as w:
            width, channels, frames = w.getsampwidth(), w.getnchannels(), w.getnframes()
            if width != 2:
                return {**info, "peaks": [], "waveform_warning": "Waveform preview currently requires 16-bit PCM; audio was not modified"}
            step = max(1, (frames+799)//800)
            while chunk := w.readframes(step):
                check_cancel(cancel)
                samples = array.array("h", chunk)
                peaks.append(max((abs(s)/32768 for s in samples), default=0))
        return {**info, "peaks": peaks}
