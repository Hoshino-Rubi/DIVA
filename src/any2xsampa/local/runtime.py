"""Persistent isolated model execution. The main UI process never imports Torch."""
from __future__ import annotations
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import shutil
import sys
import threading
from .common import ROOT, check_cancel, read, request, sha, write
from .registry import package

class RuntimeEngine:
    def __init__(self):
        self.key = None
        self.model = None
        self.head = None
        self.lock = None
        self.loads = 0

    def close(self):
        self.head = self.model = None
        gc.collect()
        if "torch" in sys.modules:
            sys.modules["torch"].cuda.empty_cache()
        self.key = None
        if self.lock is not None:
            self.lock.__exit__(None, None, None)
            self.lock = None

    def execute(self, action: str, params: dict, cancel: threading.Event, progress):
        if action == "unload":
            self.close()
            return {"unloaded": True}
        if action != "recognize":
            raise ValueError("Runtime accepts only recognize/unload")
        p = dict(params)
        output = Path(p.pop("_output"))
        r = request(p)
        spec = package(r["model"])
        if Path(spec["python"]).resolve() != Path(sys.executable).resolve():
            raise ValueError("Model package requires a different isolated interpreter")
        if r["device"] not in spec["devices"]:
            raise ValueError("Device not supported by this package")
        from any2xsampa.adapters.phoneticxeus_runtime import enforce_offline
        enforce_offline(output / "runtime-cache")
        check_cancel(cancel)
        gpu = r["device"] == "cuda"
        from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor, query_total_gpu_bytes
        if gpu and query_total_gpu_bytes() > 14*1024**3:
            raise RuntimeError("Card memory exceeds 14 GiB; no CPU fallback")
        monitor = ProcessResourceMonitor(output / "monitor", timeout_seconds=900) if gpu else nullcontext()
        with monitor:
            key = (spec["package_sha256"], r["device"])
            reused = self.key == key
            if self.key != key:
                self.close()
                from any2xsampa.adapters.candidate_inference import serial_candidate_job
                self.lock = serial_candidate_job()
                self.lock.__enter__()
                progress({"stage": "loading", "model": r["model"], "device": r["device"]})
                try:
                    self._load(spec, r["device"], output)
                    self.key = key
                    self.loads += 1
                except BaseException:
                    self.close()
                    raise
            check_cancel(cancel)
            progress({"stage": "recognizing", "model_reused": reused, "load_count": self.loads})
            try:
                if spec["backend"] == "guarded-phoneticxeus":
                    result = self._candidate(r, spec, output)
                else:
                    result = self._onnx(output)
                check_cancel(cancel)
                evidence = {**result, "model_id": spec["id"], "package_sha256": spec["package_sha256"],
                            "device": r["device"], "quality": spec["quality"], "load_count": self.loads}
                artifact = output / "runtime-result.json"
                write(artifact, evidence)
                return {"evidence_path": str(artifact), "evidence_sha256": sha(artifact)}
            except BaseException:
                self.close()
                raise

    def _load(self, spec, device, output):
        paths = spec["paths"]
        if spec["backend"] == "guarded-phoneticxeus":
            from any2xsampa.adapters.phoneticxeus_runtime import PhoneticXeusRuntime
            from any2xsampa.adapters.guarded_phoneticxeus import GuardedPhoneHead
            self.model = PhoneticXeusRuntime(Path(paths["snapshot"]), cache_dir=output / "runtime-cache")
            self.head = GuardedPhoneHead(self.model.torch, self.model.model.model.ctc.ctc_lo,
                                        self.model.vocabulary, Path(paths["candidate"]), spec["candidate_sha256"])
        else:
            from .onnx_frames import RecordedOnnxRecognizer
            def factory(model, providers):
                import onnxruntime as ort
                options = ort.SessionOptions()
                options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
                selected = [(p, {"gpu_mem_limit": 7*1024**3, "arena_extend_strategy": "kSameAsRequested"})
                            if p == "CUDAExecutionProvider" else p for p in providers]
                session = ort.InferenceSession(model, sess_options=options, providers=selected)
                if device == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
                    raise RuntimeError("CUDA provider failed; CPU fallback forbidden")
                return session
            self.model = RecordedOnnxRecognizer(
                model_path=paths["model"], vocab_path=paths["vocabulary"], device=device,
                expected_model_sha256=spec["model_sha256"], expected_vocab_sha256=spec["vocabulary_sha256"],
                cpu_thread_limit=4, session_factory=factory)
            self.model._load_runtime()

    def _candidate(self, r, spec, output):
        from any2xsampa.adapters.candidate_inference import audio_request
        from any2xsampa.adapters.phoneticxeus import PhoneticXeusPosteriorRecognizer
        ar = audio_request(Path(r["audio"]), r["start"], r["duration"])
        row = {k: ar[k] for k in ("audio_path", "audio_sha256", "start_sec", "end_sec")}
        row.update(example_id="user-crop", song_id="user-selected-audio", split="inference", audio_role="user_input")
        write(output / "FORWARD_STARTED.json", {"backend": "guarded-phoneticxeus", "full_encoder_forwards_requested": 1, "retry": False})
        features, _, payload = self.model.encode(row)
        write(output / "FORWARD_FINISHED.json", {"full_encoder_forwards_completed": 1})
        artifact = self.head.save(features, payload, output / "evidence")
        pp = Path(artifact["posterior_path"])
        shutil.copyfile(spec["paths"]["vocabulary"], pp.parent / "vocabulary.json")
        posterior = read(pp)
        posterior["raw_evidence"]["vocabulary_file"] = "vocabulary.json"
        write(pp, posterior, replace=True)
        segments = PhoneticXeusPosteriorRecognizer(pp).recognize(Path(r["audio"]))
        for s in segments:
            s.normalized_ipa = s.metadata["raw_label_trace"]["label"]
        return {"segments": [s.to_dict() for s in segments], "posterior_path": str(pp),
                "posterior_sha256": sha(pp), "time_basis": "original_audio",
                "peak_reserved_bytes": self.model.torch.cuda.max_memory_reserved()}

    def _onnx(self, output):
        from any2xsampa.pipeline.normalizer import ConfigurablePhoneNormalizer
        write(output / "FORWARD_STARTED.json", {"backend": "onnx-espeak", "full_encoder_forwards_requested": 1, "retry": False})
        from any2xsampa.adapters.onnx_espeak import OnnxEspeakInferenceError
        from dataclasses import asdict
        self.model.last_frames = []
        try:
            segments = self.model.recognize(output / "input-crop.wav")
        except OnnxEspeakInferenceError as exc:
            if not self.model.last_frames or any(f["symbol"] is not None for f in self.model.last_frames):
                raise
            segments = []  # Verified all-blank frames, e.g. an instrumental-only window.
        frames = [{**f, "candidates": [asdict(c) for c in f["candidates"]]} for f in self.model.last_frames]
        frame_path = output / "onnx-frames.json"
        write(frame_path, {"schema_version": "local_onnx_frames_v1", "frames": frames})
        self.model.last_frames = []
        write(output / "FORWARD_FINISHED.json", {"full_encoder_forwards_completed": 1})
        segments = ConfigurablePhoneNormalizer(ROOT / "src/any2xsampa/config/ipa_aliases.json").normalize(segments)
        return {"segments": [s.to_dict() for s in segments], "time_basis": "crop",
                "model_metadata": self.model.model_metadata(), "frame_evidence_path": str(frame_path),
                "frame_evidence_sha256": sha(frame_path)}
