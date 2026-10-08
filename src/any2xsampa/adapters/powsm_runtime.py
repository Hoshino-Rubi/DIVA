"""Isolated, offline, inference-only adapter for pinned upstream POWSM-CTC."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import time

from any2xsampa.adapters.powsm import MODEL_ID, REVISION, WEIGHT_SHA256
from any2xsampa.adapters.phoneticxeus_runtime import (
    enforce_offline, configure_cuda, check_gpu_budget, read_audio_crop,
)
from any2xsampa.training.limited_data_artifacts import read_json, verify_file, sha256_file


def verify_bundle(run: Path) -> dict[str, Any]:
    lock = read_json(run / "MODEL_LOCK.json")
    if (lock.get("model_id") != MODEL_ID or lock.get("revision") != REVISION
            or lock.get("weight_sha256") != WEIGHT_SHA256):
        raise ValueError("unexpected POWSM snapshot identity")
    for row in lock["files"]:
        path = (run / row["file"]).resolve()
        if not path.is_relative_to(run.resolve()):
            raise ValueError("snapshot member escapes run")
        verify_file(path, row["sha256"])
    return lock


class PowsmRuntime:
    def __init__(self, run: Path, cache: Path) -> None:
        self.run = run
        self.lock = verify_bundle(run)
        enforce_offline(cache)
        self.torch = configure_cuda()
        import yaml
        from espnet2.asr.frontend.default import DefaultFrontend
        from espnet2.asr.encoder.e_branchformer_ctc_encoder import EBranchformerCTCEncoder
        from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
        from espnet2.layers.global_mvn import GlobalMVN
        from espnet2.asr.ctc import CTC
        from espnet2.s2t.espnet_ctc_model import ESPnetS2TCTCModel

        source = (run / self.lock["source_directory"]).resolve()
        import espnet2
        if not Path(espnet2.__file__).resolve().is_relative_to(source):
            raise ValueError("runtime imported a different ESPnet source tree")
        config = yaml.safe_load((run / self.lock["config_file"]).read_text(encoding="utf-8"))
        if (config["frontend"] != "default" or config["normalize"] != "global_mvn"
                or config["encoder"] != "e_branchformer_ctc"
                or config["promptencoder"] != "transformer" or config["model"] != "espnet_ctc"):
            raise ValueError("unsupported upstream model architecture")
        self.labels = config["token_list"]
        if len(self.labels) != 40002 or self.labels[0] != "<blank>":
            raise ValueError("upstream output inventory changed")
        frontend = DefaultFrontend(**config["frontend_conf"])
        normalization = dict(config["normalize_conf"])
        normalization["stats_file"] = str(run / "model" / normalization["stats_file"])
        encoder_config = dict(config["encoder_conf"])
        prompt_config = dict(config["promptencoder_conf"])
        # Windows/FP32 path is chosen explicitly, before results; no flash dependency.
        encoder_config["use_flash_attn"] = False
        prompt_config["use_flash_attn"] = False
        encoder = EBranchformerCTCEncoder(input_size=frontend.output_size(), **encoder_config)
        prompt = TransformerEncoder(input_size=prompt_config["output_size"], input_layer=None, **prompt_config)
        model = ESPnetS2TCTCModel(vocab_size=len(self.labels), token_list=self.labels,
            frontend=frontend, specaug=None, normalize=GlobalMVN(**normalization), encoder=encoder,
            prompt_encoder=prompt, ctc=CTC(odim=len(self.labels), encoder_output_size=encoder.output_size(), **config["ctc_conf"]),
            report_cer=False, report_wer=False, **config["model_conf"])
        torch = self.torch
        state = torch.load(run / self.lock["weight_file"], map_location="cpu", weights_only=True, mmap=True)
        loading = model.load_state_dict(state, strict=True)
        del state
        self.model = model.eval().requires_grad_(False).to("cuda", dtype=torch.float32)
        if any(p.device.type != "cuda" or p.dtype != torch.float32 for p in model.parameters()):
            raise RuntimeError("all model parameters must be CUDA FP32")
        if list(model.encoder.interctc_layer_idx) != [4, 8, 10, 14]:
            raise ValueError("upstream self-conditioning configuration changed")
        self.conditioning_calls = 0
        def count_conditioning(*_: Any) -> None:
            self.conditioning_calls += 1
        self.conditioning_hook = model.encoder.conditioning_layer.register_forward_hook(count_conditioning)
        self.identity = {"model_id": MODEL_ID, "revision": REVISION, "weight_sha256": WEIGHT_SHA256,
            "snapshot_lock_sha256": sha256_file(run / "MODEL_LOCK.json"),
            "vocabulary_sha256": sha256_file(run / "vocabulary.json"),
            "espnet_revision": self.lock["espnet_revision"], "license": "cc-by-4.0"}
        self.runtime = {"torch": torch.__version__, "cuda": torch.version.cuda,
            "dtype": "float32", "device": "cuda:0", "offline": True, "cpu_threads": 4,
            "attention_implementation": "upstream eager; explicitly disabled flash",
            "upstream_recorded_torch": "2.9.1+cu128",
            "compatibility_path": "local torch2.8.0+cu126; strict weights and real replay verified separately",
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "loading": {"missing_keys": loading.missing_keys, "unexpected_keys": loading.unexpected_keys},
            "initial_memory": check_gpu_budget(torch)}
        torch.cuda.reset_peak_memory_stats()

    def encode(self, request: Mapping[str, Any]) -> tuple[Any, dict, dict, dict]:
        fields = {"example_id", "song_id", "split", "audio_role", "audio_path", "audio_sha256", "start_sec", "end_sec"}
        if set(request) != fields:
            raise ValueError("POWSM worker accepts only audio/source fields")
        train = {"freedom-flower", "half-moon-night-song", "like-you", "promise", "queen-road-east"}
        if not ((request["split"] == "train" and request["song_id"] in train)
                or (request["split"] == "dev" and request["song_id"] == "shalong")):
            raise ValueError("unexpected song/split; no final test access")
        if request["audio_role"] not in {"original", "separated_vocal"}:
            raise ValueError("this probe accepts only original or matched separated input")
        waveform, audio, crop = read_audio_crop(Path(request["audio_path"]), request["audio_sha256"],
            request["start_sec"], request["end_sec"])
        if len(waveform) > 320000:
            raise ValueError("POWSM cannot truncate an input longer than 20 seconds")
        torch = self.torch
        check_gpu_budget(torch)
        speech = torch.zeros((1, 320000), dtype=torch.float32, device="cuda")
        speech[0, :len(waveform)] = torch.from_numpy(waveform).to("cuda")
        tensor = lambda x: torch.tensor(x, dtype=torch.long, device="cuda")
        before = time.monotonic()
        self.conditioning_calls = 0
        with torch.inference_mode():
            encoded, lengths = self.model.encode(speech, tensor([320000]),
                tensor([[self.labels.index("<na>")]]), tensor([1]),
                tensor([[self.labels.index("<unk>"), self.labels.index("<pr>")]]), tensor([2]))
            if isinstance(encoded, tuple):
                encoded = encoded[0]
            logits = self.model.ctc.ctc_lo(encoded)[0, :int(lengths[0])].float().cpu().contiguous()
        torch.cuda.synchronize()
        if logits.shape != (501, 40002) or not torch.isfinite(logits).all() or self.conditioning_calls != 4:
            raise ValueError("unexpected POWSM frame geometry, conditioning, or non-finite output")
        runtime = {**self.runtime, "elapsed_seconds": time.monotonic() - before,
            "conditioning_calls": self.conditioning_calls,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "memory": check_gpu_budget(torch)}
        crop.update(powsm_model_samples=320000, powsm_zero_padding_samples=320000-len(waveform),
                    waveform_truncated=False, prefix_frames=2, emission_stride_sec=0.04)
        return logits, audio, crop, runtime
