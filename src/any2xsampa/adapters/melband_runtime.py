"""Reviewed, pinned MelBand architecture behind an optional offline worker."""
from __future__ import annotations

from contextlib import nullcontext
import importlib
from pathlib import Path
import sys
import types
from typing import Any, Callable

from any2xsampa.training.limited_data_artifacts import read_json, verify_file


def safe_model_config(path: Path) -> dict:
    import yaml

    class ConfigLoader(yaml.SafeLoader):
        pass

    # The author's immutable YAML uses a tuple for STFT window sizes. Permit
    # this one inert container tag, not arbitrary Python object construction.
    ConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node)))
    result = yaml.load(path.read_text(encoding="utf-8"), Loader=ConfigLoader)
    if not isinstance(result, dict) or set(result) != {"model", "training", "inference"}:
        raise ValueError("unexpected author separator configuration")
    if result["model"]["sample_rate"] != 44100 or result["model"]["num_stems"] != 1 or result["model"]["stereo"] is not True:
        raise ValueError("separator config differs from the declared audio contract")
    if result["inference"] != {"num_overlap": 2, "chunk_size": 352800}:
        raise ValueError("separator inference geometry differs from the author lock")
    return result


def verify_model_assets(run: Path, role: str) -> tuple[dict, dict]:
    start = read_json(run / "execution-start.json")
    verify_file(run / "policy.json", start["policy_sha256"])
    policy = read_json(run / "policy.json")
    assets = read_json(run / "assets/assets-lock.json")
    if assets["policy_sha256"] != start["policy_sha256"]:
        raise ValueError("auxiliary assets use a different experiment policy")
    model = assets["models"][role]
    if model["model"] != policy["models"][role]:
        raise ValueError("auxiliary model differs from the approved lock")
    if role == "separator":
        verify_file(Path(model["weight_path"]), model["model"]["weight_sha256"])
        code = Path(model["code_root"])
        if not code.resolve().is_relative_to(run.resolve() / "assets/separator"):
            raise ValueError("separator code must remain inside this run")
        lock_path = code.parent / "code-lock.json"
        verify_file(lock_path, model["code_lock_sha256"])
        lock = read_json(lock_path)
        if lock["revision"] != model["model"]["code_revision"]:
            raise ValueError("separator code revision differs from policy")
        names = set()
        for entry in lock["files"]:
            path = (code / entry["file"]).resolve()
            if not path.is_relative_to(code.resolve()) or entry["file"] in names:
                raise ValueError("unsafe or duplicate code member")
            names.add(entry["file"])
            verify_file(path, entry["sha256"])
        if {p.relative_to(code).as_posix() for p in code.rglob("*") if p.is_file()} != names:
            raise ValueError("undeclared separator code file")
    else:
        root = Path(model["root"])
        if {p.name for p in root.iterdir() if p.is_file()} != {r["file"] for r in model["files"]}:
            raise ValueError("Qwen local snapshot members changed")
        for entry in model["files"]:
            path = root / entry["file"]
            if path.parent != root:
                raise ValueError("invalid Qwen snapshot member")
            verify_file(path, entry["sha256"])
        verify_file(root / "model.safetensors", model["model"]["weight_sha256"])
    return model, policy


def demix_bounded(torch: Any, model: Any, mixture: Any, *, chunk_size: int, overlap: int,
                  device: str, mixed_precision: bool, check_budget: Callable[[], Any] | None = None) -> Any:
    """Author overlap-add geometry, but reject invalid output instead of nan_to_num."""
    if mixture.ndim != 2 or mixture.shape[0] != 2 or mixture.shape[1] < 1 or not torch.isfinite(mixture).all():
        raise ValueError("separator input must be finite [2, samples]")
    if overlap != 2 or chunk_size < 10:
        raise ValueError("unsupported overlap-add geometry")
    original_length = mixture.shape[1]
    step, fade = chunk_size // overlap, chunk_size // 10
    border = chunk_size - step
    reflected = original_length > 2 * border
    if reflected:
        mixture = torch.nn.functional.pad(mixture, (border, border), mode="reflect")
    mixture = mixture.to(device)
    result = torch.zeros_like(mixture, dtype=torch.float32)
    counter = torch.zeros_like(mixture, dtype=torch.float32)
    window = torch.ones(chunk_size, device=device)
    window[:fade] *= torch.linspace(0, 1, fade, device=device)
    window[-fade:] *= torch.linspace(1, 0, fade, device=device)
    amp = torch.autocast("cuda", dtype=torch.float16) if mixed_precision else nullcontext()
    with torch.inference_mode(), amp:
        for first in range(0, mixture.shape[1], step):
            if check_budget:
                check_budget()
            part = mixture[:, first:first + chunk_size]
            length = part.shape[-1]
            if length < chunk_size:
                mode = "reflect" if length > chunk_size // 2 + 1 else "constant"
                part = torch.nn.functional.pad(part, (0, chunk_size - length), mode=mode)
            prediction = model(part.unsqueeze(0))[0]
            if tuple(prediction.shape) != (2, chunk_size) or not torch.isfinite(prediction).all():
                raise RuntimeError("separator returned wrong-size or non-finite samples; no repair")
            weight = window.clone()
            if first == 0:
                weight[:fade] = 1
            elif first + chunk_size >= mixture.shape[1]:
                weight[-fade:] = 1
            result[:, first:first + length] += prediction[:, :length] * weight[:length]
            counter[:, first:first + length] += weight[:length]
    if not torch.all(counter > 0):
        raise RuntimeError("overlap-add left uncovered samples")
    output = result / counter
    if reflected:
        output = output[:, border:-border]
    if output.shape[1] != original_length or not torch.isfinite(output).all():
        raise RuntimeError("separator changed the audio clock or emitted invalid samples")
    return output.float().cpu().contiguous()


class MelBandRuntime:
    def __init__(self, run: Path, cache: Path) -> None:
        from any2xsampa.adapters.phoneticxeus_runtime import check_gpu_budget, configure_cuda, enforce_offline

        self.identity, self.policy = verify_model_assets(run, "separator")
        enforce_offline(cache)
        self.torch = torch = configure_cuda()
        code = Path(self.identity["code_root"])
        self.config = safe_model_config(code / self.identity["model"]["config"])
        if "models" in sys.modules:
            raise RuntimeError("ambiguous upstream models namespace in isolated worker")
        package = types.ModuleType("models")
        package.__path__ = [str(code / "models")]
        sys.modules["models"] = package
        module = importlib.import_module("models.mel_band_roformer")
        self.model = module.MelBandRoformer(**self.config["model"])
        state = torch.load(self.identity["weight_path"], map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError("separator checkpoint must be a plain tensor state dictionary")
        self.model.load_state_dict(state, strict=True)
        del state
        self.model.eval().requires_grad_(False).to("cuda")
        if any(p.device.type != "cuda" for p in self.model.parameters()):
            raise RuntimeError("separator parameters did not reach CUDA")
        self.initial_memory = check_gpu_budget(torch)

    def separate(self, audio: Any) -> tuple[Any, dict]:
        import time
        from any2xsampa.adapters.phoneticxeus_runtime import check_gpu_budget

        torch = self.torch
        torch.cuda.reset_peak_memory_stats()
        before = time.monotonic()
        output = demix_bounded(torch, self.model, torch.from_numpy(audio), chunk_size=self.config["inference"]["chunk_size"],
            overlap=2, device="cuda", mixed_precision=True, check_budget=lambda: check_gpu_budget(torch))
        torch.cuda.synchronize()
        return output.numpy(), {"elapsed_seconds": time.monotonic() - before, "device": "cuda:0", "autocast": "float16",
            "parameter_dtype": "float32", "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "memory_after": check_gpu_budget(torch),
            "non_finite_repair": False, "all_parameters_frozen": True, "sample_count_preserved": True}
