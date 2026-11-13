"""Candidate-4 mechanism checks with fakes; never load weights or initialize CUDA.

Local equivalence oracles execute selected AST function definitions only. The
historical modules (including melband_runtime's training import) are never
imported. Oracle tests skip when the private historical source is unavailable.
Known runtime defects remain ordinary failing regression tests, not xfails.
"""

import ast
from contextlib import nullcontext
from copy import deepcopy
import copy
from pathlib import Path
from types import SimpleNamespace
import threading
from typing import Any, Callable

import pytest

from any2xsampa.adapters import cantonese4_runtime as runtime
from any2xsampa.adapters.cantonese4_lexicon import completion_gate
from any2xsampa.local import cantonese4 as local


WORKSPACE = Path(__file__).resolve().parents[3]
CHUNK, STEP, FADE = 352800, 176400, 35280


def _functions_only(path, names, **namespace):
    if not path.is_file():
        pytest.skip("Private frozen-source equivalence oracle is unavailable")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == set(names)
    scope = {"Any": Any, "Callable": Callable, "Path": Path,
             "JSON": dict, "copy": copy, "nullcontext": nullcontext, **namespace}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


@pytest.fixture(scope="module")
def frozen_probe():
    return _functions_only(
        WORKSPACE / ".state/runs/cantonese-parallel-20261113/whisper_probe.py",
        ("generation_settings", "infer_item", "error_info"),
        MAX_NEW_TOKENS=400, RATE=16000, THREADS=4,
        read_crop=lambda *_: "synthetic samples; no audio access",
    )


@pytest.fixture(scope="module")
def frozen_separator():
    return _functions_only(
        WORKSPACE / "src/any2xsampa/adapters/melband_runtime.py", ("demix_bounded",),
    ).demix_bounded


class FakeConfig(SimpleNamespace):
    def __init__(self, **changes):
        super().__init__(**{
            "num_beams": 5, "do_sample": True, "max_new_tokens": 123,
            "num_return_sequences": 1, "return_timestamps": False,
            "return_token_timestamps": False, "eos_token_id": 2,
            "forced_eos_token_id": None, "forced_decoder_ids": [[1, 77]],
            "suppress_tokens": [3, 4], "begin_suppress_tokens": [5],
            "decoder_start_token_id": 1, "language": "yue", "task": "transcribe",
            **changes,
        })

    def to_dict(self):
        return deepcopy(vars(self))


class FakeTensor:
    def __init__(self, value):
        self.value, self.devices = value, []

    def to(self, device):
        self.devices.append(device)
        assert device == "cpu"
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return deepcopy(self.value)


class FakeProcessor:
    def __init__(self, *, failure=None, decoded="  la  la HnG ", fields=None):
        self.failure, self.decoded = failure, decoded
        self.inputs, self.decode_calls = [], []
        self.fields = fields if fields is not None else {
            "input_features": FakeTensor("features"), "attention_mask": FakeTensor("mask"),
        }

    def __call__(self, samples, **kwargs):
        self.inputs.append((samples, kwargs))
        assert kwargs == {"sampling_rate": 16000, "return_tensors": "pt"}
        if self.failure == "processor":
            raise ValueError("synthetic processor failure")
        return self.fields

    def batch_decode(self, tokens, *, skip_special_tokens, clean_up_tokenization_spaces):
        self.decode_calls.append((deepcopy(tokens), skip_special_tokens))
        assert clean_up_tokenization_spaces is False
        key = "decoded" if skip_special_tokens else "raw_decoded"
        if self.failure == key:
            raise ValueError("synthetic " + key + " failure")
        return [self.decoded if skip_special_tokens else "<start>  la  la HnG <end>"]


class FakeModel:
    def __init__(self, tokens=None, *, config=None, failure=False):
        self.generation_config = config if config is not None else FakeConfig()
        self.config = SimpleNamespace(max_target_positions=448)
        self.tokens = [1, 11, 10, 2] if tokens is None else tokens
        self.failure, self.calls = failure, []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        assert set(kwargs) in (
            {"input_features", "attention_mask", "generation_config", "temperature", "return_dict_in_generate"},
            {"input_features", "generation_config", "temperature", "return_dict_in_generate"},
        )
        assert kwargs["temperature"] == 0.0
        assert kwargs["return_dict_in_generate"] is True
        if self.failure:
            raise RuntimeError("synthetic generate failure")
        return SimpleNamespace(sequences=FakeTensor([self.tokens]))


FAKE_TORCH = SimpleNamespace(inference_mode=nullcontext)


def _infer(processor, model):
    events = []
    result = runtime.infer(
        FAKE_TORCH, (processor, model), "synthetic samples; no audio access",
        lambda name, **fields: events.append((name, fields)),
    )
    return result, events


def _old_infer(probe, processor, model):
    config, parameters = probe.generation_settings(model)
    return probe.infer_item(
        {"id": "synthetic-window"}, Path("."), (FAKE_TORCH, processor, model),
        config, parameters, lambda state, fields: {"state": state, **fields},
    )


def test_generation_overrides_and_nested_pinned_fields_match_frozen(frozen_probe):
    model = FakeModel()
    before = model.generation_config.to_dict()
    actual = runtime.generation_settings(model)
    expected, _ = frozen_probe.generation_settings(model)
    assert actual.to_dict() == expected.to_dict() == {
        **before, "num_beams": 1, "do_sample": False, "max_new_tokens": 400,
    }
    actual.forced_decoder_ids[0][1] = 999
    actual.suppress_tokens.clear()
    assert model.generation_config.to_dict() == before


@pytest.mark.parametrize("change", [
    {"num_return_sequences": 2}, {"return_timestamps": True},
    {"return_token_timestamps": True}, {"eos_token_id": None},
])
def test_generation_settings_reject_same_unsupported_configs(frozen_probe, change):
    model = FakeModel(config=FakeConfig(**change))
    with pytest.raises(ValueError):
        frozen_probe.generation_settings(model)
    with pytest.raises(ValueError):
        runtime.generation_settings(model)
    assert model.calls == []


@pytest.mark.parametrize("tokens,eos_ids,forced,decoded", [
    ([1, 11, 2], 2, None, "  la  la HnG "),
    ([1] + [10] * 400, 2, None, "la"),
    ([1, 11, 2], 2, 2, "la"),
    ([1, 11], 2, 2, "la"),
    ([1, 11, 7], [2, 7], None, "la"),
    ([1, 2, 0, 0], 2, None, ""),
])
def test_eos_forced_eos_and_raw_decode_match_frozen(
    frozen_probe, tokens, eos_ids, forced, decoded,
):
    config = FakeConfig(eos_token_id=eos_ids, forced_eos_token_id=forced)
    model, processor = FakeModel(tokens, config=config), FakeProcessor(decoded=decoded)
    expected = _old_infer(frozen_probe, FakeProcessor(decoded=decoded), FakeModel(tokens, config=config))
    actual, events = _infer(processor, model)
    keys = ("status", "generate_started", "generate_completed", "token_ids",
            "eos_observed", "truncated_or_incomplete", "truncation_status", "raw_decoded", "decoded")
    assert {key: actual[key] for key in keys} == {key: expected[key] for key in keys}
    assert completion_gate(actual) == completion_gate(expected)
    assert [name for name, _ in events] == ["generate_started", "generate_completed"]
    assert len(model.calls) == 1
    assert all(value.devices == ["cpu"] for value in processor.fields.values())
    assert [skip for _, skip in processor.decode_calls] == [False, True]
    assert model.generation_config.to_dict() == config.to_dict()


def test_generate_failure_retains_raw_once_without_decode_or_retry(frozen_probe):
    processor, model = FakeProcessor(), FakeModel(failure=True)
    actual, events = _infer(processor, model)
    expected = _old_infer(frozen_probe, FakeProcessor(), FakeModel(failure=True))
    for key in ("status", "generate_started", "generate_completed"):
        assert actual[key] == expected[key]
    assert actual["error"]["type"] == expected["error"]["type"] == "RuntimeError"
    assert actual["error"]["message"] == expected["error"]["message"]
    assert len(model.calls) == 1 and processor.decode_calls == []
    assert [name for name, _ in events] == ["generate_started", "prediction_failed"]


@pytest.mark.parametrize("failure", ["raw_decoded", "decoded"])
def test_decode_failure_retains_generated_token_and_eos_evidence(frozen_probe, failure):
    expected = _old_infer(frozen_probe, FakeProcessor(failure=failure), FakeModel())
    actual, _ = _infer(FakeProcessor(failure=failure), FakeModel())
    assert actual["status"] == expected["status"] == "failed"
    assert actual["generate_started"] is actual["generate_completed"] is True
    required = ["token_ids", "eos_observed", "truncated_or_incomplete", "truncation_status"]
    if failure == "decoded":
        required.append("raw_decoded")
    assert {key: actual.get(key) for key in required} == {key: expected[key] for key in required}, (
        "Decode failure must not discard token/EOS evidence or an already completed raw decode"
    )


def test_infer_config_failure_returns_failed_raw_instead_of_aborting_window():
    model = FakeModel(config=FakeConfig(eos_token_id=None))
    actual, events = _infer(FakeProcessor(), model)
    assert actual["status"] == "failed" and "error" in actual
    assert actual["generate_started"] is actual["generate_completed"] is False
    assert model.calls == []
    assert [name for name, _ in events] == ["prediction_failed"]


@pytest.mark.parametrize("bad", [-1, True, 2.0])
def test_invalid_token_ids_fail_like_frozen_probe(frozen_probe, bad):
    tokens = [1, bad, 2]
    expected = _old_infer(frozen_probe, FakeProcessor(), FakeModel(tokens))
    processor = FakeProcessor()
    actual, _ = _infer(processor, FakeModel(tokens))
    assert actual["status"] == expected["status"] == "failed", "Do not accept negative, boolean or floating token IDs"
    assert processor.decode_calls == []


@pytest.mark.parametrize("decoded", [None, 17])
def test_non_text_decode_is_failed_raw_like_frozen_probe(frozen_probe, decoded):
    expected = _old_infer(frozen_probe, FakeProcessor(decoded=decoded), FakeModel())
    actual, _ = _infer(FakeProcessor(decoded=decoded), FakeModel())
    assert actual["status"] == expected["status"] == "failed", "Decoder output must be text before claiming success"


@pytest.mark.parametrize("fields", [{"decoder_input_ids": FakeTensor([7])}, {}])
def test_unexpected_processor_fields_fail_before_model(fields):
    model = FakeModel()
    actual, _ = _infer(FakeProcessor(fields=fields), model)
    assert actual["status"] == "failed" and actual["generate_started"] is False
    assert model.calls == []


class CPUTorchSpy:
    """Record CUDA requests but perform every numerical operation on CPU."""

    def __init__(self, torch):
        self.torch, self.devices, self.padding, self.autocasts = torch, [], [], []
        self.nn = SimpleNamespace(functional=SimpleNamespace(pad=self.pad))

    def __getattr__(self, name):
        return getattr(self.torch, name)

    def pad(self, tensor, padding, *, mode):
        self.padding.append((tuple(tensor.shape), padding, mode))
        assert tensor.device.type == "cpu"
        return self.torch.nn.functional.pad(tensor, padding, mode=mode)

    def ones(self, *args, device):
        self.devices.append(("ones", device))
        return self.torch.ones(*args, device="cpu")

    def linspace(self, *args, device):
        self.devices.append(("linspace", device))
        return self.torch.linspace(*args, device="cpu")

    def autocast(self, device, *, dtype):
        self.autocasts.append((device, dtype))
        return nullcontext()


class TinySeparator:
    def __init__(self, torch, *, identity=False, invalid=None):
        self.torch, self.identity, self.invalid = torch, identity, invalid
        self.parts = []

    def __call__(self, part):
        assert part.device.type == "cpu" and tuple(part.shape) == (1, 2, CHUNK)
        assert not self.torch.is_grad_enabled()
        self.parts.append(part.clone())
        if self.invalid == "shape":
            return part[:, :, :-1]
        if self.invalid == "nan":
            return self.torch.full_like(part, float("nan"))
        if self.invalid == "inf":
            return self.torch.full_like(part, float("inf"))
        if self.identity:
            return part
        # Position and call terms expose changed padding, overlap weights or
        # window order that an identity-only model would conceal.
        ramp = self.torch.linspace(-0.2, 0.3, CHUNK, device="cpu")
        return part * 0.5 + ramp + len(self.parts) * 0.03125


@pytest.fixture
def cpu_demix(monkeypatch):
    torch = pytest.importorskip("torch")
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    original_to = torch.Tensor.to
    transfers = []

    def cpu_to(tensor, *args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith("cuda"):
            transfers.append(args[0])
            args = ("cpu", *args[1:])
        if str(kwargs.get("device", "")).startswith("cuda"):
            transfers.append(str(kwargs["device"]))
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("No CUDA initialization or weight loading is allowed")

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    try:
        yield torch, transfers
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("length", [
    1, STEP + 1, STEP + 2, CHUNK - 1, CHUNK, CHUNK + 1, CHUNK + STEP + 3,
])
def test_demix_matches_original_padding_overlap_edges_and_clock(
    cpu_demix, frozen_separator, length,
):
    torch, transfers = cpu_demix
    wave = torch.linspace(-0.4, 0.6, length, dtype=torch.float32)
    mixture = torch.stack((wave, -wave * 0.75))
    before = mixture.clone()
    old_spy, new_spy = CPUTorchSpy(torch), CPUTorchSpy(torch)
    old_model, new_model = TinySeparator(torch), TinySeparator(torch)
    old_checks, new_checks = [], []
    expected = frozen_separator(old_spy, old_model, mixture, chunk_size=CHUNK, overlap=2,
                                device="cuda:0", mixed_precision=True,
                                check_budget=lambda: old_checks.append(True))
    actual = runtime.demix(new_spy, new_model, mixture, lambda: new_checks.append(True))
    assert torch.equal(actual, expected), "Frozen and new overlap-add must be numerically identical"
    assert actual.shape == (2, length) and actual.device.type == "cpu"
    assert actual.dtype == torch.float32 and actual.is_contiguous()
    assert torch.equal(mixture, before)
    assert old_checks == new_checks and len(new_checks) == len(new_model.parts)
    assert all(torch.equal(a, b) for a, b in zip(old_model.parts, new_model.parts, strict=True))
    assert old_spy.padding == new_spy.padding
    assert old_spy.devices == new_spy.devices == [
        ("ones", "cuda:0"), ("linspace", "cuda:0"), ("linspace", "cuda:0"),
    ]
    assert old_spy.autocasts == new_spy.autocasts == [("cuda", torch.float16)]
    assert transfers == ["cuda:0", "cuda:0"]
    # Independently assert branch thresholds rather than only comparing copies.
    if length > CHUNK:
        assert new_spy.padding[0] == ((2, length), (STEP, STEP), "reflect")
    if length in (STEP + 1, STEP + 2):
        assert new_spy.padding[0][2] == ("constant" if length == STEP + 1 else "reflect")


def test_demix_identity_preserves_first_last_and_overlap_samples(cpu_demix):
    torch, _ = cpu_demix
    length = CHUNK + STEP + 1
    wave = torch.linspace(-0.5, 0.5, length, dtype=torch.float32)
    mixture = torch.stack((wave, wave.flip(0)))
    actual = runtime.demix(CPUTorchSpy(torch), TinySeparator(torch, identity=True), mixture, lambda: None)
    torch.testing.assert_close(actual, mixture, rtol=1e-6, atol=1e-7)
    assert torch.equal(actual[:, [0, -1]], mixture[:, [0, -1]])


@pytest.mark.parametrize("invalid", ["shape", "nan", "inf"])
def test_demix_rejects_invalid_model_samples_without_repair(cpu_demix, frozen_separator, invalid):
    torch, _ = cpu_demix
    mixture = torch.zeros((2, STEP + 2), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        frozen_separator(CPUTorchSpy(torch), TinySeparator(torch, invalid=invalid), mixture,
                         chunk_size=CHUNK, overlap=2, device="cuda:0", mixed_precision=True)
    with pytest.raises(RuntimeError):
        runtime.demix(CPUTorchSpy(torch), TinySeparator(torch, invalid=invalid), mixture, lambda: None)


def test_demix_checks_cancel_before_forward(cpu_demix):
    torch, _ = cpu_demix
    model = TinySeparator(torch)

    def cancelled():
        raise RuntimeError("synthetic cancellation")

    with pytest.raises(RuntimeError, match="synthetic cancellation"):
        runtime.demix(CPUTorchSpy(torch), model, torch.zeros((2, 5)), cancelled)
    assert model.parts == []


def test_saved_fallback_and_both_failed_windows_preserve_raw_evidence(monkeypatch):
    saved = {}

    def save_json(path, value):
        saved[path.name] = deepcopy(value)

    def save_text(path, text, **kwargs):
        assert kwargs == {"encoding": "utf-8"}
        saved[path.name] = text

    monkeypatch.setattr(local, "write", save_json)
    monkeypatch.setattr(Path, "write_text", save_text)
    failed, _ = _infer(FakeProcessor(), FakeModel(failure=True))
    complete, _ = _infer(FakeProcessor(decoded="la la HnG"), FakeModel())
    empty, _ = _infer(FakeProcessor(decoded=""), FakeModel())
    forced, _ = _infer(FakeProcessor(), FakeModel(config=FakeConfig(forced_eos_token_id=2)))
    windows = [
        {"start_sec": 0, "end_sec": 18, "small": failed, "base": complete},
        {"start_sec": 18, "end_sec": 36, "small": forced, "base": failed},
        {"start_sec": 36, "end_sec": 37, "small": empty, "base": None},
    ]
    before = deepcopy(windows)
    result = local.save_predictions(Path("synthetic-output"), windows, source={"synthetic": True},
                                    mode="replay", cancel=threading.Event())
    rows = result["windows"]
    assert len(rows) == 3 and windows == before
    assert [row["selection"]["selected_model"] for row in rows] == ["base", None, "small"]
    assert [row["raw"] for row in rows] == [{"small": w["small"], "base": w["base"]} for w in windows]
    assert rows[0]["projection"]["scored_tokens"] == ["lɐ", "lɐ", "<UNK:2>"]
    assert rows[1]["jyutping"] == "<UNK:completion>"
    assert rows[1]["projection"]["unknown_tokens"] == ["<UNK:completion>"]
    assert rows[2]["projection"]["phones"] == []
    assert result["phone_times"] is None and result["tone"] == "unknown"
    assert result["repetition_cleanup"] is False
    assert saved["result.json"] == result
    assert set(saved) == {"result.json", "jyutping.txt", "phones.ipa.txt", "phones.xsampa.txt"}
