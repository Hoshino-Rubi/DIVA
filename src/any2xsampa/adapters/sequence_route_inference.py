"""Opt-in replay of continued full-tail checkpoints, with explicit recipe identity."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from any2xsampa.adapters.phoneticxeus import PhoneticXeusPosteriorRecognizer, emission_payload
from any2xsampa.core.interfaces import UniversalPhoneRecognizer
from any2xsampa.core.models import PhoneSegment
from any2xsampa.training.limited_data_artifacts import read_json, verify_file, sha256_file, require_digest, write_json_new
from any2xsampa.training.singing_domain import architecture_identity, validate_audio_row


def verify_job(run: Path, job: Path) -> dict:
    result = read_json(job / "RESULT.json")
    if not result.get("complete") or result.get("fixed_epoch") != 20 or result.get("arm") not in {"control", "sequence_risk"}:
        raise ValueError("not a completed fixed-budget sequence-route checkpoint")
    verify_file(run / "POLICY.json", result["policy_sha256"])
    for path,digest in result["code_bindings"].items():
        verify_file(Path(path), digest)
    verify_file(Path(result["checkpoint_path"]), result["checkpoint_sha256"])
    verify_file(Path(result["parent_checkpoint_path"]), result["parent_checkpoint_sha256"])
    return result


def install_checkpoint(rt: Any, result: dict) -> Any:
    from safetensors.torch import load_file
    from any2xsampa.adapters.singing_domain_runtime import make_tail, load_trainable_state
    if rt.identity["sha256"] != result["architecture"]["sha256"]:
        raise ValueError("sequence-route architecture mismatch")
    tail = make_tail(rt.torch, rt.model, "full")
    load_trainable_state(tail, load_file(result["checkpoint_path"], device="cpu"))
    return tail.eval()


class SequenceRoutePosteriorRecognizer(UniversalPhoneRecognizer):
    def __init__(self, path: Path, *, fixture: bool = False) -> None:
        self.path, self.fixture = path, fixture

    def recognize(self, vocal_audio_path: Path, lyrics: str | None = None) -> list[PhoneSegment]:
        if lyrics is not None:
            raise ValueError("sequence-route recognition accepts audio only")
        payload = read_json(self.path)
        metadata = payload.get("producer", {}).get("model_metadata", {})
        identity = metadata.get("singing_domain_architecture", {})
        if architecture_identity(identity["snapshot"], identity["architecture"])["sha256"] != identity["sha256"]:
            raise ValueError("invalid sequence-route architecture digest")
        checkpoint = metadata.get("sequence_route_checkpoint", {})
        require_digest(checkpoint.get("sha256"), "checkpoint_sha256")
        require_digest(checkpoint.get("parent_sha256"), "parent_checkpoint_sha256")
        if checkpoint.get("arm") not in {"control", "sequence_risk"} or checkpoint.get("seed") not in {17,29,43}:
            raise ValueError("unsupported sequence-route recipe/seed")
        segments = PhoneticXeusPosteriorRecognizer(self.path, fixture=self.fixture).recognize(vocal_audio_path)
        for segment in segments:
            segment.metadata["sequence_route_checkpoint"] = checkpoint
        return segments


def save_posterior(rt: Any, result: dict, row: dict, logits: Any, source: dict, output: Path) -> Path:
    from safetensors.torch import save_file
    output.mkdir(parents=True, exist_ok=False)
    values = logits.detach().float().cpu().contiguous()
    metadata = {**rt.base.identity, "singing_domain_architecture":rt.identity,
        "sequence_route_checkpoint":{"arm":result["arm"], "seed":result["seed"],
            "held_song":result["held_song"], "sha256":result["checkpoint_sha256"],
            "parent_sha256":result["parent_checkpoint_sha256"]}}
    payload = emission_payload(values.numpy(), rt.vocabulary,
        input_audio=source["input_audio"], crop=source["crop"], model_metadata=metadata,
        geometry=source["geometry"], runtime=source["runtime"])
    raw = output / "logits.safetensors"
    save_file({"frame_logits":values},str(raw))
    vocabulary = output / "vocabulary.json"
    vocabulary.write_bytes((rt.base.snapshot / "src/model/xeusphoneme/resources/ipa_vocab.json").read_bytes())
    payload["raw_evidence"] = {"file":raw.name,"sha256":sha256_file(raw),
        "vocabulary_file":vocabulary.name,"vocabulary_sha256":sha256_file(vocabulary),
        "logit_shape":list(values.shape),"dtype":"float32"}
    path = output / "posterior.json"
    write_json_new(path,payload)
    return path
