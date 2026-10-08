"""Evaluate already-trained rendered-domain adapters on confirmed original clips."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from any2xsampa.adapters.phoneticxeus import MODEL_ID, REVISION, WEIGHT_SHA256, ctc_collapse
from any2xsampa.training.limited_data_artifacts import finite_number, read_json, sha256_file, verify_file
from any2xsampa.training.phone_error_metrics import phone_metrics


def _labels(vocabulary: Mapping[str, int]) -> list[str]:
    labels = [""] * len(vocabulary)
    for label, index in vocabulary.items():
        labels[index] = label
    return labels


def evaluate_adapter_transfer(
    rows: Sequence[Mapping[str, Any]],
    baseline_entries: Mapping[str, Mapping[str, Any]],
    adapter_reports: Mapping[int, Mapping[str, Any]],
    vocabulary_path: Path,
    *,
    conditional: bool = False,
) -> dict[str, Any]:
    try:
        import torch
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("adapter transfer requires torch and safetensors") from exc
    if not rows:
        raise ValueError("adapter transfer requires at least one confirmed original row")
    vocabulary = read_json(vocabulary_path)
    labels = _labels(vocabulary)
    if vocabulary.get("<blank>") != 0:
        raise ValueError("unexpected CTC vocabulary")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("adapter transfer requires explicit CUDA")
    head_path = Path(next(iter(adapter_reports.values()))["head"]["path"])
    verify_file(head_path, next(iter(adapter_reports.values()))["head"]["sha256"])
    head_state = load_file(str(head_path), device="cpu")
    if set(head_state) != {"weight", "bias"} or tuple(head_state["weight"].shape) != (428, 1024):
        raise ValueError("invalid frozen pretrained head")
    head = torch.nn.Linear(1024, 428, device=device)
    head.load_state_dict(head_state, strict=True)
    head.requires_grad_(False).eval()
    pairs: dict[str, list[tuple[list[str], list[str]]]] = {"baseline": []}
    details: dict[str, list[dict[str, Any]]] = {"baseline": []}
    for seed in sorted(adapter_reports):
        pairs[f"seed{seed}"] = []
        details[f"seed{seed}"] = []
    with torch.no_grad():
        for row in rows:
            entry = baseline_entries[row["example_id"]]
            raw_path = Path(entry["evidence_path"])
            verify_file(raw_path, entry["evidence_sha256"])
            tensors = load_file(str(raw_path), device="cpu")
            features = tensors["encoder_features"]
            logits = tensors["frame_logits"]
            if tuple(features.shape) != (entry["frames"], 1024) or tuple(logits.shape) != (entry["frames"], 428):
                raise ValueError("original feature evidence shape mismatch")
            if conditional:
                candidate = row["candidate_references"][0]
                if not candidate["mapping_complete"]:
                    raise ValueError("conditional transfer row has incomplete mapping")
                reference = list(candidate["reference_ipa"])
            else:
                if row.get("reviewed_reference") is None:
                    raise ValueError("official transfer requires a confirmed reviewed reference")
                reference = list(row["reviewed_reference"]["reference_ipa"])
            base_ids = ctc_collapse(logits.argmax(dim=-1).tolist())
            base_prediction = [labels[index] for index in base_ids]
            pairs["baseline"].append((reference, base_prediction))
            details["baseline"].append({"example_id": row["example_id"], "prediction": base_prediction})
            feature_cuda = features.to(device)
            for seed, report in sorted(adapter_reports.items()):
                checkpoint = Path(report["adapter"]["path"])
                verify_file(checkpoint, report["adapter"]["sha256"])
                state = load_file(str(checkpoint), device="cpu")
                if set(state) != {"down.bias", "down.weight", "up.bias", "up.weight"}:
                    raise ValueError("invalid residual adapter state")
                adapter = torch.nn.Sequential(
                    torch.nn.Linear(1024, 64), torch.nn.GELU(), torch.nn.Linear(64, 1024)
                ).to(device)
                adapter.load_state_dict({
                    "0.weight": state["down.weight"], "0.bias": state["down.bias"],
                    "2.weight": state["up.weight"], "2.bias": state["up.bias"],
                }, strict=True)
                transformed = feature_cuda + adapter(feature_cuda)
                adapted_logits = head(transformed)
                ids = ctc_collapse(adapted_logits.argmax(dim=-1).cpu().tolist())
                prediction = [labels[index] for index in ids]
                pairs[f"seed{seed}"].append((reference, prediction))
                details[f"seed{seed}"].append({"example_id": row["example_id"], "prediction": prediction})
    metrics = {name: phone_metrics(value) for name, value in pairs.items()}
    return {
        "schema_version": "original_adapter_transfer_report_v1",
        "mode": "rendered_domain_adapter_on_confirmed_original_audio" if not conditional else "rendered_domain_adapter_on_order_confirmed_offset_diagnostic",
        "model_metadata": {"model_id": MODEL_ID, "revision": REVISION, "weights_sha256": WEIGHT_SHA256},
        "rows": [row["example_id"] for row in rows], "metrics": metrics, "details": details,
        "device": "cuda:0", "encoder_retrained": False, "adapter_retrained": False,
        "checkpoint_selection_performed": False, "model_improvement_claim_allowed": False,
        "conditional_reference_only": conditional,
        "default_model_replaced": False, "test_predictions_opened": False,
    }
