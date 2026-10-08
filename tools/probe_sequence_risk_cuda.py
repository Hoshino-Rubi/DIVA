"""One disposable optimizer step verifies the new risk loss before quality runs."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from any2xsampa.training.limited_data_artifacts import read_json, write_json_new, verify_file


def probe(run: Path) -> None:
    from safetensors.torch import load_file
    from any2xsampa.adapters.singing_domain_runtime import SingingDomainRuntime, make_tail, load_trainable_state
    from any2xsampa.adapters.candidate_inference import serial_candidate_job
    from any2xsampa.adapters.phoneticxeus_monitor import ProcessResourceMonitor
    from any2xsampa.training.singing_domain_training import load_rows, predict
    from any2xsampa.training.singing_domain import partition
    from any2xsampa.training.singing_domain_losses import sequence_ctc
    from any2xsampa.training.sequence_risk import candidate_pool, ctc_candidate_risk
    freeze, spec = read_json(run / "FREEZE.json"), read_json(run / "POLICY.json")
    parent = Path(freeze["parents"]["singing"])
    checkpoint = read_json(parent / "architecture/full/freedom-flower/RESULT.json")
    verify_file(Path(checkpoint["checkpoint_path"]), checkpoint["checkpoint_sha256"])
    output = run / "risk-cuda-probe"
    output.mkdir()
    with serial_candidate_job(), ProcessResourceMonitor(output / "monitor", timeout_seconds=180):
        rt = SingingDomainRuntime(Path(read_json(parent / "FREEZE.json")["model_snapshot"]), output / "runtime-cache")
        torch = rt.torch
        rows,_ = load_rows(parent, torch)
        train,_ = partition(rows, "freedom-flower")
        row = next(r for r in train if r["audio_role"] == "original")
        tail = make_tail(torch, rt.model, "full")
        load_trainable_state(tail, load_file(checkpoint["checkpoint_path"]))
        rt.model.frontend.cpu()
        rt.model.preencoder.cpu()
        for block in list(rt.model.encoder.encoders)[:15]:
            block.cpu()
        rt.model.encoder.conditioning_layer.cpu()
        frozen = [p for p in rt.model.parameters() if not p.requires_grad]
        versions = [p._version for p in frozen]
        tail.eval()
        labels = [p for p,i in sorted(rt.vocabulary.items(), key=lambda x:x[1])]
        with torch.no_grad():
            before,_ = predict(torch, tail, row)
        pool = candidate_pool(before[0].cpu().numpy(), row["target_ids"], labels, spec["risk"])
        optimizer = torch.optim.AdamW([p for p in tail.parameters() if p.requires_grad],
            lr=spec["training"]["learning_rate"], weight_decay=spec["training"]["weight_decay"], foreach=False)
        tail.train()
        torch.cuda.reset_peak_memory_stats()
        logits,_ = predict(torch, tail, row)
        ctc = sequence_ctc(torch, logits, row["target_ids"])
        risk, details = ctc_candidate_risk(torch, logits, pool, temperature=spec["risk"]["posterior_temperature"])
        (ctc + risk).backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in tail.parameters() if p.requires_grad):
            raise ValueError("risk gradients missing/non-finite")
        torch.nn.utils.clip_grad_norm_([p for p in tail.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step()
        if [p._version for p in frozen] != versions:
            raise ValueError("frozen parameters changed in risk probe")
        write_json_new(output / "RESULT.json", {"example_id":row["example_id"], "held_song":"freedom-flower",
            "ctc":float(ctc.detach()),"risk":float(risk.detach()), "risk_details":details,
            "optimizer_steps":1, "probe_only":True, "temporary_weights_saved":False,
            "frozen_parameters_unchanged":True,"gradients_finite":True,
            "peak_reserved_bytes":torch.cuda.max_memory_reserved()})
        print("risk CUDA probe",float(ctc.detach()),float(risk.detach()),"peakGiB",torch.cuda.max_memory_reserved()/2**30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",required=True,type=Path)
    probe(parser.parse_args().run.resolve())
