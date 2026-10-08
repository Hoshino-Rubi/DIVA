from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import pytest

from any2xsampa.training.sequence_risk import (
    weighted_phone_distance, candidate_pool, ctc_candidate_risk, whole_sequence_ctc,
)
from any2xsampa.training.whole_sequence_reference import audit_song, check_clock_anchors
from any2xsampa.training.limited_data_artifacts import read_json

POLICY = read_json(Path(__file__).resolve().parents[1] / "src/any2xsampa/config/sequence_routes_v1.json")


def test_weighted_edits_preserve_difficult_phones():
    spec = POLICY["risk"]
    assert weighted_phone_distance(["kʰ"], [], spec) == 2
    assert weighted_phone_distance([], ["kʰ"], spec) == 3
    assert weighted_phone_distance(["kʰ"], ["k"], spec) == 2
    assert weighted_phone_distance(["k"], ["kʰ"], spec) == 2
    assert weighted_phone_distance(["i", "i"], ["i"], spec) == 1
    assert weighted_phone_distance(["i"], ["i", "i"], spec) == 2
    assert weighted_phone_distance(["kʰ", "i"], ["kʰ", "i"], spec) == 0


def test_training_reference_does_not_change_acoustic_candidates():
    logits = np.array([[0.,2.,1.],[3.,0.,1.],[0.,1.,2.]])
    labels = ["<blank>", "i", "kʰ"]
    first = candidate_pool(logits, [1], labels, POLICY["risk"])
    second = candidate_pool(logits, [2], labels, POLICY["risk"])
    acoustic = lambda p: {tuple(r["ids"]) for r in p["candidates"] if any(o.startswith("audio_only") for o in r["origins"])}
    assert acoustic(first) == acoustic(second)
    assert not first["decode_received_reference"]
    assert any(r["ids"] == [1] and r["cost"] == 0 for r in first["candidates"])


def test_risk_matches_exact_ctc_sequence_probability_and_gradient():
    torch = pytest.importorskip("torch")
    from any2xsampa.adapters.dual_ctc import sequence_log_probability
    data = np.array([[[1.,2.,0.],[2.,0.,1.],[0.,1.,2.]]], dtype=np.float64)
    pool = {"frame_count": 3, "candidates": [
        {"ids": [], "cost": 2.}, {"ids": [1], "cost": 0.},
        {"ids": [1,1], "cost": 1.}, {"ids": [2], "cost": .5}]}
    logits = torch.tensor(data, requires_grad=True)
    loss, details = ctc_candidate_risk(torch, logits, pool)
    scores = np.array([sequence_log_probability(data[0], r["ids"]) for r in pool["candidates"]])
    probability = np.exp(scores/2 - (scores/2).max())
    probability /= probability.sum()
    assert details["exact_sequence_log_probabilities"] == pytest.approx(scores, abs=1e-10)
    assert float(loss.detach()) == pytest.approx(float(probability @ [2.,0.,1.,.5]), abs=1e-10)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    delta = 1e-6
    upper, lower = data.copy(), data.copy()
    upper[0,1,1] += delta
    lower[0,1,1] -= delta
    numerical = (ctc_candidate_risk(torch, torch.tensor(upper), pool)[0] - ctc_candidate_risk(torch, torch.tensor(lower), pool)[0])/(2*delta)
    assert float(logits.grad[0,1,1]) == pytest.approx(float(numerical), abs=1e-7)


def test_padding_does_not_enter_risk_or_gradient():
    torch = pytest.importorskip("torch")
    data = torch.tensor([[[1.,2.],[2.,1.],[1.,2.],[float("nan"),float("nan")]]], requires_grad=True)
    pool = {"frame_count":3,"candidates":[{"ids":[1],"cost":0.},{"ids":[],"cost":1.}]}
    loss,_ = ctc_candidate_risk(torch, data, pool, valid_frames=3)
    loss.backward()
    assert torch.isfinite(loss) and torch.equal(data.grad[:,3], torch.zeros_like(data.grad[:,3]))
    with pytest.raises(ValueError, match="invalid/stale"):
        ctc_candidate_risk(torch, data, pool)


@pytest.mark.parametrize("ids", [[0], [1,1,1], [10]])
def test_invalid_or_infeasible_risk_sequences(ids):
    torch = pytest.importorskip("torch")
    pool = {"frame_count":3,"candidates":[{"ids":ids,"cost":1.}]}
    with pytest.raises(ValueError, match="invalid candidate"):
        ctc_candidate_risk(torch, torch.zeros(1,3,3), pool)


def test_whole_sequence_retains_repeats_and_chunk_gradients():
    torch = pytest.importorskip("torch")
    from any2xsampa.training.singing_domain_losses import sequence_ctc
    first = torch.tensor([[[0.,2.],[2.,0.]]], requires_grad=True)
    second = torch.tensor([[[0.,2.],[2.,0.]]], requires_grad=True)
    selections = [{"keep_start_frame":0,"keep_end_frame":2,"global_start_frame":0},
                  {"keep_start_frame":0,"keep_end_frame":2,"global_start_frame":2}]
    combined = whole_sequence_ctc(torch, [first,second], selections, [1,1])
    expected = sequence_ctc(torch, torch.cat([first,second], dim=1), [1,1])
    assert torch.equal(combined, expected)
    combined.backward()
    assert torch.isfinite(first.grad).all() and torch.isfinite(second.grad).all()
    selections[1]["global_start_frame"] = 1
    with pytest.raises(ValueError, match="gap, overlap"):
        whole_sequence_ctc(torch, [first,second], selections, [1,1])


def song_fixture():
    notes = [{"note_id":f"promise-n{i:03d}","note_index":i,"mapping_status":"resolved_proxy",
              "model_target_ids":[1],"model_target_ipa":["i"],"unresolved_tokens":[],
              "start_sec":float(i),"end_sec":float(i)+.5} for i in range(1,4)]
    song = {"song_id":"promise","split":"train","notes":notes,
            "source_claims":{"phoneme_correctness":"user_asserted_correct"},
            "original_audio":{"sha256":"a"*64},"human_vsqx":{"sha256":"b"*64}}
    rows = [{"song_id":"promise","example_id":str(i),"status":"human_confirmed","audio_role":"original",
             "audio_sha256":"a"*64,"source_vsqx_sha256":"b"*64,"start_sec":float(i),"end_sec":float(i)+.5,
             "source_phones":[{"note_id":n["note_id"],"phone_index":0,"phone":"i","audibility":"full"}]}
             for i,n in enumerate(notes,1)]
    return song,rows


def test_whole_reference_does_not_promote_source_approval_to_timing():
    song,rows = song_fixture()
    report = audit_song(song, rows)
    assert report["source_structure_passed"]
    assert not report["human_whole_song_original_timeline_claimed"]
    path = {"anchors":[{"original_sec":float(i),"rendered_sec":float(i)+10,
        "rendered_min_sec":float(i)+10,"rendered_max_sec":float(i)+10} for i in range(5)]}
    result = check_clock_anchors(report, [path,path])
    assert not result["machine_sequence_training_eligible"]
    assert not result["existing_reviewed_windows_modified"]


def test_unknown_mapping_excludes_complete_song_and_hash_mismatch_rejected():
    song,rows = song_fixture()
    song["notes"].append({"note_id":"promise-n004","note_index":4,"mapping_status":"unresolved",
        "model_target_ids":[],"model_target_ipa":[],"unresolved_tokens":["unknown"],"start_sec":4.,"end_sec":5.})
    report = audit_song(song,rows)
    assert report["unresolved_tokens"] == ["unknown"] and not report["source_structure_passed"]
    rows[0]["source_vsqx_sha256"] = "c"*64
    with pytest.raises(ValueError, match="source/role mismatch"):
        audit_song(song,rows)


def test_language_condition_requires_explicit_supported_value():
    from any2xsampa.adapters.powsm_language import validate_condition
    assert validate_condition("<unk>") == "<unk>"
    assert validate_condition("<yue>") == "<yue>"
    for value in (None,"<eng>","lyrics"):
        with pytest.raises(ValueError):
            validate_condition(value)


def test_whole_review_pending_export_and_strict_approval_scope():
    from any2xsampa.training.whole_sequence_review import validate_answers
    bindings={"original_audio":{"path":"source.wav","sha256":"a"*64}}
    packet={"schema_version":"whole_sequence_review_packet_v1","items":[
        {"review_id":"S01","song_id":"freedom-flower","source_bindings":bindings}]}
    answers={"schema_version":"whole_sequence_review_answers_v1","packet_sha256":"b"*64,"answers":[
        {"review_id":"S01","status":"pending","phone_timing_confirmed":False,
         "source_bindings":bindings,"statement":"尚不确定重复段"}]}
    pending=validate_answers(packet,answers,packet_sha256="b"*64)
    assert pending["pending"][0]["statement"] == "尚不确定重复段"
    assert not pending["starts_training"]
    answers["answers"][0]["status"]="sequence_confirmed"
    with pytest.raises(ValueError,match="scope affirmation"):
        validate_answers(packet,answers,packet_sha256="b"*64)
    answers["answers"][0]["explicit_sequence_scope_confirmed"]=True
    confirmed=validate_answers(packet,answers,packet_sha256="b"*64)
    assert len(confirmed["accepted"])==1 and not confirmed["accepted"][0]["phone_timing_confirmed"]
    assert not confirmed["reference_times_modified"]
    answers["answers"][0]["phone_timing_confirmed"]=True
    with pytest.raises(ValueError,match="does not establish phone timing"):
        validate_answers(packet,answers,packet_sha256="b"*64)


def test_whole_review_rejects_packet_and_source_mismatch():
    from any2xsampa.training.whole_sequence_review import validate_answers
    packet={"schema_version":"whole_sequence_review_packet_v1","items":[
        {"review_id":"S01","song_id":"freedom-flower","source_bindings":{"source":"one"}}]}
    answers={"schema_version":"whole_sequence_review_answers_v1","packet_sha256":"b"*64,
        "answers":[{"review_id":"S01","status":"pending","phone_timing_confirmed":False,
                    "source_bindings":{"source":"two"},"statement":"保留原话"}]}
    with pytest.raises(ValueError,match="packet hash"):
        validate_answers(packet,answers,packet_sha256="a"*64)
    with pytest.raises(ValueError,match="source versions"):
        validate_answers(packet,answers,packet_sha256="b"*64)


def test_final_release_checks_counts_insertions_and_rendered_aspiration():
    import importlib.util
    path=Path(__file__).resolve().parents[1]/"tools/evaluate_sequence_routes.py"
    spec=importlib.util.spec_from_file_location("sequence_release_fixture",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def metric(ref,s,d,i,asp_ref,asp_errors):
        return {"reference_phones":ref,"edit_distance":s+d+i,"substitutions":s,"deletions":d,"insertions":i,"per":(s+d+i)/ref,
            "groups":{"aspirated":{"reference_count":asp_ref,"substitutions":asp_errors,"deletions":0,"insertions":0,"per":asp_errors/asp_ref},
                      "glottal_h":{"reference_count":5,"substitutions":2,"deletions":0,"insertions":0,"per":.4}}}
    original=metric(214,52,17,11,6,1);old=metric(214,56,25,11,6,1);rendered=metric(679,200,0,0,25,12)
    assert module.release_gate(original,rendered,old,rendered,POLICY["release"])["passed"]
    worse=copy.deepcopy(original);worse.update(substitutions=45,insertions=18)
    assert "original_insertions" in module.release_gate(worse,rendered,old,rendered,POLICY["release"])["failures"]
    bad_render=copy.deepcopy(rendered);bad_render["groups"]["aspirated"].update(substitutions=17,per=17/25)
    assert "rendered_aspirated_protected" in module.release_gate(original,bad_render,old,rendered,POLICY["release"])["failures"]
    invalid=copy.deepcopy(original);invalid["per"]=float("nan")
    with pytest.raises(ValueError,match="metric arithmetic"):
        module.release_gate(invalid,rendered,old,rendered,POLICY["release"])
