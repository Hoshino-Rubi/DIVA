"""Save exact E evidence once; expensive independent codec checks run separately.

This changes serialization scheduling only, not head prediction, loss, optimizer,
sampling, epoch selection, reference denominator or safety limits.
"""
from pathlib import Path
from any2xsampa.training.limited_data_artifacts import read_json,write_json_new,sha256_file
from any2xsampa.training.event_head_training import predict
from any2xsampa.training.phone_error_metrics import phone_metrics
from input_native_probe_common import parameter_digest


def evaluate_fast(torch,head,arm,rows,spec,freeze,output):
    if arm!='E':raise ValueError('fast recovery serializer is for E only')
    from safetensors.numpy import save_file
    vocab=read_json(Path(freeze['vocabulary']['path']));labels={i:p for p,i in vocab.items()}
    digest=parameter_digest(head);details=[];head.eval()
    for row in rows:
        ids,q,b=predict(torch,head,row['features'],arm,spec)
        if not torch.isfinite(q).all() or not torch.isfinite(b).all():raise ValueError('nonfinite evaluation evidence')
        dest=output/row['example_id'];dest.mkdir(parents=True,exist_ok=False)
        raw=dest/'scores.safetensors';save_file({'frame_log_probs':q.numpy(),'boundary_logits':b.numpy()},str(raw))
        vp=dest/'vocabulary.json';vp.write_bytes(Path(freeze['vocabulary']['path']).read_bytes())
        pp=dest/'events.json';write_json_new(pp,{'schema_version':'segmental_event_evidence_v1','scoring_spec':'mean_logp_start_hold_v1',
            'reference_conditioning':False,'input_audio':row['audio_metadata'],'crop':row['crop'],'geometry':row['geometry'],
            'scores':{'file':raw.name,'sha256':sha256_file(raw)},'vocabulary':{'file':vp.name,'sha256':sha256_file(vp)},
            'decoded_event_ids':ids,'head_sha256':digest,'times_are_ground_truth':False,
            'independent_codec_verification_pending':True})
        prediction=[labels[i] for i in ids]
        details.append({'example_id':row['example_id'],'song_id':row['song_id'],'audio_role':row['audio_role'],
            'reference':row['target_ipa'],'prediction':prediction,'metrics':phone_metrics([(row['target_ipa'],prediction)]),
            'event_path':str(pp),'event_sha256':sha256_file(pp),'scores_path':str(raw),'scores_sha256':sha256_file(raw),
            'independent_codec_verification_pending':True})
    return {'details':details,'original':phone_metrics([(r['reference'],r['prediction']) for r in details if r['audio_role']=='original']),
        'rendered':phone_metrics([(r['reference'],r['prediction']) for r in details if r['audio_role']=='rendered'])}
