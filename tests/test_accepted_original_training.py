from copy import deepcopy
from pathlib import Path
import pytest
from any2xsampa.training.original_adaptation import validate_original_index, _load_evidence, _gate, _rendered_gate
from any2xsampa.training.reviewed_training_reference import validate_training_reference, SCHEMA
from any2xsampa.training.limited_data_artifacts import sha256_object,sha256_file,write_json_new
from any2xsampa.adapters.phoneticxeus import MODEL_ID,REVISION,WEIGHT_SHA256


def cache_case():
    rows=[dict(example_id='a',song_id='promise',split='train',audio_role='original',audio_sha256='a'*64,start_sec=0,end_sec=10,model_target_ids=[4,4]),
          dict(example_id='b',song_id='shalong',split='dev',audio_role='original',audio_sha256='b'*64,start_sec=0,end_sec=18,model_target_ids=[4])]
    entries=[{**r,'frames':4,'test_outputs_sealed':False,'vocabulary_sha256':'v','model_snapshot_lock_sha256':'s'} for r in rows]
    index=dict(purpose='original-reviewed-v2',all_parameters_frozen=True,no_labels_or_score_received=True,entries=entries,
        model_metadata=dict(model_id=MODEL_ID,revision=REVISION,weights_sha256=WEIGHT_SHA256,snapshot_lock_sha256='s'))
    return index,rows


def test_combined_train_dev_cache_loads_once(tmp_path):
    torch=pytest.importorskip('torch');save=pytest.importorskip('safetensors.torch').save_file
    index,rows=cache_case();validate_original_index(index,rows,'v')
    for entry in index['entries']:
        path=tmp_path/(entry['example_id']+'.safetensors')
        save({'encoder_features':torch.zeros(4,1024),'frame_logits':torch.zeros(4,428)},str(path))
        entry.update(evidence_path=str(path),evidence_sha256=sha256_file(path))
    result=_load_evidence(index,rows,torch,root=tmp_path)
    assert [r['split'] for r in result]==['train','dev']
    with pytest.raises(ValueError,match='exactly'):_load_evidence(index,rows[:1],torch,root=tmp_path)


@pytest.mark.parametrize('field,value',[('audio_role','rendered'),('split','test'),('song_id','ke-yi-ke-zai'),('start_sec',1),('vocabulary_sha256','wrong'),('test_outputs_sealed',True),('frames',2)])
def test_original_cache_rejects_invalid_binding(field,value):
    index,rows=cache_case();index['entries'][0][field]=value
    with pytest.raises(ValueError):validate_original_index(index,rows,'v')


def reference_case(tmp_path):
    phone=dict(phone='a',audibility='full',evidence_scope='inherited_source_interior')
    absent=dict(phone='i',audibility='absent',evidence_scope='explicit_human_edge')
    row=dict(example_id='a',song_id='promise',split='train',audio_role='original',audio_sha256='a'*64,start_sec=0,end_sec=10,
             status='human_confirmed',phones=[phone],source_phones=[phone,absent],omitted_phones=[absent],edge_evidence={'start':{},'end':{}})
    path=tmp_path/'source.json';write_json_new(path,{'examples':[row]})
    ref=dict(schema_version=SCHEMA,examples=[row],reference_sources=[dict(path=str(path),sha256=sha256_file(path))]);ref['manifest_hash']=sha256_object(ref)
    return ref


def test_absent_evidence_preserved_without_geometry(tmp_path):
    value=reference_case(tmp_path);validate_training_reference(value)
    assert 'boundary_state' not in value['examples'][0]
    value['examples'][0]['phones'].append(value['examples'][0]['omitted_phones'][0]);value['manifest_hash']=sha256_object({k:v for k,v in value.items() if k!='manifest_hash'})
    with pytest.raises(ValueError,match='changed'):validate_training_reference(value)


def test_bound_source_mutation_rejected(tmp_path):
    value=reference_case(tmp_path);(tmp_path/'source.json').write_text('{}')
    with pytest.raises(ValueError):validate_training_reference(value)


def test_protection_gates_do_not_promote_overall_only_improvement():
    old={'metrics':{'per':.5,'groups':{'aspirated':{'reference_count':20,'per':.2}}}}
    new={'metrics':{'per':.4,'groups':{'aspirated':{'reference_count':20,'per':.24}}}}
    baseline={'metrics':{'per':.7}}
    assert not _gate(new,baseline,old)['passed']
    assert not _rendered_gate(new,old)['passed']
    new['metrics']['groups']['aspirated']['per']=.2
    assert _gate(new,baseline,old)['passed']


def test_legacy_reference_case_uses_explicit_worker_lowercase_identity():
    index,rows=cache_case();rows[0]['example_id']='A'
    validate_original_index(index,rows,'v')
    rows.append({**rows[0],'example_id':'a'})
    with pytest.raises(ValueError,match='coverage'):validate_original_index(index,rows,'v')
