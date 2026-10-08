"""Opt-in real/fixture replay for approved whole-sequence continuation checkpoints."""
from pathlib import Path
from any2xsampa.adapters.phoneticxeus import PhoneticXeusPosteriorRecognizer
from any2xsampa.adapters.sequence_route_inference import install_checkpoint,save_posterior
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,require_digest
from any2xsampa.training.singing_domain import architecture_identity
from any2xsampa.core.interfaces import UniversalPhoneRecognizer
from any2xsampa.core.models import PhoneSegment


def verify_job(run:Path,job:Path)->dict:
    result=read_json(job/'RESULT.json')
    if (result.get('complete') is not True or result.get('fixed_epoch')!=20
            or result.get('arm')!='whole_sequence' or not result.get('frozen_parameters_unchanged')):
        raise ValueError('incomplete/unrecognized whole-sequence checkpoint')
    for name,key in [('POLICY.json','policy_sha256'),('CODE_FREEZE.json','code_freeze_sha256'),
                     ('WHOLE_REFERENCES.json','whole_reference_sha256'),('WHOLE_CACHE.json','whole_cache_sha256')]:
        verify_file(run/name,result[key])
    for path,digest in result['code_bindings'].items():verify_file(Path(path),digest)
    verify_file(Path(result['checkpoint_path']),result['checkpoint_sha256'])
    verify_file(Path(result['parent_checkpoint_path']),result['parent_checkpoint_sha256'])
    held=result['held_song']
    if held is not None and (held in result['training_songs'] or held in result['whole_training_songs']):
        raise ValueError('whole-sequence checkpoint leaked its held song')
    return result


class WholeSequencePosteriorRecognizer(UniversalPhoneRecognizer):
    def __init__(self,path:Path,*,fixture:bool=False)->None:
        self.path,self.fixture=path,fixture

    def recognize(self,vocal_audio_path:Path,lyrics:str|None=None)->list[PhoneSegment]:
        if lyrics is not None:raise ValueError('whole-sequence recognition accepts audio only')
        payload=read_json(self.path);metadata=payload.get('producer',{}).get('model_metadata',{})
        identity=metadata.get('singing_domain_architecture',{})
        if architecture_identity(identity['snapshot'],identity['architecture'])['sha256']!=identity['sha256']:
            raise ValueError('invalid whole-sequence architecture identity')
        checkpoint=metadata.get('sequence_route_checkpoint',{})
        require_digest(checkpoint.get('sha256'),'checkpoint_sha256');require_digest(checkpoint.get('parent_sha256'),'parent_sha256')
        if checkpoint.get('arm')!='whole_sequence' or checkpoint.get('seed') not in {17,29,43}:
            raise ValueError('unsupported whole-sequence recipe/seed')
        segments=PhoneticXeusPosteriorRecognizer(self.path,fixture=self.fixture).recognize(vocal_audio_path)
        for segment in segments:segment.metadata['sequence_route_checkpoint']=checkpoint
        return segments
