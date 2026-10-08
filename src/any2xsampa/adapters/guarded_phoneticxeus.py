"""Explicit guarded inference candidate; no default-model registration."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
from any2xsampa.training.limited_data_artifacts import read_json,verify_file,sha256_file,write_json_new
from any2xsampa.training.phoneticxeus_adaptation import make_residual_adapter
from any2xsampa.adapters.teacher_guard import guard_logits
from any2xsampa.adapters.phoneticxeus import emission_payload


class GuardedPhoneHead:
    def __init__(self, torch: Any, head: Any, vocabulary: Mapping[str,int], candidate_path: Path, candidate_sha256: str):
        from safetensors.torch import load_file
        verify_file(candidate_path,candidate_sha256)
        self.candidate=read_json(candidate_path)
        if self.candidate.get('schema_version')!='guarded_phone_candidate_v1' or self.candidate.get('default_replacement') is not False or self.candidate.get('test_opening') is not False:
            raise ValueError('unsupported or auto-deploying candidate')
        # Portable packages retain the frozen hashes and resolve only file locations.
        for key in ('policy', 'teacher_checkpoint', 'student_checkpoint'):
            item = dict(self.candidate[key])
            file = Path(item['path'])
            item['path'] = str((candidate_path.parent / file).resolve())
            self.candidate[key] = item
        policy=self.candidate['policy'];verify_file(Path(policy['path']),policy['sha256']);self.policy=read_json(Path(policy['path']))
        self.torch,self.head,self.vocabulary=torch,head,vocabulary
        if len(vocabulary)!=428 or any(p.requires_grad for p in head.parameters()):raise ValueError('guard requires frozen 428-label head')
        self.teacher=make_residual_adapter(torch).eval().requires_grad_(False).to('cuda')
        self.student=make_residual_adapter(torch).eval().requires_grad_(False).to('cuda')
        for name,module in [('teacher_checkpoint',self.teacher),('student_checkpoint',self.student)]:
            spec=self.candidate[name];verify_file(Path(spec['path']),spec['sha256']);module.load_state_dict(load_file(spec['path']),strict=True)
        self.candidate_sha256=candidate_sha256

    def predict(self, features: Any) -> dict:
        with self.torch.inference_mode():
            f=features.to('cuda')
            teacher=self.head(self.teacher(f)).cpu().numpy()
            student=self.head(self.student(f)).cpu().numpy()
        guarded,mask=guard_logits(student,teacher,self.vocabulary,self.policy)
        return dict(frame_logits=guarded,teacher_logits=teacher,student_logits=student,guard_mask=mask)

    def save(self, features: Any, source_payload: Mapping[str,Any], output: Path) -> dict:
        from safetensors.numpy import save_file
        import numpy as np
        if source_payload.get('split')=='test' or source_payload.get('test_outputs_sealed'):
            raise ValueError('sealed test input forbidden')
        output.mkdir(parents=True,exist_ok=False)
        result=self.predict(features);raw=output/'evidence.safetensors'
        save_file({**{k:v for k,v in result.items() if k!='guard_mask'},'guard_mask':result['guard_mask'].astype(np.uint8),'encoder_features':features.cpu().numpy()},str(raw))
        p=source_payload['producer']
        metadata={**p['model_metadata'],'guarded_candidate_sha256':self.candidate_sha256,
                  'teacher_checkpoint_sha256':self.candidate['teacher_checkpoint']['sha256'],
                  'student_checkpoint_sha256':self.candidate['student_checkpoint']['sha256'],
                  'guard_policy_sha256':self.candidate['policy']['sha256']}
        payload=emission_payload(result['frame_logits'],self.vocabulary,input_audio=p['input_audio'],crop=p['crop'],model_metadata=metadata,geometry=source_payload['decoding']['geometry'],runtime=p['runtime'])
        payload['producer']['name']='phoneticxeus-guarded-candidate'
        payload['producer']['adaptation_kind']='audio_only_teacher_guarded_residual'
        payload['raw_evidence']={'file':raw.name,'sha256':sha256_file(raw),'dtype':'float32','full_label_count':428}
        payload['guard_evidence']={'selected_frames':int(result['guard_mask'].sum()),'total_frames':len(result['guard_mask']),'candidate_sha256':self.candidate_sha256,'inference_reference_used':False}
        for k in ('example_id','song_id','split','audio_role'):
            if k in source_payload:payload[k]=source_payload[k]
        pp=output/'posterior.json';write_json_new(pp,payload)
        return {'posterior_path':str(pp.resolve()),'posterior_sha256':sha256_file(pp),'evidence_path':str(raw.resolve()),'evidence_sha256':sha256_file(raw),'guarded_frames':int(result['guard_mask'].sum()),'frames':len(result['guard_mask'])}
