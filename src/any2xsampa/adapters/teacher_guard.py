"""Opt-in acoustic teacher/student logit routing, preserving all 428 classes."""
from __future__ import annotations
from typing import Any, Mapping
from any2xsampa.training.limited_data_artifacts import sha256_object


def guard_logits(student: Any, teacher: Any, vocabulary: Mapping[str,int], policy: Mapping[str,Any]) -> tuple[Any, Any]:
    import numpy as np
    if policy.get('schema_version')!='teacher_aspiration_guard_policy_v1' or policy.get('policy_sha256')!=sha256_object({k:v for k,v in policy.items() if k!='policy_sha256'}):
        raise ValueError('guard policy hash or schema mismatch')
    s,t=np.asarray(student),np.asarray(teacher)
    if s.shape!=t.shape or s.ndim!=2 or s.shape[1]!=428 or s.shape[0]<1 or s.dtype!=np.float32 or t.dtype!=np.float32:
        raise ValueError('guard needs aligned FP32 frames and the full 428-label inventory')
    if len(vocabulary)!=428 or sorted(vocabulary.values())!=list(range(428)) or vocabulary.get('<blank>')!=0:
        raise ValueError('guard vocabulary mismatch')
    if not np.isfinite(s).all() or not np.isfinite(t).all():raise ValueError('non-finite guard logits')
    if policy['radius_frames']!=2 or policy.get('acoustic_only') is not True or policy.get('reference_conditioning') is not False:
        raise ValueError('unfrozen or reference-conditioned guard')
    symbols=sorted(p for p in vocabulary if '\u02b0' in p)
    if policy['guard_symbols']!=symbols:raise ValueError('guard classes changed')
    ids=[vocabulary[p] for p in symbols]
    anchors=np.isin(t.argmax(-1),ids);mask=anchors.copy()
    for radius in (1,2):
        mask[radius:] |= anchors[:-radius]
        mask[:-radius] |= anchors[radius:]
    result=s.copy();result[mask]=t[mask]
    return result,mask
