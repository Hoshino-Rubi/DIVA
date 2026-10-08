import copy
import importlib.util
from pathlib import Path
import numpy as np
from any2xsampa.adapters.teacher_guard import guard_logits
from any2xsampa.training.limited_data_artifacts import sha256_object


def builder():
    spec=importlib.util.spec_from_file_location('portable_builder',Path(__file__).parents[1]/'tools/build_portable_release.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_report_paths_are_removed_but_acoustic_policy_is_identical():
    vocabulary={'<blank>':0,'pʰ':1,**{f'p{i}':i for i in range(2,428)}}
    original=dict(schema_version='teacher_aspiration_guard_policy_v1',radius_frames=2,
        guard_symbols=['pʰ'],acoustic_only=True,reference_conditioning=False,
        report_bindings=[dict(path='C:/private/research/report.json',sha256='1'*64)])
    original['policy_sha256']=sha256_object(original)
    before=copy.deepcopy(original)
    result=builder().public_guard_policy(original,'2'*64)
    assert 'report_bindings' not in result
    assert result['report_binding_sha256s']==['1'*64]
    assert result['origin_policy_file_sha256']=='2'*64
    rng=np.random.default_rng(17)
    student=rng.standard_normal((12,428)).astype(np.float32)
    teacher=rng.standard_normal((12,428)).astype(np.float32);teacher[5,1]=20
    a,m=guard_logits(student,teacher,vocabulary,original)
    b,n=guard_logits(student,teacher,vocabulary,result)
    assert np.array_equal(a,b) and np.array_equal(m,n)
    assert original==before


def test_runtime_install_provenance_is_not_distributed():
    kept=builder().kept
    assert not kept(Path('numpy-1.dist-info/direct_url.json'))
    assert kept(Path('numpy-1.dist-info/METADATA'))
    assert kept(Path('numpy-1.dist-info/licenses/LICENSE.txt'))
