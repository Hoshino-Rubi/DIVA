from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from verify_innovation_stages import compensate_shift


def test_known_spike_translation_not_called_instability():
    w=np.zeros((10,2));w[3,0]=1;w[6,1]=1
    later=np.vstack((w[:1],w[:-1]));earlier=np.vstack((w[1:],w[-1:]))
    a,m=compensate_shift(later,1);b,n=compensate_shift(earlier,-1)
    np.testing.assert_array_equal(a,w[1:-1]);np.testing.assert_array_equal(b,w[1:-1])
    assert np.all(m==1) and np.all(n==1)


@pytest.mark.parametrize('shift',[0,2,-2])
def test_only_known_one_frame_diagnostic_shift(shift):
    with pytest.raises(ValueError):compensate_shift(np.ones((4,2)),shift)
