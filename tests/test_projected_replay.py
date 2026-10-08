import numpy as np
import pytest
from any2xsampa.training.projected_replay import projection_coefficient


@pytest.mark.parametrize('seed', range(10))
def test_projection_orthogonalizes_only_conflicting_actual_updates(seed):
    rng=np.random.default_rng(seed);d=rng.normal(size=17);g=rng.normal(size=17)
    dot=float(d@g);norm=float(g@g);factor=projection_coefficient(dot,norm)
    projected=d-factor*g
    assert projected@g >= -1e-12
    if dot>=0: assert np.array_equal(d,projected)
    else: assert abs(projected@g)<1e-12


def test_zero_gradient_does_not_invent_a_constraint():
    assert projection_coefficient(0,0)==0
    with pytest.raises(ValueError):projection_coefficient(-1,0)
    with pytest.raises(ValueError):projection_coefficient(float('nan'),1)
