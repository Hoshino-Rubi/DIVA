from __future__ import annotations
import numpy as np
import pytest
from any2xsampa.adapters.ctc_factor_exchange import exchange_factors, log_probabilities, reconstruction_check, diagnostics
from any2xsampa.adapters.phoneticxeus import ctc_collapse


def test_self_reconstruction_and_normalization():
    x=np.random.default_rng(17).normal(size=(40,428)).astype('float32')*4
    y=exchange_factors(x,x)
    assert reconstruction_check(x,y.astype('float32'))['collapsed_sequence_equal']
    np.testing.assert_allclose(np.exp(y).sum(-1),1,atol=2e-14)
    np.testing.assert_allclose(y,log_probabilities(x),atol=2e-14)


def test_factor_roles_and_blank_probability_not_blank_decision():
    x=np.full((2,428),-100.);v=x.copy()
    x[:,0]=np.log(.4);x[:,1:3]=np.log(.3)
    v[:,0]=np.log(.001);v[:,1]=np.log(.998);v[:,2]=np.log(.001)
    y=exchange_factors(x,v)
    np.testing.assert_allclose(np.exp(y[:,0]),np.exp(log_probabilities(x)[:,0]))
    assert (x.argmax(-1)==0).all() and (y.argmax(-1)==1).all()
    assert diagnostics(x,v,y)['blank_argmax_changed_frames']==2


def test_extreme_blank_preserves_finite_conditional_distribution():
    x=np.full((3,428),-10000.);x[:,0]=10000.
    v=x.copy();v[:,7]=-9999.
    y=exchange_factors(x,v)
    assert np.isfinite(y).all()
    assert np.all(y.argmax(-1)==0)
    assert ctc_collapse(y.argmax(-1).tolist())==[]


def test_repeated_labels_are_not_removed_across_blank():
    x=np.full((5,428),-100.)
    for t,k in enumerate([1,1,0,1,1]):x[t,k]=100.
    assert ctc_collapse(exchange_factors(x,x).argmax(-1).tolist())==[1,1]


@pytest.mark.parametrize('bad',[np.zeros((0,428)),np.zeros((3,427)),np.full((3,428),np.nan),np.full((3,428),np.inf)])
def test_reject_invalid_arrays(bad):
    with pytest.raises(ValueError):exchange_factors(bad,bad)


def test_reject_shifted_geometry():
    with pytest.raises(ValueError):exchange_factors(np.zeros((4,428)),np.zeros((5,428)))


def test_constant_frame_offsets_do_not_change_factors():
    rng=np.random.default_rng(4);x=rng.normal(size=(10,428));v=rng.normal(size=x.shape)
    np.testing.assert_allclose(exchange_factors(x,v),exchange_factors(x+100,v-300),atol=1e-12)


def test_no_side_effects_and_all_classes_remain():
    rng=np.random.default_rng(6);x=rng.normal(size=(4,428));v=rng.normal(size=x.shape)
    a=x.copy();b=v.copy();y=exchange_factors(x,v)
    assert y.shape==(4,428)
    np.testing.assert_array_equal(x,a);np.testing.assert_array_equal(v,b)


def test_normalized_factors_are_not_average_posteriors():
    rng=np.random.default_rng(8);x=rng.normal(size=(3,428));v=rng.normal(size=x.shape)
    y=np.exp(exchange_factors(x,v));px=np.exp(log_probabilities(x));pv=np.exp(log_probabilities(v))
    assert not np.allclose(y,.5*(px+pv))
    np.testing.assert_allclose(y[:,1:]/y[:,1:].sum(-1,keepdims=True),pv[:,1:]/pv[:,1:].sum(-1,keepdims=True))
