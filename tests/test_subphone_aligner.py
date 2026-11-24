import numpy as np
import pytest

from any2xsampa.pipeline.subphone_aligner import measure_subphone_boundaries


@pytest.mark.parametrize("gain", [0.03, 0.6])
def test_aspiration_and_checked_coda_are_detected_at_low_gain(gain):
    rate = 16_000
    times = np.arange(rate // 4) / rate
    signal = np.random.default_rng(17).normal(0, .004, len(times))
    release = (times >= .045) & (times < .080)
    signal[release] += .3 * np.random.default_rng(20).uniform(-1, 1, release.sum())
    voiced = (times >= .080) & (times < .180)
    signal[voiced] += np.sin(2 * np.pi * 210 * times[voiced])
    bounds = measure_subphone_boundaries(signal * gain, rate, .02, .22, onset=True, coda=True)
    assert bounds.release_ms == pytest.approx(45, abs=12)
    assert bounds.voicing_onset_ms == pytest.approx(80, abs=12)
    assert bounds.coda_closure_ms == pytest.approx(180, abs=12)
