from __future__ import annotations

import math

import pytest

from any2xsampa.core.io import dump_json
from any2xsampa.core.models import PhoneCandidate, PhoneSegment, TimeRange


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_time_range_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        TimeRange(0.0, value)


@pytest.mark.parametrize("value", [math.nan, math.inf, -0.1, 1.1, True])
def test_phone_candidate_rejects_invalid_confidence(value: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        PhoneCandidate("a", value)


def test_phone_segment_rejects_invalid_range_and_confidence() -> None:
    with pytest.raises(ValueError, match="start_sec"):
        TimeRange(0.2, 0.1)
    with pytest.raises(ValueError, match="confidence"):
        PhoneSegment(TimeRange(0.0, 0.1), [PhoneCandidate("a", 1.0)], math.nan)


def test_json_dump_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="JSON compliant"):
        dump_json({"value": math.nan})
