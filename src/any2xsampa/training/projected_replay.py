"""Projection of the actual AdamW displacement, not an unpreconditioned gradient."""
from __future__ import annotations
import math
from typing import Any


def projection_coefficient(dot: float, norm_squared: float) -> float:
    if not math.isfinite(dot) or not math.isfinite(norm_squared) or norm_squared < 0:
        raise ValueError('invalid projection geometry')
    if norm_squared == 0:
        if dot != 0:
            raise ValueError('nonzero dot with zero reference gradient')
        return 0.
    return min(dot, 0.) / norm_squared


def project_optimizer_step(torch: Any, parameters: list[Any], before: list[Any], reference: list[Any]) -> dict:
    """Keep the first-order replay-loss change nonpositive for THIS batch.

    This does not guarantee finite-step loss, PER, or unseen-data retention.
    Adam moments stay unchanged; decoupled decay is included in displacement.
    """
    with torch.no_grad():
        descent = [old-p for old,p in zip(before,parameters)]
        dot = sum((d.double()*g.double()).sum() for d,g in zip(descent,reference))
        norm = sum(g.double().square().sum() for g in reference)
        factor = projection_coefficient(float(dot),float(norm))
        if factor:
            for p,old,d,g in zip(parameters,before,descent,reference):
                p.copy_(old-(d-factor*g))
        actual_dot = sum(((old-p).double()*g.double()).sum() for old,p,g in zip(before,parameters,reference))
        # Rounding in the final FP32 parameter write can leave tiny residuals.
        allowance = 1e-6 * max(1., abs(float(dot)))
        if not math.isfinite(float(actual_dot)) or float(actual_dot) < -allowance:
            raise RuntimeError('projected actual update violates replay halfspace')
        return {'projected':bool(factor),'dot_before':float(dot),'dot_after':float(actual_dot),
                'reference_gradient_norm':float(norm.sqrt()),'rounding_tolerance':allowance}
