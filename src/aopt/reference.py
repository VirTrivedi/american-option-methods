"""High-accuracy American option reference price from CRR trees."""
from __future__ import annotations
from dataclasses import dataclass
from .params import OptionParams
from .tree import tree_price


@dataclass(frozen=True)
class ReferencePrice:
    """Reference price with an estimated absolute error."""
    price: float
    error_estimate: float
    n_base: int


def averaged_price(params: OptionParams, n_steps: int) -> float:
    """Mean of the N and (N+1) step tree prices."""
    return 0.5 * (tree_price(params, n_steps) + tree_price(params, n_steps + 1))


def richardson(params: OptionParams, n_steps: int) -> float:
    """First-order Richardson extrapolation of the averaged price: 2*A(2N) - A(N)."""
    return 2.0 * averaged_price(params, 2 * n_steps) - averaged_price(params, n_steps)


def reference_price(params: OptionParams, n_base: int = 1000) -> ReferencePrice:
    """American price via averaged-and-extrapolated CRR trees."""
    a1, a2, a4 = (averaged_price(params, n_base * m) for m in (1, 2, 4))
    r1, r2 = 2.0 * a2 - a1, 2.0 * a4 - a2
    return ReferencePrice(price=r2, error_estimate=abs(r2 - r1), n_base=n_base)