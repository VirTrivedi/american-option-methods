"""High-accuracy American option reference price from CRR trees."""
from __future__ import annotations
from dataclasses import dataclass
from .params import OptionParams
from .tree import bermudan_steps, tree_price


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

def bermudan_reference(params: OptionParams, n_ex: int, n_base: int = 2000) -> ReferencePrice:
    """Bermudan price (n_ex equally spaced dates) via averaged-and-extrapolated CRR trees."""
    m = max(1, round(n_base / n_ex))

    def averaged(mult: int) -> tuple[float, float]:
        sizes = (mult * n_ex, (mult + 1) * n_ex)
        prices = [tree_price(params, n, exercise_steps=bermudan_steps(n, n_ex)) for n in sizes]
        return 0.5 * (prices[0] + prices[1]), 0.5 * (sizes[0] + sizes[1])

    (a1, n1), (a2, n2), (a4, n4) = averaged(m), averaged(2 * m), averaged(4 * m)
    r1 = (n2 * a2 - n1 * a1) / (n2 - n1)
    r2 = (n4 * a4 - n2 * a2) / (n4 - n2)
    
    return ReferencePrice(price=r2, error_estimate=abs(r2 - r1), n_base=m * n_ex)