"""High-accuracy American option reference price from CRR trees."""
from __future__ import annotations
from dataclasses import dataclass
import math
from .params import OptionParams
from .pde import solve_pde
from .tree import alignment_unit, bermudan_steps, tree_price


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

def aligned_reference(params: OptionParams, unit: int, n_base: int, n_ex: int | None = None) -> ReferencePrice:
    """Averaged-and-extrapolated CRR price using only step counts that are multiples of ``unit``."""
    m = max(1, round(n_base / unit))

    def averaged(mult: int) -> tuple[float, float]:
        sizes = (mult * unit, (mult + 1) * unit)
        steps = (lambda n: None) if n_ex is None else (lambda n: bermudan_steps(n, n_ex))
        prices = [tree_price(params, n, exercise_steps=steps(n)) for n in sizes]

        return 0.5 * (prices[0] + prices[1]), 0.5 * (sizes[0] + sizes[1])

    (a1, n1), (a2, n2), (a4, n4) = averaged(m), averaged(2 * m), averaged(4 * m)
    r1 = (n2 * a2 - n1 * a1) / (n2 - n1)
    r2 = (n4 * a4 - n2 * a2) / (n4 - n2)

    return ReferencePrice(price=r2, error_estimate=abs(r2 - r1), n_base=m * unit)


def bermudan_reference(params: OptionParams, n_ex: int, n_base: int = 2000) -> ReferencePrice:
    """Bermudan price (n_ex equally spaced dates) via averaged-and-extrapolated CRR trees."""
    return aligned_reference(params, math.lcm(n_ex, alignment_unit(params)), n_base, n_ex)


def dividend_tree_reference(params: OptionParams, n_base: int = 2000) -> ReferencePrice:
    """American price with discrete dividends from aligned trees (N, N+unit, ... so ex-dates are always on steps)."""
    return aligned_reference(params, alignment_unit(params), n_base)


@dataclass(frozen=True)
class DividendReference:
    """American reference with discrete dividends: extrapolated PDE, cross-checked against aligned trees."""
    price: float  # extrapolated PDE
    tree_price: float  # extrapolated aligned tree
    uncertainty: float  # |PDE - tree|
    pde_order: float  # observed order from M = N = m/4, m/2, m (clipped to [1, 2])


def dividend_reference(params: OptionParams, m: int = 3200, n_base: int = 4000) -> DividendReference:
    """PDE at M=N=m/4, m/2, m with Richardson at the observed order, and the aligned-tree reference."""
    v1, v2, v3 = (solve_pde(params, k, k, american_method="penalty").price for k in (m // 4, m // 2, m))
    ratio = (v1 - v2) / (v2 - v3) if v2 != v3 else 4.0
    order = min(2.0, max(1.0, math.log2(ratio))) if ratio > 0 else 1.0
    pde = v3 + (v3 - v2) / (2.0**order - 1.0)
    tree = dividend_tree_reference(params, n_base).price

    return DividendReference(price=pde, tree_price=tree, uncertainty=abs(pde - tree), pde_order=order)