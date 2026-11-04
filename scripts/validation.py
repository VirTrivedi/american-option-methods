"""Measured values of the validation checks that matter most (the same cases as the tests), written to results/tables/validation.csv."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from aopt.params import OptionParams
from aopt.pde import solve_pde
from aopt.tree import exercise_boundary, tree_price

OUT = Path(__file__).resolve().parents[1] / "results" / "tables" / "validation.csv"
K, R, T, TD = 100.0, 0.05, 1.0, 0.5
N_TREE = 4000


def make(**kw) -> OptionParams:
    return OptionParams(**{**dict(S=100.0, K=K, T=T, r=R, q=0.0, sigma=0.2, option_type="put"), **kw})


def main() -> None:
    rows = []

    def add(check: str, quantity: str, measured: float, reference: str, test: str) -> None:
        rows.append(dict(check=check, quantity=quantity, measured=measured, compared_with=reference, test=test))

    p = make(r=0.0)
    add("r = 0 put is never exercised early", "tree N=500: American - European", tree_price(p, 500) - tree_price(p, 500, european=True), "0", "test_tree.py::test_zero_rate_american_put_equals_european")
    add("r = 0 put is never exercised early", "PDE M=400, N=200: American - European", solve_pde(p, 400, 200).price - solve_pde(p, 400, 200, european=True).price, "0", "test_pde.py::test_zero_rate_american_put_equals_european")

    threshold = K * (1.0 - math.exp(-R * (T - TD)))
    p = make(dividends=((TD, 0.8 * threshold),), option_type="call")
    add("Merton: call, D below K(1 - exp(-r(T - t_d)))", f"threshold (K={K:.0f}, r={R}, T-t_d={T - TD})", threshold, "closed form", "test_dividends.py::test_merton_below_threshold_call_equals_european")
    add("Merton: call, D = 0.8 x threshold", f"tree N={N_TREE}: American - European", tree_price(p, N_TREE) - tree_price(p, N_TREE, european=True), "0 (empty exercise region)", "test_dividends.py::test_merton_below_threshold_call_equals_european")
    add("Merton: call, D = 0.8 x threshold", "PDE M=N=800: American - European", solve_pde(p, 800, 800, american_method="penalty").price - solve_pde(p, 800, 800, european=True).price, "0 (penalty tolerance)", "test_dividends.py::test_merton_below_threshold_call_equals_european")
    p = make(dividends=((TD, 3 * threshold),), option_type="call")
    add("Merton: call, D = 3 x threshold", f"tree N={N_TREE}: American - European (premium)", tree_price(p, N_TREE) - tree_price(p, N_TREE, european=True), "> 0", "test_dividends.py::test_merton_above_threshold_exercise_only_just_before_ex_date")

    d = 1.0
    p = make(dividends=((TD, d),))
    _, s_star = exercise_boundary(p, N_TREE)
    last = int(np.flatnonzero(~np.isnan(s_star[: N_TREE // 2])).max())
    add("put exercise region is empty before the ex-date", "window length t_d - t_vanish, D=1 (tree N=4000)", TD - last / N_TREE, f"D/(rK) = {d / (R * K):.4f}", "test_dividends.py::test_put_boundary_jumps_at_ex_date")
    add("put exercise region is empty before the ex-date", "boundary jump S*(after) - S*(before), D=1 (tree N=4000)", float(s_star[N_TREE // 2 + 1] - s_star[last]), "sanity band 10-25", "test_dividends.py::test_put_boundary_jumps_at_ex_date")

    r, sigma, kk, tt, n = 0.08, 0.4, 100.0, 30.0, 3000
    g = 2 * r / sigma ** 2
    target = kk * g / (g + 1)
    _, s_star = exercise_boundary(make(r=r, sigma=sigma, T=tt, K=kk, S=100.0), n)
    spacing = 2 * sigma * math.sqrt(tt / n)
    errs = {frac * tt: abs(s_star[int(frac * n)] - target) / target for frac in (0.05, 0.10, 0.20)}
    # the three times land on the same lattice (even step indices), so the boundary is the same node and the three errors coincide
    add("perpetual put boundary (T=30, r=8%, σ=40%)", f"max over t = {', '.join(f'{t:.1f}' for t in errs)} of |S*(t) - {target:.1f}| / {target:.1f}, tree N={n}", max(errs.values()),
        f"relative node spacing {spacing:.3f}", "test_tree.py::test_perpetual_put_boundary")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()