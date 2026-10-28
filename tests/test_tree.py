import itertools
import numpy as np
import pytest
from aopt.bs import bs_price
from aopt.params import OptionParams
from aopt.tree import exercise_boundary, tree_price


def make(**kw):
    base = dict(S=100.0, K=100.0, T=1.0, r=0.05, q=0.02, sigma=0.2, option_type="put")
    return OptionParams(**{**base, **kw})


def bs(p: OptionParams) -> float:
    return float(bs_price(p.S, p.K, p.T, p.r, p.q, p.sigma, p.option_type))


GRID = [make(S=S, K=100.0, T=T, r=r, q=q, sigma=sig, option_type=typ)
        for S, T, r, q, sig, typ in itertools.product(
            (70.0, 100.0, 130.0), (0.25, 1.0, 2.0), (0.0, 0.03, 0.08), (0.0, 0.04), (0.15, 0.4), ("put", "call"))]


def test_european_converges_to_bs_with_shrinking_tolerance():
    p = make()
    errs = [abs(tree_price(p, n, european=True) - bs(p)) for n in (50, 200, 800, 3200)]
    # CRR error is O(1/N); measured constant for these inputs is ~1.95 (even N), so allow 2.5/N
    for n, e in zip((50, 200, 800, 3200), errs):
        assert e < 2.5 / n
    assert errs[-1] < errs[0] / 10


@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity_european_tree(q):
    c, p = make(q=q, option_type="call"), make(q=q, option_type="put")
    n = 400
    lhs = tree_price(c, n, european=True) - tree_price(p, n, european=True)
    rhs = c.S * np.exp(-q * c.T) - c.K * np.exp(-c.r * c.T)
    assert lhs == pytest.approx(rhs, abs=1e-9)  # exact on the tree, not just in the limit


@pytest.mark.parametrize("p", GRID, ids=lambda p: f"{p.option_type}-S{p.S}-T{p.T}-r{p.r}-q{p.q}-s{p.sigma}")
def test_american_dominates_european_and_intrinsic(p):
    am, eu = tree_price(p, 200), tree_price(p, 200, european=True)
    assert am - eu >= -1e-12
    assert am >= float(p.payoff(p.S)) - 1e-12


def test_zero_rate_american_put_equals_european():
    p = make(r=0.0, q=0.0)
    assert tree_price(p, 500) == pytest.approx(tree_price(p, 500, european=True), abs=1e-10)


def test_no_dividend_american_call_equals_european():
    p = make(q=0.0, option_type="call")
    assert tree_price(p, 500) == pytest.approx(tree_price(p, 500, european=True), abs=1e-10)


def test_dividend_american_call_has_premium():
    p = make(q=0.08, r=0.03, option_type="call", S=120.0)
    assert tree_price(p, 500) > tree_price(p, 500, european=True) + 1e-3


def test_put_monotonicity():
    n = 300
    f = lambda **kw: tree_price(make(**kw), n)
    sig = [f(sigma=s) for s in (0.1, 0.2, 0.3, 0.5)]
    K = [f(K=k) for k in (80.0, 100.0, 120.0)]
    S = [f(S=s) for s in (80.0, 100.0, 120.0)]
    assert all(np.diff(sig) > 0) and all(np.diff(K) > 0) and all(np.diff(S) < 0)


@pytest.mark.parametrize("bad", [0, -3, 1.5])
def test_invalid_n_steps(bad):
    with pytest.raises(ValueError, match="n_steps"):
        tree_price(make(), bad)


def test_invalid_probability_raises_clear_error():
    with pytest.raises(ValueError, match="risk-neutral probability"):
        tree_price(make(r=2.0, sigma=0.05), 2)


def test_known_american_put_value():
    # S=K=100, T=1, r=5%, q=0, sigma=20%: American put ~ 6.0903 (literature value)
    p = make(q=0.0)
    assert tree_price(p, 2000) == pytest.approx(6.0903, abs=2e-3)


def test_put_boundary_below_strike_and_approaches_strike():
    n = 1000
    p = make(q=0.0, T=1.0)
    t, s_star = exercise_boundary(p, n)
    valid = s_star[~np.isnan(s_star)]
    assert valid.size > 0 and np.all(valid <= p.K + 1e-9)
    # At maturity the boundary is the nearest ITM node: within two node spacings of K
    assert s_star[-1] >= p.K * np.exp(-2 * p.sigma * np.sqrt(p.T / n)) * (1 - 1e-9)
    # S*(t) rises toward K as t -> T (sampled coarsely to ignore node-spacing steps)
    sample = s_star[[100, 500, 900, 990, 998]]
    assert np.all(np.diff(sample) > 0)
    assert sample[-1] > 0.98 * p.K


def test_boundary_nan_when_no_exercise_region():
    t, s_star = exercise_boundary(make(r=0.0, q=0.0), 100)
    assert np.all(np.isnan(s_star[:-1]))  # r = 0 put is never exercised early
    t, s_star = exercise_boundary(make(q=0.0, option_type="call"), 100)
    assert np.all(np.isnan(s_star[:-1]))


def test_call_boundary_above_strike():
    p = make(q=0.08, r=0.03, option_type="call")
    _, s_star = exercise_boundary(p, 500)
    valid = s_star[:-1][~np.isnan(s_star[:-1])]
    assert valid.size > 0 and np.all(valid >= p.K - 1e-9)


def test_perpetual_put_boundary():
    # q = 0: S* = K * b / (b - 1) with b = -2r/sigma^2 (the negative root of the ODE), i.e.
    # S* = K * g / (g + 1) for g = 2r/sigma^2. Far from maturity the finite-T boundary is flat
    r, sigma, K, T, n = 0.08, 0.4, 100.0, 30.0, 3000
    g = 2 * r / sigma**2
    target = K * g / (g + 1)  # = 50
    _, s_star = exercise_boundary(make(r=r, q=0.0, sigma=sigma, T=T, K=K), n)
    spacing = 2 * sigma * np.sqrt(T / n)  # relative node spacing
    for frac in (0.05, 0.1, 0.2):
        assert s_star[int(frac * n)] == pytest.approx(target, rel=spacing)