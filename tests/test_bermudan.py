import pytest
from aopt.bs import bs_price
from aopt.params import OptionParams
from aopt.reference import bermudan_reference, reference_price
from aopt.tree import bermudan_steps, tree_price


def make(**kw):
    base = dict(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.2, option_type="put")
    return OptionParams(**{**base, **kw})


def test_bermudan_steps_mapping_and_validation():
    assert bermudan_steps(100, 4) == frozenset({25, 50, 75, 100})
    with pytest.raises(ValueError, match="multiple"):
        bermudan_steps(100, 3)
    with pytest.raises(ValueError):
        tree_price(make(), 100, exercise_steps={101})
    with pytest.raises(ValueError, match="either"):
        tree_price(make(), 100, european=True, exercise_steps={5})


def test_all_steps_equals_american():
    p = make()
    assert tree_price(p, 300, exercise_steps=range(0, 301)) == pytest.approx(tree_price(p, 300), abs=1e-12)


def test_no_exercise_dates_equals_european():
    for opt in ("put", "call"):
        p = make(q=0.03, option_type=opt)
        assert tree_price(p, 300, exercise_steps=set()) == pytest.approx(tree_price(p, 300, european=True), abs=1e-12)


def test_price_nondecreasing_in_nested_date_sets():
    p = make()
    n = 800  # divisible by 2, 4, ..., 32; each date set contains the previous one
    prices = [tree_price(p, n, exercise_steps=bermudan_steps(n, k)) for k in (1, 2, 4, 8, 16, 32)]
    assert all(b >= a - 1e-12 for a, b in zip(prices, prices[1:]))
    assert prices[0] == pytest.approx(tree_price(p, n, european=True), abs=1e-12)  # only date T
    assert prices[-1] <= tree_price(p, n) + 1e-12


def test_zero_rate_bermudan_put_equals_european_put():
    p = make(r=0.0)
    assert tree_price(p, 400, exercise_steps=bermudan_steps(400, 20)) == pytest.approx(tree_price(p, 400, european=True), abs=1e-10)


def test_bermudan_reference_consistency():
    p = make()
    a, b = bermudan_reference(p, 10, 1000), bermudan_reference(p, 10, 3000)
    # Measured: values differ by ~1.5e-4 across bases, with error estimates 1.4e-4 to 5e-4.
    assert abs(a.price - b.price) < 5e-4
    assert 0.0 <= b.error_estimate < 1e-3
    assert b.price < reference_price(p, 1000).price  # discretization of exercise dates lowers the price


def test_bermudan_with_one_date_matches_black_scholes_reference():
    p = make()
    one = bermudan_reference(p, 1, 2000)
    assert one.price == pytest.approx(float(bs_price(100, 100, 1, 0.05, 0.0, 0.2, "put")), abs=2e-4)


def test_bermudan_boundary_shift_matches_continuity_correction():
    """Discrete exercise dates shift the put boundary up by about exp(beta*sigma*sqrt(dt)) (Broadie-Glasserman-Kou form).

    Measured against the PDE American boundary (n_ex = 50): ratio 1.0147 against 1.0166 predicted with beta = 0.5826, i.e.
    an implied beta of about 0.5 (0.47-0.52 for n_ex = 10..100). The form is confirmed; the constant is not exactly 0.5826.
    """
    import numpy as np

    from aopt.pde import solve_pde
    from aopt.tree import exercise_boundary

    p, n_ex, n = make(), 50, 6000
    pde = solve_pde(p, 800, 800, return_boundary=True)
    t, s = exercise_boundary(p, n, bermudan_steps(n, n_ex))
    steps = np.array(sorted(bermudan_steps(n, n_ex)))[:-1]
    t, s = t[steps], s[steps]
    keep = ~np.isnan(s) & (t > 0.2) & (t < 0.8)
    ratio = np.median(s[keep] / np.interp(t[keep], pde.boundary_t, pde.boundary_S))
    predicted = np.exp(0.5826 * p.sigma * np.sqrt(p.T / n_ex))
    print(f"[bgk] median Bermudan/American boundary ratio {ratio:.4f}, predicted {predicted:.4f}, share of shift {(ratio - 1) / (predicted - 1):.2f}")
    assert ratio > 1.0
    assert 0.7 < (ratio - 1.0) / (predicted - 1.0) < 1.05