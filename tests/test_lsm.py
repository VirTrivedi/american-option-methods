import numpy as np
import pytest
from aopt.bs import bs_price
from aopt.lsm import basis_matrix, lsm_both, lsm_boundary, lsm_price, max_train_paths
from aopt.params import OptionParams
from aopt.reference import bermudan_reference


Z_MAX = 4.0
N_PATHS = 100_000


def make(**kw):
    base = dict(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.2, option_type="put")
    return OptionParams(**{**base, **kw})


def z_score(name: str, est, ref: float) -> float:
    z = (est.price - ref) / est.stderr
    print(f"[z] {name} ({est.mode}): estimate {est.price:.5f} +/- {est.stderr:.5f}, reference {ref:.5f}, z = {z:+.2f}")
    return z


@pytest.fixture(scope="module")
def put_fit():
    p = make()
    return p, lsm_both(p, N_PATHS, n_ex=50, degree=3, basis="laguerre", seed=11, n_eval=200_000, return_policy=True)


def check_european_equivalent(name: str, p: OptionParams, seed: int, max_shortfall: float) -> None:
    """When early exercise has no value the true policy never exercises, so any fitted rule can only lose value."""
    ref = float(bs_price(p.S, p.K, p.T, p.r, p.q, p.sigma, p.option_type))
    in_sample, out_of_sample = lsm_both(p, N_PATHS, n_ex=50, seed=seed, n_eval=200_000)
    assert abs(z_score(name, in_sample, ref)) < Z_MAX
    z_score(name, out_of_sample, ref)
    shortfall = ref - out_of_sample.price
    print(f"[shortfall] {name}: European - out-of-sample = {shortfall:+.4f} (bound {max_shortfall})")
    assert out_of_sample.price <= ref + Z_MAX * out_of_sample.stderr
    assert shortfall < max_shortfall


def test_call_without_dividends_matches_black_scholes():
    # Measured shortfall 0.048 (0.46% of the price); bound at twice that.
    check_european_equivalent("call q=0 vs BS", make(option_type="call"), seed=12, max_shortfall=0.10)


def test_zero_rate_put_matches_european():
    # Measured shortfall 0.035 (0.44%), z = -3.13 two-sided: a real, predictable-sign policy bias; bound at twice that.
    check_european_equivalent("put r=0 vs BS", make(r=0.0), seed=13, max_shortfall=0.07)


def test_agrees_with_bermudan_tree_benchmark(put_fit):
    p, (in_sample, out_of_sample) = put_fit
    ref = bermudan_reference(p, 50, 2000).price
    # Allow benchmark error (~6e-5) to be negligible against the MC standard error (~0.01).
    assert abs(z_score("put n_ex=50 vs Bermudan tree", in_sample, ref)) < Z_MAX
    assert abs(z_score("put n_ex=50 vs Bermudan tree", out_of_sample, ref)) < Z_MAX


def test_out_of_sample_is_a_lower_estimate(put_fit):
    p, (_, out_of_sample) = put_fit
    ref = bermudan_reference(p, 50, 2000).price
    z = z_score("oos upper-bound check", out_of_sample, ref)
    assert out_of_sample.price <= ref + Z_MAX * out_of_sample.stderr
    assert z < Z_MAX


def test_deep_in_the_money_put_exercises_on_first_date():
    p = make(S=50.0)
    in_sample, out_of_sample = lsm_both(p, 20_000, n_ex=20, seed=14, n_eval=20_000, return_policy=True)
    assert np.mean(in_sample.exercise_index == 1) > 0.99
    dt = p.T / 20
    expected = 100.0 * np.exp(-p.r * dt) - 50.0  # discounted K - E[S_1 e^{-r dt}]
    z = z_score("deep ITM put", out_of_sample, expected)
    assert abs(z) < Z_MAX


def test_same_seed_is_identical_and_different_seed_differs():
    p = make()
    a = lsm_both(p, 20_000, n_ex=10, seed=5, n_eval=20_000)
    b = lsm_both(p, 20_000, n_ex=10, seed=5, n_eval=20_000)
    c = lsm_both(p, 20_000, n_ex=10, seed=6, n_eval=20_000)
    for x, y in zip(a, b):
        assert x.price == y.price and x.stderr == y.stderr
        np.testing.assert_array_equal(x.coefficients, y.coefficients)
    assert a[0].price != c[0].price


def test_antithetic_reduces_standard_error():
    p = make()
    anti = lsm_both(p, 50_000, n_ex=20, seed=21, antithetic=True, n_eval=100_000)
    plain = lsm_both(p, 50_000, n_ex=20, seed=21, antithetic=False, n_eval=100_000)
    for a, b in zip(anti, plain):
        print(f"[se] {a.mode}: antithetic {a.stderr:.5f} vs plain {b.stderr:.5f} (ratio {a.stderr / b.stderr:.2f})")
        assert a.stderr < b.stderr


def test_boundary_below_strike_and_roughly_increasing(put_fit):
    p, (result, _) = put_fit
    t, s = lsm_boundary(p, result)
    ok = ~np.isnan(s)
    assert ok.mean() > 0.9  # at most a few early dates lack data below the strike
    assert np.all(s[ok] <= p.K + 1e-9)
    corr = np.corrcoef(t[ok], s[ok])[0, 1]
    late_minus_early = s[ok][-10:].mean() - s[ok][:10].mean()
    print(f"[boundary] finite {ok.sum()}/{ok.size}, corr(t, S*) = {corr:.3f}, late-early = {late_minus_early:.2f}")
    # Stated tolerance: the unsmoothed regression boundary is noisy, so require a clear upward trend only.
    assert corr > 0.8 and late_minus_early > 3.0


def test_few_in_the_money_paths_gives_finite_output():
    p = make(S=300.0, sigma=0.1, T=0.25)  # a put this far OTM has almost no ITM paths
    for est in lsm_both(p, 5_000, n_ex=10, degree=6, seed=7, n_eval=5_000):
        assert np.isfinite(est.price) and np.isfinite(est.stderr) and est.price >= 0.0
    assert np.isnan(est.coefficients).any()  # skipped dates are marked NaN, never fed into the policy


def test_call_boundary_runs_and_is_above_strike():
    p = make(q=0.08, r=0.03, option_type="call")
    result = lsm_price(p, 50_000, n_ex=20, seed=8, mode="out_of_sample", n_eval=20_000)
    _, s = lsm_boundary(p, result)
    assert np.all(s[~np.isnan(s)] >= p.K - 1e-9)


def test_memory_cap_and_argument_validation():
    p = make()
    cap = max_train_paths(50, 3)
    with pytest.raises(ValueError, match="GB"):
        lsm_both(p, cap + 1000, n_ex=50)
    with pytest.raises(ValueError, match="even"):
        lsm_both(p, 1001, n_ex=10, antithetic=True)
    with pytest.raises(ValueError, match="mode"):
        lsm_price(p, 1000, n_ex=5, mode="nope")


def test_basis_shapes_and_conditioning():
    s = np.linspace(50, 100, 200)
    cond = {}
    for family in ("monomial", "laguerre", "chebyshev"):
        X = basis_matrix(s, 100.0, family, 5, "put")
        assert X.shape == (200, 6) and np.all(np.isfinite(X))
        cond[family] = np.linalg.cond(X)
    print(f"[cond] degree 5 on S in [50, 100]: {cond}")
    # Measured: chebyshev 3.7e3 < monomial 3.3e5 < laguerre 4.7e7 (weighted Laguerre columns are nearly collinear on
    # x in [0.5, 1]). This is why the fit uses lstsq (SVD) on S/K-normalized inputs, never the normal equations,
    # which would square these numbers.
    assert cond["chebyshev"] < cond["monomial"] < cond["laguerre"]
    assert cond["laguerre"] < 1e9


def test_monomial_and_chebyshev_are_the_same_estimator():
    # Same polynomial span, so the least-squares fit and exercise rule coincide up to rounding.
    p = make()
    a = lsm_price(p, 20_000, n_ex=20, degree=4, basis="monomial", seed=2, mode="out_of_sample", n_eval=20_000)
    b = lsm_price(p, 20_000, n_ex=20, degree=4, basis="chebyshev", seed=2, mode="out_of_sample", n_eval=20_000)
    print(f"[equiv] monomial {a.price:.8f} vs chebyshev {b.price:.8f}")
    assert a.price == pytest.approx(b.price, abs=1e-6)


def test_european_value_regressor():
    p = make()
    fit, oos = lsm_both(p, 50_000, n_ex=50, degree=1, basis="monomial", seed=4, n_eval=100_000, european_basis=True)
    plain, _ = lsm_both(p, 50_000, n_ex=50, degree=1, basis="monomial", seed=4, n_eval=1000)
    assert fit.coefficients.shape == (49, 3) and plain.coefficients.shape == (49, 2)
    ref = bermudan_reference(p, 50, 2000).price
    print(f"[euro-basis] degree 1 + European: oos {oos.price:.4f}, benchmark {ref:.4f}; in-sample plain degree 1: {plain.price:.4f}")
    assert oos.price <= ref + Z_MAX * oos.stderr  # still a feasible rule
    assert oos.price > ref - 0.05  # measured -0.029; degree 1 alone is ~0.10 low (basis sweep)
    _, s = lsm_boundary(p, fit)  # boundary extraction must handle the date-dependent column
    assert np.all(s[~np.isnan(s)] <= p.K + 1e-9)


def test_common_evaluation_set_is_shared_across_fits():
    p = make()
    a = lsm_both(p, 10_000, n_ex=10, seed=1, n_eval=20_000, eval_seed=5)[1]
    b = lsm_both(p, 10_000, n_ex=10, seed=2, n_eval=20_000, eval_seed=5)[1]
    c = lsm_both(p, 10_000, n_ex=10, seed=1, n_eval=20_000)[1]  # per-fit evaluation set
    assert a.price != b.price  # different policies on the same paths
    assert a.price != c.price  # same policy on different paths