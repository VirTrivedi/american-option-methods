"""Discrete cash dividends: tree (spot-jump with interpolation), PDE (event hook + cubic spline) and LSM."""
import math
import warnings
import numpy as np
import pytest
from aopt.bs import bs_price
from aopt.lsm import fit_lsm, lsm_both, lsm_price
from aopt.params import OptionParams
from aopt.pde import solve_pde
from aopt.reference import bermudan_reference, dividend_reference
from aopt.tree import aligned_steps, escrowed_price, exercise_boundary, tree_price


TD = 0.5
K, R, T = 100.0, 0.05, 1.0
MERTON = K * (1.0 - math.exp(-R * (T - TD)))
Z_MAX = 4.0
N_TREE = 4000


def make(divs=((TD, 3.0),), **kw):
    base = dict(S=100.0, K=K, T=T, r=R, q=0.0, sigma=0.2, option_type="put")
    return OptionParams(**{**base, "dividends": divs, **kw})


def pde_price(p, m=800, **kw):
    return solve_pde(p, m, m, american_method="penalty", **kw).price


def test_validation_and_sorting():
    p = make(divs=((0.7, 1.0), (0.3, 2.0)))
    assert p.dividends == ((0.3, 2.0), (0.7, 1.0))
    for bad in (((0.0, 1.0),), ((1.0, 1.0),), ((0.5, 0.0),), ((0.5, -1.0),), ((0.5, 1.0), (0.5, 2.0))):
        with pytest.raises(ValueError):
            make(divs=bad)


def test_no_dividends_reproduces_existing_prices_exactly():
    # golden values: pareto_raw.csv written by the Step 3 code, before dividends existed (printed to 8 digits)
    p = make(divs=(), q=0.0)
    assert p == make(divs=()) and p.dividends == ()
    assert tree_price(p, 400) == pytest.approx(6.08840590, abs=5e-9)
    assert pde_price(p, 400) == pytest.approx(6.08871787, abs=5e-9)
    assert solve_pde(p, 400, 400).price == pytest.approx(6.08871788, abs=5e-9)


def test_no_dividends_boundary_regression_snapshot():
    # snapshot taken after the dividend work, of the unchanged no-dividend branch (q=0.0, N=1000); guards against drift
    _, s = exercise_boundary(make(divs=()), 1000)
    assert [s[i] for i in (100, 500, 900)] == pytest.approx([80.6513, 83.7706, 90.3759], abs=1e-3)


def test_aligned_steps_and_snapping_warning():
    p = make(divs=((0.3, 1.0), (0.5, 1.0)))
    assert aligned_steps(p, 1000) % 10 == 0 and aligned_steps(p, 1000) >= 1000
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # aligned N must not warn
        tree_price(p, aligned_steps(p, 400))
    with pytest.warns(UserWarning, match="snapping error"):
        tree_price(p, 401)
    with pytest.raises(ValueError, match="snap"):
        tree_price(make(divs=((0.001, 1.0),)), 10)


def test_snapping_misaligned_tree_vs_aligned_and_pde():
    p = make()
    aligned, ref = tree_price(p, 1000), pde_price(p, 1600)
    with pytest.warns(UserWarning):
        mis = tree_price(p, 1001)  # t_d/dt = 500.5 -> snaps to step 500 or 501, error dt/2 = 5e-4 years
    print(f"[snap] aligned N=1000 {aligned:.5f}, misaligned N=1001 {mis:.5f}, PDE(1600) {ref:.5f}")
    assert abs(aligned - ref) < 3e-3  # measured 2.1e-3: the N=1000 tree error (O(1/N) with interpolation)
    assert abs(mis - ref) < 6e-3  # measured 4.6e-3
    assert abs(mis - ref) > abs(aligned - ref)  # misalignment costs accuracy


@pytest.mark.parametrize("typ", ["put", "call"])
def test_small_dividend_limit(typ):
    base = tree_price(make(divs=(), option_type=typ), N_TREE)
    errs = [abs(tree_price(make(divs=((TD, d),), option_type=typ), N_TREE) - base) for d in (0.1, 0.01, 0.001)]
    pdes = [abs(pde_price(make(divs=((TD, d),), option_type=typ)) - pde_price(make(divs=(), option_type=typ))) for d in (0.1, 0.01, 0.001)]
    print(f"[limit {typ}] tree |dV| {errs}, pde |dV| {pdes}")
    assert errs[0] > errs[1] > errs[2] and pdes[0] > pdes[1] > pdes[2]
    assert errs[2] < 1e-3 and pdes[2] < 1e-3  # |dV/dD| <= 1 so D=1e-3 moves the price < 1e-3 (+ interpolation error)


@pytest.mark.parametrize("typ,sign", [("put", +1), ("call", -1)])
def test_monotone_in_dividend_size(typ, sign):
    ds = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    for fn in (lambda p: tree_price(p, N_TREE), pde_price):
        v = [fn(make(divs=((TD, d),) if d else (), option_type=typ)) for d in ds]
        assert np.all(sign * np.diff(v) >= -1e-9), v


def test_merton_below_threshold_call_equals_european():
    p = make(divs=((TD, 0.8 * MERTON),), option_type="call")
    tree_gap = tree_price(p, N_TREE) - tree_price(p, N_TREE, european=True)
    pde_gap = pde_price(p) - solve_pde(p, 800, 800, european=True).price
    print(f"[merton] threshold {MERTON:.4f}, D={0.8 * MERTON:.4f}: tree gap {tree_gap:.2e}, pde gap {pde_gap:.2e}")
    assert abs(tree_gap) < 1e-9  # no exercise anywhere, so the two recursions are identical
    assert abs(pde_gap) < 1e-6  # penalty tolerance O(1/rho) only


def test_merton_above_threshold_exercise_only_just_before_ex_date():
    p = make(divs=((TD, 3 * MERTON),), option_type="call")
    n = N_TREE
    premium = tree_price(p, n) - tree_price(p, n, european=True)
    _, s_star = exercise_boundary(p, n)
    steps = np.flatnonzero(~np.isnan(s_star[:-1]))
    print(f"[merton] D=3*threshold: premium {premium:.4f}, exercise steps {steps}, boundary {s_star[steps]}")
    assert premium > 0.01
    assert list(steps) == [n // 2]
    pde = solve_pde(p, 800, 800, american_method="penalty", return_boundary=True)
    ex_t = pde.boundary_t[~np.isnan(pde.boundary_S) & (pde.boundary_t < T - 1e-9)]
    print(f"[merton] PDE exercise times {ex_t}")
    assert ex_t.size >= 1 and np.all(np.abs(ex_t - TD) < 1e-9)


def test_put_not_exercised_at_cum_dividend_step_tree_and_pde():
    p = make(divs=((TD, 3.0),))
    n = N_TREE
    _, s_star = exercise_boundary(p, n)
    assert np.isnan(s_star[n // 2])
    pde = solve_pde(p, 800, 800, american_method="penalty", return_boundary=True)
    at_event = np.abs(pde.boundary_t - TD) < 1e-9
    assert at_event.sum() == 1 and np.isnan(pde.boundary_S[at_event][0])


def test_put_not_exercised_at_cum_dividend_lsm():
    p = make(divs=((TD, 3.0),))
    fit = fit_lsm(p, 200_000, 50, 3, "laguerre", seed=21, return_policy=True)
    k_d = 25
    frac_at = float(np.mean(fit.exercise_index == k_d))
    frac_neighbours = float(np.mean((fit.exercise_index == k_d - 1) | (fit.exercise_index == k_d + 1)))
    print(f"[lsm] fraction exercising at the ex-date step {frac_at:.5f} (neighbouring steps {frac_neighbours:.5f})")
    assert frac_at < 0.002  # regression noise only; neighbouring dates exercise ~100x more often


def test_put_boundary_jumps_at_ex_date():
    # Before the ex-date waiting is worth D at the cost of interest K*r*dt, so the exercise region exists only for
    # t < t_d - D/(r*K) (0.2 yr for D=1) and the boundary then vanishes; just after t_d it re-emerges ~16 higher.
    d = 1.0
    p = make(divs=((TD, d),))
    n = N_TREE
    _, s_star = exercise_boundary(p, n)
    last_before = np.flatnonzero(~np.isnan(s_star[: n // 2])).max()
    pre, post = s_star[last_before], s_star[n // 2 + 1]
    window = TD - last_before / n
    print(f"[jump] tree D=1: boundary vanishes at t={last_before / n:.4f} (window {window:.4f} vs D/(rK)={d / (R * K):.4f}), "
          f"S*={pre:.3f}; first post-ex S*={post:.3f}; jump {post - pre:+.3f}")
    assert 10.0 < post - pre < 25.0  # measured +16.9 (node spacing is ~0.3)
    assert window == pytest.approx(d / (R * K), abs=0.02)  # measured 0.199

    pde = solve_pde(p, 1600, 1600, american_method="penalty", return_boundary=True)
    t, b = pde.boundary_t, pde.boundary_S
    pre_mask = (t < TD - 1e-9) & ~np.isnan(b)
    post_mask = (t > TD + 1e-9) & (t < T - 1e-9) & ~np.isnan(b)
    pre_p, post_p = b[pre_mask][np.argmax(t[pre_mask])], b[post_mask][np.argmin(t[post_mask])]
    print(f"[jump] PDE D=1: boundary vanishes at t={t[pre_mask].max():.4f}, S*={pre_p:.3f}; first post-ex S*={post_p:.3f}; jump {post_p - pre_p:+.3f}")
    assert 10.0 < post_p - pre_p < 25.0  # measured +16.0
    assert abs((post_p - pre_p) - (post - pre)) < 2.0  # tree and PDE agree on the jump (measured 0.9)


def test_european_one_dividend_methods_agree_and_vs_escrow():
    p = make(option_type="call")
    t_ = tree_price(p, N_TREE, european=True)
    d_ = solve_pde(p, 1600, 1600, european=True).price
    esc_bs = float(bs_price(p.S - p.dividend_pv(), K, T, R, 0.0, 0.2, "call"))
    # the Merton-regime American call equals the European, so the Bermudan LSM must reproduce it (one-sided: LSM <= value)
    q = make(divs=((TD, 0.8 * MERTON),), option_type="call")
    ref = tree_price(q, N_TREE, european=True)
    oos = lsm_both(q, 200_000, 50, 3, "laguerre", seed=5, n_eval=1_000_000)[1]
    z = (oos.price - ref) / oos.stderr
    print(f"[euro] tree {t_:.6f}, pde {d_:.6f}, BS(S - D e^-r td) {esc_bs:.6f} (diff {t_ - esc_bs:+.4f}); LSM(OOS) {oos.price:.5f} +/- {oos.stderr:.5f} vs {ref:.5f}: z={z:+.2f}")
    assert abs(t_ - d_) < 5e-4  # tree N=4000 vs PDE M=N=1600, measured gap below
    assert 0.05 < t_ - esc_bs < 0.2  # model risk, not a bug: measured +0.116 (S vol applies to S, not S - PV(D)); the bound only pins the sign
    assert -4.0 < z < 4.0


def test_escrowed_european_equals_black_scholes_on_escrowed_spot():
    p = make(option_type="call")
    esc = escrowed_price(p, N_TREE, european=True)
    bs = float(bs_price(p.S - p.dividend_pv(), K, T, R, 0.0, 0.2, "call"))
    assert esc == pytest.approx(bs, abs=2.5 / N_TREE * 10)  # O(1/N) tree error


SCENARIOS = {
    "one_dividend_put": make(),
    "quarterly_put": make(divs=((0.25, 1.0), (0.5, 1.0), (0.75, 1.0))),
    "large_dividend_call": make(divs=((TD, 6.0),), option_type="call"),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_cross_method_tree_pde_lsm(name):
    p = SCENARIOS[name]
    ref = dividend_reference(p)
    tree = tree_price(p, aligned_steps(p, 2000))
    pde = pde_price(p, 800)
    n_ex = 48 if name == "quarterly_put" else 50
    berm = bermudan_reference(p, n_ex, 2000)
    oos = lsm_both(p, 200_000, n_ex, 3, "laguerre", seed=31, n_eval=1_000_000)[1]
    z = (oos.price - berm.price) / oos.stderr
    print(f"[cross {name}] reference {ref.price:.5f} (PDE vs tree {ref.uncertainty:.1e}); tree(N=2000) {tree - ref.price:+.2e}; "
          f"PDE(800) {pde - ref.price:+.2e}; Bermudan tree {berm.price:.5f}; LSM(OOS) {oos.price:.5f} +/- {oos.stderr:.5f}, z vs Bermudan {z:+.2f}, "
          f"LSM - American ref {oos.price - ref.price:+.4f}")
    assert ref.uncertainty < 2e-4  # measured 8e-6 .. 5e-5
    assert abs(tree - ref.price) < 3e-3  # tree N=2000 error, measured 3e-4 .. 1.6e-3
    assert abs(pde - ref.price) < 1e-3  # PDE M=N=800, measured <= 4.6e-4
    # one-sided: a Bermudan LSM policy is a lower bound (z < 4); regression shortfall measured up to -0.025 (call, z=-3.5), bound 0.05
    assert z < Z_MAX and -0.05 < oos.price - berm.price
    assert oos.price < ref.price + 4 * oos.stderr



def test_lsm_requires_ex_dates_on_exercise_grid():
    with pytest.raises(ValueError, match="exercise grid"):
        lsm_price(make(divs=((0.33, 1.0),)), 1000, 50)