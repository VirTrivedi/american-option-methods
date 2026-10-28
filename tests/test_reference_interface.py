import numpy as np
import pytest
from aopt.bs import bs_price
from aopt.interface import price
from aopt.params import OptionParams
from aopt.reference import averaged_price, reference_price
from aopt.tree import tree_price


P = OptionParams(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.2, option_type="put")


def test_reference_matches_literature_and_reports_error():
    ref = reference_price(P, n_base=500)
    assert ref.price == pytest.approx(6.0904, abs=2e-4)
    assert 0.0 <= ref.error_estimate < 1e-3


def test_reference_error_estimate_is_honest():
    # Compare a cheap reference against a much finer one: actual gap should be within ~3x estimate
    coarse, fine = reference_price(P, n_base=250), reference_price(P, n_base=2000)
    assert abs(coarse.price - fine.price) <= 3 * coarse.error_estimate + fine.error_estimate


def test_reference_european_case_matches_bs():
    # Where the American put has no premium (r=0), the extrapolated value must hit Black-Scholes
    p = OptionParams(100.0, 100.0, 1.0, 0.0, 0.0, 0.2, "put")
    assert reference_price(p, 500).price == pytest.approx(float(bs_price(100, 100, 1, 0, 0, 0.2, "put")), abs=1e-5)


def test_convergence_order_against_reference():
    """Log-log slope of |V_N - V_ref| vs N: raw CRR is first order (slope ~ -1) on average."""
    ref = reference_price(P, n_base=2000).price
    ns = np.array([50, 100, 200, 400, 800, 1600])
    errs = np.array([abs(tree_price(P, int(n)) - ref) for n in ns])
    slope = np.polyfit(np.log(ns), np.log(errs), 1)[0]
    assert -1.4 < slope < -0.6, f"slope {slope:.2f}"
    # Averaged tree should be strictly more accurate than raw tree at every size
    avg_errs = np.array([abs(averaged_price(P, int(n)) - ref) for n in ns])
    assert np.all(avg_errs < errs)


def test_interface_dispatch():
    assert price(P, "tree", n_steps=200) == tree_price(P, 200)
    assert price(P, "tree", n_steps=200, european=True) == tree_price(P, 200, european=True)
    assert price(P, "tree_reference", n_base=250) == pytest.approx(reference_price(P, 250).price)


@pytest.mark.parametrize("method", ["pde", "lsm"])
def test_interface_unimplemented(method):
    with pytest.raises(NotImplementedError, match="not implemented"):
        price(P, method)


def test_interface_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        price(P, "garch")