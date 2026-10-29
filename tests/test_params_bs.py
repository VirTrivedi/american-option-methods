import numpy as np
import pytest
from aopt.bs import bs_delta, bs_price, bs_gamma
from aopt.params import OptionParams


def make(**kw):
    base = dict(S=100.0, K=100.0, T=1.0, r=0.05, q=0.02, sigma=0.2, option_type="put")
    return OptionParams(**{**base, **kw})


@pytest.mark.parametrize("bad", [dict(S=0), dict(K=-1), dict(T=0), dict(sigma=-0.1), dict(r=float("nan")), dict(q=float("inf")), dict(option_type="straddle")])
def test_params_reject_invalid(bad):
    with pytest.raises(ValueError):
        make(**bad)


def test_params_frozen():
    with pytest.raises(Exception):
        make().S = 5.0


def test_bs_textbook_value():
    # Hull-style check: S=K=100, T=1, r=5%, q=0, sigma=20% -> call 10.4506, put 5.5735
    assert bs_price(100, 100, 1, 0.05, 0.0, 0.2, "call") == pytest.approx(10.450584, abs=1e-6)
    assert bs_price(100, 100, 1, 0.05, 0.0, 0.2, "put") == pytest.approx(5.573526, abs=1e-6)


def test_bs_put_call_parity_vectorized():
    S = np.linspace(50, 150, 11)
    c = bs_price(S, 100, 0.75, 0.04, 0.03, 0.3, "call")
    p = bs_price(S, 100, 0.75, 0.04, 0.03, 0.3, "put")
    np.testing.assert_allclose(c - p, S * np.exp(-0.03 * 0.75) - 100 * np.exp(-0.04 * 0.75), atol=1e-12)


def test_bs_delta_matches_finite_difference():
    h = 1e-4
    for typ in ("call", "put"):
        fd = (bs_price(100 + h, 100, 1, 0.05, 0.02, 0.2, typ) - bs_price(100 - h, 100, 1, 0.05, 0.02, 0.2, typ)) / (2 * h)
        assert bs_delta(100, 100, 1, 0.05, 0.02, 0.2, typ) == pytest.approx(fd, abs=1e-7)


def test_bs_gamma_matches_finite_difference():
    h = 1e-2
    f = lambda s: bs_price(s, 100, 1, 0.05, 0.02, 0.2, "call")
    fd = (f(100 + h) - 2 * f(100) + f(100 - h)) / h**2
    assert bs_gamma(100, 100, 1, 0.05, 0.02, 0.2) == pytest.approx(fd, abs=1e-6)