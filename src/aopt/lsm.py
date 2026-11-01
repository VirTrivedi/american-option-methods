"""Longstaff-Schwartz Monte Carlo for Bermudan options under GBM with continuous yield."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import eval_laguerre
from .bs import bs_price
from .params import OptionParams


BasisFamily = Literal["monomial", "laguerre", "chebyshev"]
Mode = Literal["in_sample", "out_of_sample"]
DEFAULT_MAX_MEMORY_BYTES = 2_000_000_000
Z95 = 1.959963984540054


@dataclass(frozen=True)
class LSMResult:
    """LSM estimate with its standard error and the fitted per-date regression coefficients."""
    price: float
    stderr: float
    ci95: tuple[float, float]
    mode: Mode
    coefficients: NDArray[np.float64]
    n_paths: int
    n_eval: int
    n_ex: int
    degree: int
    basis: BasisFamily
    antithetic: bool
    seed: int
    exercise_index: NDArray[np.int32] | None = None
    european_basis: bool = False


def basis_matrix(s: NDArray[np.float64], strike: float, family: BasisFamily, degree: int, option_type: str) -> NDArray[np.float64]:
    """Basis functions of x = S/K (columns 0..degree), well scaled so QR/lstsq stays stable."""
    x = np.asarray(s, dtype=float) / strike
    if family == "monomial":
        return np.vander(x, degree + 1, increasing=True)
    
    if family == "laguerre":
        w = np.exp(-0.5 * x)
        return np.stack([w * eval_laguerre(j, x) for j in range(degree + 1)], axis=1)
    
    if family == "chebyshev":
        c, w = (0.5, 0.5) if option_type == "put" else (1.5, 1.0)
        return np.polynomial.chebyshev.chebvander((x - c) / w, degree)
    
    raise ValueError(f"basis must be 'monomial', 'laguerre' or 'chebyshev', got {family!r}")


def design_matrix(params: OptionParams, s: NDArray[np.float64], k: int, n_ex: int, family: BasisFamily,
                  degree: int, european_basis: bool) -> NDArray[np.float64]:
    """Regression design at exercise date k < n_ex; optionally adds the European option value / K as a column."""
    X = basis_matrix(s, params.K, family, degree, params.option_type)
    if european_basis:
        tau = params.T * (1.0 - k / n_ex)
        s_eur = s - params.dividend_pv(params.T - tau, inclusive=True) if params.dividends else s  # escrowed spot (approximate)
        s_eur = np.maximum(s_eur, 1e-8 * params.K)
        ev = bs_price(s_eur, params.K, tau, params.r, params.q, params.sigma, params.option_type) / params.K
        X = np.column_stack([X, ev])

    return X


def training_bytes(n_paths: int, n_ex: int, degree: int) -> int:
    """Approximate peak training memory: the path matrix plus ~10 work vectors and the basis matrix."""
    return 8 * n_paths * (n_ex + 10 + (degree + 1))


def max_train_paths(n_ex: int, degree: int, max_memory_bytes: int = DEFAULT_MAX_MEMORY_BYTES) -> int:
    """Largest training path count that fits in ``max_memory_bytes``."""
    return max_memory_bytes // (8 * (n_ex + 10 + (degree + 1)))


def simulate_paths(params: OptionParams, n_paths: int, n_ex: int, rng: np.random.Generator, antithetic: bool) -> NDArray[np.float64]:
    """Exact GBM at dates 1..n_ex, shape (n_paths, n_ex). With antithetic, rows [n/2:] mirror rows [:n/2]."""
    dt = params.T / n_ex
    drift = (params.r - params.q - 0.5 * params.sigma**2) * dt
    vol = params.sigma * math.sqrt(dt)
    out = np.empty((n_paths, n_ex))
    if antithetic:
        half = n_paths // 2
        out[:half] = rng.standard_normal((half, n_ex))
        out[half:] = -out[:half]
    else:
        out[:] = rng.standard_normal((n_paths, n_ex))

    out *= vol
    out += drift
    start = np.full(n_paths, math.log(params.S))
    prev, divs = 0, div_steps(params, n_ex)
    for stop in [*sorted(divs), n_ex]:  # segments between ex-dates; the cum-dividend price is stored
        seg = out[:, prev:stop]
        np.cumsum(seg, axis=1, out=seg)
        seg += start[:, None]
        if stop < n_ex:
            start = np.log(params.ex_dividend(np.exp(seg[:, -1]), divs[stop]))

        prev = stop

    np.exp(out, out=out)

    return out


def div_steps(params: OptionParams, n_ex: int) -> dict[int, float]:
    """{exercise-date index k: cash dividend}; ex-dates must lie exactly on the exercise grid t_k = k*T/n_ex."""
    out: dict[int, float] = {}
    for t, d in params.dividends:
        k = round(t * n_ex / params.T)
        if abs(k * params.T / n_ex - t) > 1e-9 * params.T or not 1 <= k <= n_ex - 1:
            raise ValueError(f"ex-date {t} is not on the exercise grid k*T/n_ex (n_ex={n_ex}); choose n_ex so that t*n_ex/T is an integer")
        
        out[k] = d

    return out


def check_args(n_paths: int, n_ex: int, degree: int, antithetic: bool) -> None:
    if n_ex < 2:
        raise ValueError("n_ex must be >= 2 (one regression date plus maturity)")
    
    if degree < 0 or n_paths < 2:
        raise ValueError("need degree >= 0 and n_paths >= 2")
    
    if antithetic and n_paths % 2:
        raise ValueError("antithetic sampling needs an even n_paths")


def unit_values(values: NDArray[np.float64], antithetic: bool) -> NDArray[np.float64]:
    """Independent sampling units: antithetic pair means, or the paths themselves."""
    if not antithetic:
        return values
    
    half = values.size // 2

    return 0.5 * (values[:half] + values[half:])


def mean_stderr(units: NDArray[np.float64]) -> tuple[float, float]:
    return float(units.mean()), float(units.std(ddof=1) / math.sqrt(units.size))


def fit(params: OptionParams, S: NDArray[np.float64], n_ex: int, degree: int, family: BasisFamily, european_basis: bool = False):
    """Backward LSM regression; returns (coefficients, discounted path values at t=0, exercise index)."""
    disc = math.exp(-params.r * params.T / n_ex)
    n_basis = degree + 1 + int(european_basis)
    min_itm = max(10, 3 * n_basis)  # below this a regression is meaningless: do not exercise at that date
    cf = params.payoff(S[:, -1])  # cashflow valued at the current date
    exercise_index = np.where(cf > 0.0, n_ex, 0).astype(np.int32)
    coeffs = np.full((n_ex - 1, n_basis), np.nan)
    for k in range(n_ex - 1, 0, -1):
        cf *= disc  # now valued at date k
        s = S[:, k - 1]
        pay = params.payoff(s)
        itm = np.flatnonzero(pay > 0.0)
        if itm.size < min_itm:
            continue

        X = design_matrix(params, s[itm], k, n_ex, family, degree, european_basis)
        coef = np.linalg.lstsq(X, cf[itm], rcond=None)[0]
        exercise = pay[itm] > X @ coef
        chosen = itm[exercise]
        cf[chosen] = pay[chosen]
        exercise_index[chosen] = k
        coeffs[k - 1] = coef

    return coeffs, cf * disc, exercise_index


def evaluate_policy(params: OptionParams, coeffs: NDArray[np.float64], degree: int, family: BasisFamily, n_eval: int,
    rng: np.random.Generator, antithetic: bool, chunk_size: int = 100_000, european_basis: bool = False) -> tuple[float, float]:
    """Mean and standard error of the stored exercise policy on fresh paths, simulated in chunks."""
    n_ex = coeffs.shape[0] + 1
    dt = params.T / n_ex
    drift = (params.r - params.q - 0.5 * params.sigma**2) * dt
    vol = params.sigma * math.sqrt(dt)
    divs = div_steps(params, n_ex)
    chunk_size = max(2, chunk_size - chunk_size % 2) if antithetic else max(1, chunk_size)
    count, mean, m2 = 0, 0.0, 0.0  # Chan et al. running merge of chunk statistics
    remaining = n_eval
    while remaining > 0:
        c = min(chunk_size, remaining + (remaining % 2 if antithetic else 0))
        remaining -= c
        half = c // 2
        logS = np.full(c, math.log(params.S))
        alive = np.ones(c, dtype=bool)
        value = np.zeros(c)
        for k in range(1, n_ex + 1):
            if k - 1 in divs:  # drop at the previous date, after its exercise decision
                logS = np.log(params.ex_dividend(np.exp(logS), divs[k - 1]))

            if antithetic:
                z = rng.standard_normal(half)
                z = np.concatenate([z, -z])
            else:
                z = rng.standard_normal(c)

            logS += drift + vol * z
            s = np.exp(logS)
            pay = params.payoff(s)
            cand = np.flatnonzero(alive & (pay > 0.0))
            if k < n_ex:
                coef = coeffs[k - 1]
                if np.isnan(coef[0]) or cand.size == 0:
                    continue

                X = design_matrix(params, s[cand], k, n_ex, family, degree, european_basis)
                cand = cand[pay[cand] > X @ coef]

            value[cand] = pay[cand] * math.exp(-params.r * dt * k)
            alive[cand] = False

        units = unit_values(value, antithetic)
        n_b, mean_b = units.size, float(units.mean())
        m2_b = float(((units - mean_b) ** 2).sum())
        delta = mean_b - mean
        total = count + n_b
        mean += delta * n_b / total
        m2 += m2_b + delta**2 * count * n_b / total
        count = total

    return mean, math.sqrt(m2 / (count - 1) / count)


def result(mode, price, se, coeffs, n_paths, n_eval, n_ex, degree, family, antithetic, seed, policy, european_basis=False):
    return LSMResult(price=price, stderr=se, ci95=(price - Z95 * se, price + Z95 * se), mode=mode, coefficients=coeffs, 
                     n_paths=n_paths, n_eval=n_eval, n_ex=n_ex, degree=degree, basis=family, antithetic=antithetic, 
                     seed=seed, exercise_index=policy, european_basis=european_basis)


def fit_lsm(params: OptionParams, n_paths: int = 100_000, n_ex: int = 50, degree: int = 3, basis: BasisFamily = "laguerre", seed: int = 0, 
            antithetic: bool = True, return_policy: bool = False, max_memory_bytes: int = DEFAULT_MAX_MEMORY_BYTES, european_basis: bool = False) -> LSMResult:
    """Simulate training paths (``default_rng(seed)``), run the backward regression, return the in-sample result."""
    check_args(n_paths, n_ex, degree, antithetic)
    need = training_bytes(n_paths, n_ex, degree)
    if need > max_memory_bytes:
        raise ValueError(
            f"{n_paths} training paths x {n_ex} dates needs ~{need / 1e9:.1f} GB (limit {max_memory_bytes / 1e9:.1f} GB, "
            f"max {max_train_paths(n_ex, degree, max_memory_bytes)} paths); train on fewer paths and raise n_eval instead"
        )
    
    S = simulate_paths(params, n_paths, n_ex, np.random.default_rng(seed), antithetic)
    coeffs, v0, ex_idx = fit(params, S, n_ex, degree, basis, european_basis)
    del S
    price, se = mean_stderr(unit_values(v0, antithetic))

    return result("in_sample", price, se, coeffs, n_paths, n_paths, n_ex, degree, basis, antithetic, seed, ex_idx if return_policy else None, european_basis)


def evaluate_fit(params: OptionParams, fit: LSMResult, n_eval: int | None = None, chunk_size: int = 100_000, eval_seed: int | None = None) -> LSMResult:
    """Apply a fitted policy to fresh independent paths; ``n_eval`` defaults to n_paths."""
    n_eval = fit.n_paths if n_eval is None else n_eval
    rng = np.random.default_rng([fit.seed, 1] if eval_seed is None else [eval_seed, 2])
    price, se = evaluate_policy(params, fit.coefficients, fit.degree, fit.basis, n_eval, rng, fit.antithetic, chunk_size, fit.european_basis)

    return result("out_of_sample", price, se, fit.coefficients, fit.n_paths, n_eval, fit.n_ex, fit.degree, fit.basis, fit.antithetic, fit.seed, None, fit.european_basis)


def lsm_both(params: OptionParams, n_paths: int = 100_000, n_ex: int = 50, degree: int = 3, basis: BasisFamily = "laguerre", seed: int = 0, 
             antithetic: bool = True, n_eval: int | None = None, chunk_size: int = 100_000, return_policy: bool = False, 
             max_memory_bytes: int = DEFAULT_MAX_MEMORY_BYTES, european_basis: bool = False, eval_seed: int | None = None) -> tuple[LSMResult, LSMResult]:
    """Fit once; return (in-sample, out-of-sample). Training and evaluation use independent random streams."""
    fit = fit_lsm(params, n_paths, n_ex, degree, basis, seed, antithetic, return_policy, max_memory_bytes, european_basis)

    return fit, evaluate_fit(params, fit, n_eval, chunk_size, eval_seed)


def lsm_price(params: OptionParams, n_paths: int = 100_000, n_ex: int = 50, degree: int = 3, basis: BasisFamily = "laguerre", seed: int = 0, 
              antithetic: bool = True, mode: Mode = "in_sample", n_eval: int | None = None, chunk_size: int = 100_000, return_policy: bool = False, 
              max_memory_bytes: int = DEFAULT_MAX_MEMORY_BYTES, european_basis: bool = False, eval_seed: int | None = None) -> LSMResult:
    """LSM estimate in the requested mode; in-sample skips the out-of-sample evaluation entirely."""
    if mode not in ("in_sample", "out_of_sample"):
        raise ValueError(f"mode must be 'in_sample' or 'out_of_sample', got {mode!r}")
    
    fit = fit_lsm(params, n_paths, n_ex, degree, basis, seed, antithetic, return_policy, max_memory_bytes, european_basis)

    return fit if mode == "in_sample" else evaluate_fit(params, fit, n_eval, chunk_size, eval_seed)


def lsm_boundary(params: OptionParams, result: LSMResult, n_grid: int = 4000) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Exercise boundary at dates 1..n_ex-1 implied by the fitted coefficients, unsmoothed."""
    n_ex = result.coefficients.shape[0] + 1
    is_put = params.option_type == "put"
    grid = np.linspace(0.2 * params.K, params.K, n_grid) if is_put else np.linspace(params.K, 5.0 * params.K, n_grid)
    sign = -1.0 if is_put else 1.0
    out = np.full(n_ex - 1, np.nan)
    for k in range(1, n_ex):
        coef = result.coefficients[k - 1]
        if np.isnan(coef[0]):
            continue

        def gap(s: float) -> float:  # exercise value minus fitted continuation
            X = design_matrix(params, np.array([s]), k, n_ex, result.basis, result.degree, result.european_basis)
            return float(sign * (s - params.K) - (X @ coef)[0])

        f = sign * (grid - params.K) - design_matrix(params, grid, k, n_ex, result.basis, result.degree, result.european_basis) @ coef
        hits = np.flatnonzero(f >= 0.0)
        if hits.size == 0:
            continue

        i = int(hits.max() if is_put else hits.min())
        j = i + 1 if is_put else i - 1  # neighbouring grid point on the continuation side
        if not 0 <= j < n_grid:
            out[k - 1] = grid[i]
        elif f[i] == 0.0:
            out[k - 1] = grid[i]
        else:
            out[k - 1] = brentq(gap, min(grid[i], grid[j]), max(grid[i], grid[j]))

    return np.arange(1, n_ex) * params.T / n_ex, out