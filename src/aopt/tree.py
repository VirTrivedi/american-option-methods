"""Cox-Ross-Rubinstein binomial tree for European and American options."""
from __future__ import annotations
import math
import warnings
from collections.abc import Collection
from fractions import Fraction
import numpy as np
from numpy.typing import NDArray
from .params import OptionParams


# Ties between exercise and continuation are not exercise
EXERCISE_TOL = 1e-12
ALIGN_TOL = 1e-9  # an ex-date is "on" the tree when t/dt is within this of an integer


def crr_factors(params: OptionParams, n_steps: int) -> tuple[float, float, float, float]:
    """Return (dt, u, p, discount) for the CRR tree with risk-neutral drift r - q."""
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
    dt = params.T / n_steps
    u = math.exp(params.sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((params.r - params.q) * dt) - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise ValueError(
            f"risk-neutral probability p={p:.4f} is outside [0, 1]; the CRR tree is "
            f"unstable for these inputs (|r-q|*sqrt(dt)/sigma too large). Increase n_steps "
            f"(currently {n_steps}) or check r, q, sigma."
        )
    return dt, u, p, math.exp(-params.r * dt)


def exercise_mask(n_steps: int, european: bool, exercise_steps: Collection[int] | None) -> NDArray[np.bool_]:
    """allowed[n] is True if early exercise is permitted at tree step n (maturity is always a payoff)."""
    if european and exercise_steps is not None:
        raise ValueError("pass either european=True or exercise_steps, not both")
    
    if european:
        return np.zeros(n_steps + 1, dtype=bool)
    
    if exercise_steps is None:
        return np.ones(n_steps + 1, dtype=bool)
    
    steps = np.asarray(sorted(exercise_steps), dtype=int)
    if steps.size and (steps[0] < 0 or steps[-1] > n_steps):
        raise ValueError(f"exercise_steps must lie in [0, {n_steps}]")
    
    allowed = np.zeros(n_steps + 1, dtype=bool)
    allowed[steps] = True

    return allowed


def bermudan_steps(n_steps: int, n_ex: int) -> frozenset[int]:
    """Tree steps of n_ex equally spaced exercise dates T/n_ex, ..., T (maturity included, t = 0 excluded)."""
    if n_ex < 1 or n_steps % n_ex != 0:
        raise ValueError(f"n_steps={n_steps} must be a positive multiple of n_ex={n_ex}")
    
    return frozenset(k * (n_steps // n_ex) for k in range(1, n_ex + 1))


def alignment_unit(params: OptionParams) -> int:
    """Smallest N0 such that every N that is a multiple of N0 puts all ex-dates exactly on tree steps."""
    unit = 1
    for t, _ in params.dividends:
        frac = Fraction(t / params.T).limit_denominator(10_000)
        if abs(float(frac) - t / params.T) > 1e-12:
            raise ValueError(f"ex-time {t} is not a simple fraction of T={params.T}; no exact alignment exists")
        
        unit = math.lcm(unit, frac.denominator)

    return unit


def aligned_steps(params: OptionParams, n_min: int, also_multiple_of: int = 1) -> int:
    """Smallest N >= n_min with all ex-dates on tree steps (and a multiple of ``also_multiple_of``, e.g. n_ex)."""
    unit = math.lcm(alignment_unit(params), also_multiple_of)

    return unit * max(1, math.ceil(n_min / unit))


def dividend_steps(params: OptionParams, n_steps: int) -> dict[int, float]:
    """{tree step: cash amount}. Ex-dates are snapped to the nearest step; a warning reports the time error."""
    dt = params.T / n_steps
    out: dict[int, float] = {}
    for t, d in params.dividends:
        step = round(t / dt)
        if not 1 <= step <= n_steps - 1:
            raise ValueError(f"ex-date {t} snaps to step {step}, outside 1..{n_steps - 1}; use more steps")
        
        if step in out:
            raise ValueError(f"two ex-dates snap to step {step}; use more steps")
        
        err = abs(step * dt - t)
        if err > ALIGN_TOL * params.T:
            warnings.warn(f"ex-date {t} snapped to step {step} (t={step * dt:.6f}); snapping error {err:.2e} years. "
                          f"aligned_steps() gives an N with exact alignment", stacklevel=3)
            
        out[step] = d

    return out


def induct(params: OptionParams, n_steps: int, european: bool, track_boundary: bool, 
           exercise_steps: Collection[int] | None = None) -> tuple[float, NDArray[np.float64] | None]:
    """Backward induction; optionally record the exercise boundary."""
    _, u, p, disc = crr_factors(params, n_steps)
    allowed = exercise_mask(n_steps, european, exercise_steps)
    log_u = math.log(u)
    is_put = params.option_type == "put"

    def nodes(n: int) -> NDArray[np.float64]:
        # Stock price at step n for j = 0..n up-moves: S * u^(2j - n)
        return params.S * np.exp(log_u * (2.0 * np.arange(n + 1) - n))

    boundary = np.full(n_steps + 1, np.nan) if track_boundary else None
    div_at = dividend_steps(params, n_steps)

    def record(n: int, s: NDArray[np.float64], exercise: NDArray[np.float64], cont: NDArray[np.float64]) -> None:
        mask = (exercise > cont + EXERCISE_TOL) & (exercise > 0.0)
        if mask.any():  # s is ascending: puts exercise below S*, calls above it
            boundary[n] = s[mask].max() if is_put else s[mask].min()

    s_T = nodes(n_steps)
    values = params.payoff(s_T)
    if track_boundary:
        record(n_steps, s_T, values, np.zeros_like(values))

    for n in range(n_steps - 1, -1, -1):
        cont = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if n in div_at:
            # cont is the value just after the drop, at post-dividend price s_j; the pre-dividend value at s_j is that
            # layer linearly interpolated at s_j - D, extrapolated linearly below the lowest node
            s = nodes(n)
            y = params.ex_dividend(s, div_at[n])
            below = y < s[0]
            cont_y = np.interp(y, s, cont)
            if below.any():
                cont_y[below] = cont[0] + (y[below] - s[0]) * (cont[1] - cont[0]) / (s[1] - s[0])

            cont = cont_y

        if not allowed[n]:
            values = cont
        else:
            s = nodes(n)
            exercise = params.payoff(s)
            if track_boundary:
                record(n, s, exercise, cont)

            values = np.maximum(exercise, cont)

    return float(values[0]), boundary


def tree_price(params: OptionParams, n_steps: int = 500, european: bool = False, exercise_steps: Collection[int] | None = None) -> float:
    """Price with an ``n_steps`` CRR tree (American by default)."""
    return induct(params, n_steps, european, False, exercise_steps)[0]


def exercise_boundary(params: OptionParams, n_steps: int = 500, exercise_steps: Collection[int] | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Early-exercise boundary on the CRR tree (American, or Bermudan with ``exercise_steps``)."""
    _, boundary = induct(params, n_steps, False, True, exercise_steps)
    return np.linspace(0.0, params.T, n_steps + 1), boundary


def escrowed_price(params: OptionParams, n_steps: int = 500, european: bool = False) -> float:
    """Escrowed-dividend model: CRR tree on S* = S - PV(dividends), payoff on S* + PV(remaining dividends)."""
    s0_star = params.S - params.dividend_pv(0.0)
    if s0_star <= 0.0:
        raise ValueError("present value of dividends exceeds the spot; the escrowed model is undefined")
    
    base = OptionParams(S=s0_star, K=params.K, T=params.T, r=params.r, q=params.q, sigma=params.sigma, option_type=params.option_type)
    _, u, p, disc = crr_factors(base, n_steps)
    log_u, dt = math.log(u), params.T / n_steps

    def spot(n: int) -> NDArray[np.float64]:
        return s0_star * np.exp(log_u * (2.0 * np.arange(n + 1) - n)) + params.dividend_pv(n * dt, inclusive=True)

    values = params.payoff(spot(n_steps))
    for n in range(n_steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if not european:
            values = np.maximum(values, params.payoff(spot(n)))

    return float(values[0])