"""Cox-Ross-Rubinstein binomial tree for European and American options."""
from __future__ import annotations
import math
import numpy as np
from numpy.typing import NDArray
from .params import OptionParams


# Ties between exercise and continuation are not exercise
EXERCISE_TOL = 1e-12


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


def induct(params: OptionParams, n_steps: int, european: bool, track_boundary: bool) -> tuple[float, NDArray[np.float64] | None]:
    """Backward induction; optionally record the exercise boundary."""
    _, u, p, disc = crr_factors(params, n_steps)
    log_u = math.log(u)
    is_put = params.option_type == "put"

    def nodes(n: int) -> NDArray[np.float64]:
        # Stock price at step n for j = 0..n up-moves: S * u^(2j - n)
        return params.S * np.exp(log_u * (2.0 * np.arange(n + 1) - n))

    boundary = np.full(n_steps + 1, np.nan) if track_boundary else None

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
        if european:
            values = cont
        else:
            s = nodes(n)
            exercise = params.payoff(s)
            if track_boundary:
                record(n, s, exercise, cont)
            values = np.maximum(exercise, cont)
    return float(values[0]), boundary


def tree_price(params: OptionParams, n_steps: int = 500, european: bool = False) -> float:
    """Price with an ``n_steps`` CRR tree (American by default)."""
    return induct(params, n_steps, european, track_boundary=False)[0]


def exercise_boundary(params: OptionParams, n_steps: int = 500) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Early-exercise boundary of an American option on the CRR tree."""
    _, boundary = induct(params, n_steps, european=False, track_boundary=True)
    return np.linspace(0.0, params.T, n_steps + 1), boundary