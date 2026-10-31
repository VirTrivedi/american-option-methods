"""Crank-Nicolson finite-difference pricer for European and American options."""
from __future__ import annotations
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_banded
from .params import OptionParams


AmericanMethod = Literal["psor", "penalty"]
# Hook applied at an event time: (S_grid, V) -> V just before the event (earlier in calendar time).
EventHook = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class Event:
    """A jump condition applied at calendar time ``t`` (0 < t < T), e.g. an ex-dividend date."""
    t: float
    apply: EventHook


@dataclass(frozen=True)
class PDEResult:
    """Solver output. ``S``/``V`` is the full grid at t = 0; boundary arrays are None unless requested."""
    price: float
    delta: float
    gamma: float
    S: NDArray[np.float64]
    V: NDArray[np.float64]
    boundary_t: NDArray[np.float64] | None
    boundary_S: NDArray[np.float64] | None
    mean_iterations: float  # PSOR sweeps or penalty Newton iterations per time step (0 if European)
    n_steps: int  # linear-system time steps actually taken (includes Rannacher half-steps)


def default_half_width(params: OptionParams) -> float:
    """L = max(5*sigma*sqrt(T) + |r-q|*T, 1.5), widened by |ln(S0/K)| so S0 is inside the domain."""
    base = max(5.0 * params.sigma * math.sqrt(params.T) + abs(params.r - params.q) * params.T, 1.5)

    return base + abs(math.log(params.S / params.K))


def make_grid(params: OptionParams, M: int, L: float) -> tuple[NDArray[np.float64], float, int]:
    """Uniform log-grid with M intervals, shifted so that ln(S0) is node i0."""
    h = 2.0 * L / M
    i0 = round((math.log(params.S) - (math.log(params.K) - L)) / h)
    if not 1 <= i0 <= M - 1:
        raise ValueError(f"S0={params.S} lies outside the PDE domain; increase L or M")
    x = math.log(params.S) + (np.arange(M + 1) - i0) * h

    return x, h, i0


def time_steps(T: float, N: int, event_taus: Sequence[float], rannacher: bool) -> tuple[list[tuple[float, float]], dict[int, int]]:
    """Return [(dtau, theta)] and {step index: event index applied after that step}."""
    bounds = [0.0, *event_taus, T]
    steps: list[tuple[float, float]] = []
    event_after: dict[int, int] = {}
    for j in range(len(bounds) - 1):
        span = bounds[j + 1] - bounds[j]
        n = max(1, round(N * span / T))
        dt = span / n
        k = min(2, n) if rannacher else 0
        steps += [(dt / 2.0, 1.0)] * (2 * k) + [(dt, 0.5)] * (n - k)
        if j < len(bounds) - 2:
            event_after[len(steps) - 1] = j

    return steps, event_after


def boundary_values(params: OptionParams, S_lo: float, S_hi: float, tau: float, european: bool) -> tuple[float, float]:
    """Dirichlet values at the lower/upper end of the grid at time-to-maturity tau."""
    K, r, q = params.K, params.r, params.q
    if params.option_type == "put":
        lo = K * math.exp(-r * tau) - S_lo * math.exp(-q * tau)
        return (lo if european else max(lo, K - S_lo)), 0.0
    
    hi = S_hi * math.exp(-q * tau) - K * math.exp(-r * tau)

    return 0.0, (hi if european else max(hi, S_hi - K))


def exercise_boundary_point(S: NDArray[np.float64], V: NDArray[np.float64], g: NDArray[np.float64], is_put: bool, tol: float) -> float:
    """Critical price at one time level, smoothed below node resolution; NaN if no exercise region."""
    gap = V - g
    in_ex = np.flatnonzero((gap <= tol) & (g > 0.0))
    if in_ex.size == 0:
        return math.nan
    
    j = int(in_ex.max() if is_put else in_ex.min())
    step = 1 if is_put else -1
    a, b = j + step, j + 2 * step  # nearest and next-nearest continuation nodes
    if not (0 <= b < S.size):
        return float(S[j])
    
    sa, sb = math.sqrt(max(gap[a], 0.0)), math.sqrt(max(gap[b], 0.0))
    if sb <= sa or g[a] <= 0.0:  # no usable slope, or the bracketing node is already beyond the kink
        return float(S[j])
    
    s_star = S[a] - sa * (S[b] - S[a]) / (sb - sa)
    far = 2.0 * S[j] - S[a]  # one spacing on the contact side of S_j

    return float(min(max(s_star, min(far, S[a])), max(far, S[a])))


def solve_pde(params: OptionParams, M: int = 400, N: int = 400, *, european: bool = False, american_method: AmericanMethod = "psor",
              rannacher: bool = True, L: float | None = None, omega: float | None = None, tol: float = 1e-10, max_iter: int = 10_000,
              penalty: float = 1e8, events: Sequence[Event] = (), return_boundary: bool = False) -> PDEResult:
    """Price with Crank-Nicolson on M space intervals and N time steps.
    Args:
        american_method: "psor" (red-black projected SOR, vectorized) or "penalty" (penalty with semi-smooth Newton).
        omega: SOR relaxation; None picks the optimal value for the step's tridiagonal system.
        tol: PSOR stops when the largest update in a sweep is below ``tol``.
        penalty: penalty coefficient rho (1/time); the constraint violation is O(1/rho).
        events: jump conditions applied at their times (see ``Event``).
        return_boundary: also compute the early-exercise boundary at every time step."""
    if M < 4 or N < 1:
        raise ValueError("need M >= 4 space intervals and N >= 1 time steps")
    
    if american_method not in ("psor", "penalty"):
        raise ValueError(f"american_method must be 'psor' or 'penalty', got {american_method!r}")
    
    T, K, r, q, sigma = params.T, params.K, params.r, params.q, params.sigma
    is_put = params.option_type == "put"
    x, h, i0 = make_grid(params, M, default_half_width(params) if L is None else L)
    S = np.exp(x)
    g = params.payoff(S)

    mu = r - q - 0.5 * sigma**2
    a = 0.5 * sigma**2 / h**2 - 0.5 * mu / h  # coefficients of (A V)_i = a V_{i-1} + b V_i + c V_{i+1}
    b = -(sigma**2) / h**2 - r
    c = 0.5 * sigma**2 / h**2 + 0.5 * mu / h

    ordered = sorted(events, key=lambda e: e.t)
    if any(not 0.0 < e.t < T for e in ordered) or any(u.t == v.t for u, v in zip(ordered, ordered[1:])):
        raise ValueError("event times must be distinct and lie strictly inside (0, T)")
    
    steps, event_after = time_steps(T, N, [T - e.t for e in reversed(ordered)], rannacher)
    hooks = [e.apply for e in reversed(ordered)]  # ascending tau

    V = g.copy()
    tau = 0.0
    total_iter = 0
    b_tau: list[float] = [0.0]  # tau at each recorded boundary point
    b_S: list[float] = [K] if return_boundary else []
    ex_tol = 1e-8 * K
    active = np.zeros(M - 1, dtype=bool)  # penalty warm start: nodes exercised at the previous step

    for idx, (dt, theta) in enumerate(steps):
        tau += dt
        lo, hi = boundary_values(params, S[0], S[-1], tau, european)
        rhs = V[1:-1].copy()
        if theta < 1.0:
            rhs += (1.0 - theta) * dt * (a * V[:-2] + b * V[1:-1] + c * V[2:])

        low, diag, up = -theta * dt * a, 1.0 - theta * dt * b, -theta * dt * c
        Vn = V.copy()
        Vn[0], Vn[-1] = lo, hi

        if european:
            rhs[0] -= low * lo
            rhs[-1] -= up * hi
            ab = np.zeros((3, M - 1))
            ab[0, 1:], ab[1], ab[2, :-1] = up, diag, low
            Vn[1:-1] = solve_banded((1, 1), ab, rhs)
        elif american_method == "psor":
            w = omega
            if w is None:
                rho_j = 2.0 * math.sqrt(low * up) / diag if low * up > 0 else 0.0
                w = 2.0 / (1.0 + math.sqrt(1.0 - rho_j**2))
            Vn, sweeps = psor_step(Vn, rhs, g, low, diag, up, w, tol, max_iter)
            total_iter += sweeps
        else:
            rhs[0] -= low * lo
            rhs[-1] -= up * hi
            Vn, n_newton, active = penalty_step(Vn, rhs, g, low, diag, up, dt, penalty, active, max_iter)
            total_iter += n_newton

        V = Vn

        if idx in event_after:
            V = hooks[event_after[idx]](S, V)

        if return_boundary and not european:
            b_tau.append(tau)
            b_S.append(exercise_boundary_point(S, V, g, is_put, ex_tol))

    b_tau[-1] = T  # snap accumulated floating-point drift
    # Greeks at S0 (node i0): V_S = V_x/S, V_SS = (V_xx - V_x)/S^2.
    _, delta_grid, gamma_grid = grid_greeks(x, V)

    return PDEResult(price=float(V[i0]), delta=float(delta_grid[i0 - 1]), gamma=float(gamma_grid[i0 - 1]), S=S, V=V,
        boundary_t=np.maximum(T - np.array(b_tau), 0.0)[::-1] if return_boundary and not european else None,
        boundary_S=np.array(b_S)[::-1] if return_boundary and not european else None,
        mean_iterations=total_iter / len(steps), n_steps=len(steps),)


def psor_step(V: NDArray[np.float64], rhs: NDArray[np.float64], g: NDArray[np.float64], low: float, diag: float,
              up: float, omega: float, tol: float, max_iter: int) -> tuple[NDArray[np.float64], int]:
    """Red-black projected SOR for  diag*V_i + low*V_{i-1} + up*V_{i+1} = rhs_i,  V >= g."""
    M = V.size - 1
    colors = (  # (node slice, left-neighbour slice, right-neighbour slice, rhs slice)
        (slice(1, M, 2), slice(0, M - 1, 2), slice(2, M + 1, 2), slice(0, M - 1, 2)),
        (slice(2, M, 2), slice(1, M - 1, 2), slice(3, M + 1, 2), slice(1, M - 1, 2)),
    )
    V = np.maximum(V, g)
    for sweep in range(1, max_iter + 1):
        change = 0.0
        for node, left, right, r_sl in colors:
            gs = (rhs[r_sl] - low * V[left] - up * V[right]) / diag
            new = np.maximum(g[node], V[node] + omega * (gs - V[node]))
            change = max(change, float(np.max(np.abs(new - V[node]))))
            V[node] = new

        if change < tol:
            return V, sweep
        
    raise RuntimeError(f"PSOR did not converge in {max_iter} sweeps (last update {change:.2e})")


def penalty_step(V: NDArray[np.float64], rhs: NDArray[np.float64], g: NDArray[np.float64], low: float, diag: float, up: float, dt: float, 
                 rho: float, active: NDArray[np.bool_], max_iter: int) -> tuple[NDArray[np.float64], int, NDArray[np.bool_]]:
    """Penalty method (Forsyth-Vetzal): add dt*rho*(g - V) on the active set, iterate the set to a fixed point."""
    ab = np.zeros((3, rhs.size))
    ab[0, 1:], ab[2, :-1] = up, low
    gi = g[1:-1]
    for n_iter in range(1, max_iter + 1):
        pen = np.where(active, dt * rho, 0.0)
        ab[1] = diag + pen
        interior = solve_banded((1, 1), ab, rhs + pen * gi)
        new_active = interior < gi
        if np.array_equal(new_active, active):
            V = V.copy()
            V[1:-1] = interior
            return V, n_iter, active
        
        active = new_active

    raise RuntimeError(f"penalty iteration did not converge in {max_iter} iterations")


def grid_greeks(x: NDArray[np.float64], V: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Central-difference (S, delta, gamma) at the interior nodes of a uniform log-grid."""
    h = x[1] - x[0]
    S = np.exp(x[1:-1])
    Vx = (V[2:] - V[:-2]) / (2.0 * h)
    Vxx = (V[2:] - 2.0 * V[1:-1] + V[:-2]) / h**2

    return S, Vx / S, (Vxx - Vx) / S**2