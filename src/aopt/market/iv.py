"""Implied volatility: European Black-Scholes and American (PDE, spot-jump dividends) inversions."""
from __future__ import annotations
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from scipy.optimize import brentq
from ..bs import bs_price
from ..params import OptionParams
from ..pde import solve_pde

SIGMA_LO, SIGMA_HI = 0.01, 3.0
OK = "ok"


@dataclass(frozen=True)
class Contract:
    """Everything an inversion needs except the option price. ``dividends`` are (t, D) with t in ACT/365 years."""
    S: float
    K: float
    T: float
    r: float
    cp_flag: str  # "C" or "P"
    dividends: tuple[tuple[float, float], ...] = ()
    q: float = 0.0

    @property
    def option_type(self) -> str:
        return "call" if self.cp_flag == "C" else "put"

    def params(self, sigma: float) -> OptionParams:
        return OptionParams(S=self.S, K=self.K, T=self.T, r=self.r, q=self.q, sigma=sigma, option_type=self.option_type, dividends=self.dividends)

    def escrowed_spot(self) -> float:
        """Spot less the present value of the dividends: the spot a European Black-Scholes formula should be fed."""
        return self.S - self.params(0.2).dividend_pv(0.0)


def bs_value(c: Contract, sigma: float) -> float:
    """European Black-Scholes on the escrowed spot (identical to plain BS when there are no dividends)."""
    return float(bs_price(c.escrowed_spot(), c.K, c.T, c.r, c.q, sigma, c.option_type))


def bs_vega(c: Contract, sigma: float) -> float:
    """dPrice/dSigma of the escrowed European price (per unit of volatility)."""
    s = c.escrowed_spot()
    d1 = (math.log(s / c.K) + (c.r - c.q + 0.5 * sigma**2) * c.T) / (sigma * math.sqrt(c.T))

    return float(s * math.exp(-c.q * c.T) * math.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi) * math.sqrt(c.T))


def invert(f_value, price: float, intrinsic: float) -> tuple[float, str]:
    """Solve f_value(sigma) = price on [SIGMA_LO, SIGMA_HI]; NaN plus a reason code if no solution."""
    if not math.isfinite(price) or price <= 0.0:
        return math.nan, "nonpositive_price"
    
    if price < intrinsic - 1e-12:
        return math.nan, "price_below_intrinsic"
    
    try:
        lo, hi = f_value(SIGMA_LO), f_value(SIGMA_HI)
    except RuntimeError:
        return math.nan, "solver_failure"
    
    if price <= lo:
        return math.nan, "price_below_min_vol_value"
    
    if price >= hi:
        return math.nan, "price_above_max_vol_value"
    
    try:
        return float(brentq(lambda s: f_value(s) - price, SIGMA_LO, SIGMA_HI, xtol=1e-7, rtol=1e-10)), OK
    except (ValueError, RuntimeError):
        return math.nan, "solver_failure"


def bs_iv(price: float, c: Contract) -> tuple[float, str]:
    """European BS implied vol (escrowed dividends). Reasons: nonpositive_price, price_below_intrinsic (below the European
    lower bound), price_below_min_vol_value, price_above_max_vol_value, solver_failure."""
    s = c.escrowed_spot()
    if s <= 0:
        return math.nan, "dividends_exceed_spot"
    
    fwd = s * math.exp(-c.q * c.T)
    lower = max(fwd - c.K * math.exp(-c.r * c.T), 0.0) if c.cp_flag == "C" else max(c.K * math.exp(-c.r * c.T) - fwd, 0.0)

    return invert(lambda sig: bs_value(c, sig), price, lower)


def american_iv(price: float, c: Contract, M: int = 400, N: int = 400) -> tuple[float, str]:
    """American implied vol from the PDE pricer (penalty method, Rannacher on, spot-jump dividends)."""
    intrinsic = max(c.S - c.K, 0.0) if c.cp_flag == "C" else max(c.K - c.S, 0.0)

    def value(sigma: float) -> float:
        try:
            return solve_pde(c.params(sigma), M, N, american_method="penalty").price
        except RuntimeError:  # the penalty active set can cycle at very low vol (sigma ~ 0.01); PSOR is the fallback
            return solve_pde(c.params(sigma), M, N, american_method="psor").price

    return invert(value, price, intrinsic)


def european_spotjump_iv(price: float, c: Contract, M: int = 400, N: int = 400) -> tuple[float, str]:
    """European implied vol under the spot-jump dividend model (PDE, no early exercise): separates the dividend-model
    difference (BS escrowed vs spot-jump) from genuine early exercise (European vs American spot-jump)."""
    lower = max(c.escrowed_spot() - c.K * math.exp(-c.r * c.T), 0.0) if c.cp_flag == "C" else max(c.K * math.exp(-c.r * c.T) - c.escrowed_spot(), 0.0)

    return invert(lambda sig: solve_pde(c.params(sig), M, N, european=True).price, price, lower)


def task(args):
    kind, price, contract, M, N = args
    if kind == "bs":
        return bs_iv(price, contract)
    
    if kind == "european":
        return european_spotjump_iv(price, contract, M, N)
    
    return american_iv(price, contract, M, N)


def batch_iv(kind: str, prices: list[float], contracts: list[Contract], M: int = 400, N: int = 400, workers: int = 1) -> list[tuple[float, str]]:
    """Invert many contracts; ``kind`` is "bs", "american" or "european" (spot-jump PDE). workers > 1 uses a process pool (results keep input order)."""
    if kind not in ("bs", "american", "european"):
        raise ValueError("kind must be 'bs', 'american' or 'european' (spot-jump, no early exercise)")
    
    tasks = [(kind, p, c, M, N) for p, c in zip(prices, contracts)]
    if workers <= 1 or len(tasks) < 8:
        return [task(t) for t in tasks]
    
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(task, tasks, chunksize=max(1, len(tasks) // (4 * workers))))


def iv_interval(kind: str, bid: float, mid: float, ask: float, c: Contract, M: int = 400, N: int = 400) -> dict[str, float | str]:
    """IV at bid, mid and ask and the half-width of the bid-ask interval (NaN if either end has no solution)."""
    f = (lambda p: bs_iv(p, c)) if kind == "bs" else (lambda p: american_iv(p, c, M, N))
    (lo, r_lo), (md, r_md), (hi, r_hi) = f(bid), f(mid), f(ask)
    half = 0.5 * (hi - lo) if math.isfinite(lo) and math.isfinite(hi) else math.nan

    return dict(iv_bid=lo, iv_mid=md, iv_ask=hi, half_width=half, reason_mid=r_md, reason_bid=r_lo, reason_ask=r_hi)