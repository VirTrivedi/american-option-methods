"""Synthetic option chains from a known vol surface (American PDE prices), for tests and the pipeline demo."""
from __future__ import annotations
from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd
from ..params import OptionParams
from ..pde import solve_pde
from .rates import ZeroCurve
from .schema import QUOTE_COLUMNS, validate_quotes

GEN_GRID = 800  # chains are priced on a finer PDE grid than the M=N=400 used for inversion, so grid error is part of the test


def vol_surface(strike_over_spot: float, years: float, base: float = 0.25, skew: float = -0.12, smile: float = 0.35, term: float = 0.03) -> float:
    """sigma(K/S, T) = base + skew*x + smile*x^2 + term*sqrt(T), x = ln(K/S)."""
    x = math.log(strike_over_spot)

    return base + skew * x + smile * x * x + term * math.sqrt(years)


def make_chain(spot: float = 100.0, strikes_over_spot=(0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15), expiry_days=(35, 91, 182),
               dividends_abs: tuple[tuple[float, float], ...] = (), curve: ZeroCurve | None = None, rel_spread: float = 0.02,
               noise_rel: float = 0.0, tick: float = 0.0, seed: int = 0, surface: dict | None = None, quote_time: datetime = datetime(2024, 3, 15, 16, 0),
               underlying: str = "SYN") -> pd.DataFrame:
    """Standard quote table with extra columns true_iv, true_mid, true_mid_european."""
    rng = np.random.default_rng(seed)
    curve = curve or ZeroCurve(np.array([30.0, 365.0]), np.array([0.045, 0.05]))
    rows = []
    for days in expiry_days:
        expiry = (quote_time + timedelta(days=days)).replace(hour=0, minute=0)
        T = days / 365.0  # 16:00 to 16:00 on the expiry date
        r = float(curve.rate_at(days))
        divs = tuple(((d - 6.5 / 24.0) / 365.0, a) for d, a in dividends_abs if 0 < d < days)  # ex-date open is 6.5h before the 16:00 clock
        for m in strikes_over_spot:
            K = round(spot * m, 2)
            sigma = vol_surface(K / spot, T, **(surface or {}))
            for cp, kind in (("C", "call"), ("P", "put")):
                p = OptionParams(S=spot, K=K, T=T, r=r, q=0.0, sigma=sigma, option_type=kind, dividends=divs)
                true_mid = solve_pde(p, GEN_GRID, GEN_GRID, american_method="penalty").price
                eur = solve_pde(p, GEN_GRID, GEN_GRID, european=True).price
                mid = true_mid * (1.0 + noise_rel * rng.standard_normal())
                bid, ask = mid * (1 - rel_spread / 2), mid * (1 + rel_spread / 2)
                if tick > 0:
                    bid, ask = round(math.floor(bid / tick) * tick, 8), round(math.ceil(ask / tick) * tick, 8)

                rows.append(dict(quote_time=pd.Timestamp(quote_time), underlying=underlying, expiry=pd.Timestamp(expiry), strike=K, cp_flag=cp, bid=bid, ask=ask,
                                 volume=100.0, open_interest=1000.0, spot=spot, spot_time=pd.Timestamp(quote_time), vendor_iv=sigma, vendor_delta=math.nan,
                                 exercise_style="A", am_settlement=False, true_iv=sigma, true_mid=true_mid, true_mid_european=eur))
                
    df = pd.DataFrame(rows)
    validate_quotes(df[QUOTE_COLUMNS])

    return df


def distributions_table(quote_time: datetime, dividends_abs: tuple[tuple[float, float], ...]) -> pd.DataFrame:
    """Matching distrd-style table (ex_date, amount, is_forecast) for dividends given as (days from quote date, amount)."""
    return pd.DataFrame({"ex_date": [pd.Timestamp(quote_time.date()) + pd.Timedelta(days=d) for d, _ in dividends_abs],
                         "amount": [a for _, a in dividends_abs], "is_forecast": False})