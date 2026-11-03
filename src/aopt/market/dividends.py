"""Map OptionMetrics distributions (optionm.distrd) to the pricers' ``dividends`` field."""
from __future__ import annotations
from collections.abc import Collection
from datetime import datetime
import pandas as pd
from .timeutil import EX_DATE_TIME, year_fraction


class UnsupportedDistribution(Exception):
    """Raised when distrd lists a distribution that is not a plain cash dividend."""


def load_distributions(df: pd.DataFrame, cash_types: Collection[str] | None) -> pd.DataFrame:
    """Validate a distrd extract and return columns (ex_date, amount, is_forecast)."""
    if df.empty:
        return pd.DataFrame({"ex_date": pd.Series(dtype="datetime64[ns]"), "amount": pd.Series(dtype=float), "is_forecast": pd.Series(dtype=bool)})
    
    for col in ("ex_date", "amount", "distr_type"):
        if col not in df.columns:
            raise ValueError(f"distrd extract is missing column {col!r}; columns present: {list(df.columns)}")
        
    types = df["distr_type"].astype(str).str.strip()
    allowed = {str(t) for t in cash_types} if cash_types is not None else set()
    bad = df[~types.isin(allowed)]
    if not bad.empty:
        raise UnsupportedDistribution(
            f"{len(bad)} distribution(s) with distr_type not confirmed as cash: {sorted(set(types[bad.index]))}; rows:\n{bad.to_string()}\n"
            "Confirm which distr_type codes are ordinary cash dividends (or exclude this ticker/window) before continuing.")
    
    for flag in ("cancel_flag", "approx_flag"):  # a cancelled or approximate (forecast) distribution must not be priced as an actual dividend
        if flag in df.columns and (df[flag].astype(str).str.replace(r"\.0$", "", regex=True) != "0").any():
            raise UnsupportedDistribution(f"distrd rows with {flag} != 0:\n{df[df[flag].astype(str).str.replace(r'\.0$', '', regex=True) != '0'].to_string()}\nDecide how to treat them before continuing.")
        
    amount = df["amount"].astype(float)
    if not (amount > 0).all():
        raise ValueError("distrd contains non-positive amounts")

    return pd.DataFrame({"ex_date": pd.to_datetime(df["ex_date"]), "amount": amount.to_numpy(), "is_forecast": False}).sort_values("ex_date").reset_index(drop=True)


def dividends_for_expiry(table: pd.DataFrame, quote_time: datetime, expiry_ts: datetime, scale: float = 1.0) -> tuple[tuple[float, float], ...]:
    """((t, D), ...) with t the ACT/365 time of the ex-date open and quote_time < ex < expiry. ``scale`` multiplies the amounts."""
    out = []
    for ex_date, amount in zip(table["ex_date"], table["amount"]):
        ex_ts = datetime.combine(ex_date.date(), EX_DATE_TIME)
        if quote_time < ex_ts < expiry_ts:
            out.append((year_fraction(quote_time, ex_ts), float(amount) * scale))

    return tuple(out)