"""Risk-free rates from the OptionMetrics zero curve."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import pandas as pd

RATE_UNITS = "percent"  # zerocd.rate is a continuously compounded annual rate in percent


@dataclass(frozen=True)
class ZeroCurve:
    """Zero curve: calendar days to maturity and continuously compounded rates (decimal)."""
    days: npt.NDArray[np.float64]
    rate: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.days.size < 2 or np.any(np.diff(self.days) <= 0):
            raise ValueError("zero curve needs >= 2 strictly increasing maturities")

    def rate_at(self, days: float | npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Linear interpolation in days; flat beyond the ends (the curve covers 1 day to years, so ends are not hit in practice)."""
        return np.interp(np.asarray(days, dtype=float), self.days, self.rate)

    def shifted(self, bp: float) -> ZeroCurve:
        """Curve with a parallel shift in basis points (sensitivity analysis)."""
        return ZeroCurve(self.days, self.rate + bp * 1e-4)


def curve_from_zerocd(df: pd.DataFrame) -> ZeroCurve:
    """Build a curve from a one-date zerocd table, asserting units."""
    for col in ("days", "rate"):
        if col not in df.columns:
            raise ValueError(f"zerocd table is missing column {col!r}")
    
    if "date" in df.columns and df["date"].nunique() != 1:
        raise ValueError("zerocd table must contain exactly one date")
    
    d = df.sort_values("days")
    days, rate = d["days"].to_numpy(float), d["rate"].to_numpy(float)
    if not (np.isfinite(days).all() and np.isfinite(rate).all()) or (days <= 0).any():
        raise ValueError("zerocd contains non-finite values or non-positive days")
    
    if rate.max() > 30.0 or rate.min() < -2.0:
        raise ValueError(f"zerocd rates {rate.min()}..{rate.max()} are not plausible percent values")
    
    if rate.max() < 0.1:
        raise ValueError(f"zerocd rates (max {rate.max()}) look like decimals, but {RATE_UNITS} is assumed; check the units")
    
    if np.any(np.diff(days) <= 0):
        raise ValueError("zerocd has duplicate maturities for the snapshot date")

    return ZeroCurve(days, rate / 100.0)