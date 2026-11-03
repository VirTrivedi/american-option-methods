"""Load OptionMetrics Ivy DB CSV extracts (written by scripts/pull_wrds.py) into the standard quote schema."""
from __future__ import annotations
from datetime import datetime, time
from pathlib import Path
import numpy as np
import pandas as pd
from .schema import QUOTE_COLUMNS, validate_quotes

STRIKE_SCALE = 1000.0  # opprcd.strike_price is stored as dollars * 1000
MISSING = -99.99  # OptionMetrics sentinel for a missing impl_volatility / delta
SNAPSHOT_TIME = time(16, 0)  # end-of-day prices: quote and spot are both as of the 16:00 ET close
OPPRCD_COLUMNS = ["secid", "date", "exdate", "cp_flag", "strike_price", "best_bid", "best_offer", "volume", "open_interest", "impl_volatility",
                  "delta", "ss_flag", "am_settlement", "contract_size", "cfadj"]  # exercise_style comes from opinfd (per security)
LOAD_REJECT_REASONS = ("nonstandard_settlement", "adjusted_contract", "nonstandard_contract_size")


def latest_raw(directory: Path, kind: str, ticker: str, date: str) -> Path:
    """Most recent pull for (kind, ticker, date): files are named {kind}_{ticker}_{date}_{pull timestamp}.csv."""
    matches = sorted(directory.glob(f"{kind}_{ticker}_{date}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"no {kind}_{ticker}_{date}_*.csv in {directory}; run scripts/pull_wrds.py first")

    return matches[-1]


def flag(series: pd.Series) -> pd.Series:
    """Normalise a 0/1 flag stored as int, float or string to strings '0'/'1'."""
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def load_option_prices(df: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise an opprcd extract (strikes in dollars, IV/delta NaN when missing)."""
    missing = [c for c in OPPRCD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"opprcd extract is missing columns {missing}")
    
    d = df.copy()
    d["date"], d["exdate"] = pd.to_datetime(d["date"]), pd.to_datetime(d["exdate"])
    if d["date"].nunique() != 1:
        raise ValueError("opprcd extract must contain exactly one snapshot date")
    
    if not set(d["cp_flag"].astype(str).str.strip()) <= {"C", "P"}:
        raise ValueError(f"cp_flag values {set(d['cp_flag'])} are not C/P")
    
    d["cp_flag"] = d["cp_flag"].astype(str).str.strip()
    if "exercise_style" not in info.columns:
        raise ValueError(f"opinfd extract is missing exercise_style; columns: {list(info.columns)}")
    
    styles = set(info.loc[info["secid"].isin(d["secid"].unique()), "exercise_style"].astype(str).str.strip())
    if len(styles) != 1 or not styles <= {"A", "E"}:
        raise ValueError(f"opinfd exercise_style for this secid is {styles}; expected exactly one of A/E")
    
    d["exercise_style"] = styles.pop()
    d["ss_flag"], am = flag(d["ss_flag"]), flag(d["am_settlement"])
    if not set(d["ss_flag"]) <= {"0", "1"} or not set(am) <= {"0", "1"}:
        raise ValueError(f"ss_flag {set(d['ss_flag'])} / am_settlement {set(am)} are not 0/1 flags")
    
    d["am_settlement"] = am == "1"
    d["strike"] = d["strike_price"].astype(float) / STRIKE_SCALE
    if (d["strike_price"].astype(float) % 1 != 0).any():
        raise ValueError("strike_price is not integer-valued; the x1000 scaling assumption does not hold")
    
    for col in ("best_bid", "best_offer", "open_interest", "volume"):
        if not np.isfinite(d[col].astype(float)).all() or (d[col] < 0).any():
            raise ValueError(f"{col} has NaN or negative values")
        
    for col in ("impl_volatility", "delta"):
        d[col] = d[col].astype(float).replace(MISSING, np.nan)

    if ((d["impl_volatility"] <= 0) | (d["impl_volatility"] > 10)).any():
        raise ValueError("impl_volatility outside (0, 10] after removing the -99.99 sentinel")
    
    if (d["delta"].abs() > 1.0 + 1e-9).any():
        raise ValueError("|delta| > 1 after removing the sentinel")
    
    if (d["cfadj"].astype(float) <= 0).any() or (d["contract_size"].astype(float) <= 0).any():
        raise ValueError("cfadj and contract_size must be positive")

    return d


def underlying_close(df: pd.DataFrame, secid: int, date: pd.Timestamp) -> float:
    """Close of ``secid`` on ``date`` from a secprd extract. A negative close means no trade (bid/ask average): refuse it."""
    for col in ("secid", "date", "close"):
        if col not in df.columns:
            raise ValueError(f"secprd extract is missing column {col!r}")
        
    row = df[(df["secid"] == secid) & (pd.to_datetime(df["date"]) == date)]
    if len(row) != 1:
        raise ValueError(f"expected exactly one secprd row for secid {secid} on {date.date()}, found {len(row)}")
    
    close = float(row["close"].iloc[0])
    if not close > 0:
        raise ValueError(f"secprd close {close} is not positive (negative = bid/ask average, not a traded close); choose another date")

    return close


def build_quotes(prices: pd.DataFrame, spot: float, underlying: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Standard quote table from a validated opprcd extract; rows failing the schema's own flags are dropped and counted (sequentially)."""
    d = prices
    rejects: dict[str, int] = {}
    for reason in LOAD_REJECT_REASONS:
        bad = {"nonstandard_settlement": lambda x: x["ss_flag"] != "0", "adjusted_contract": lambda x: ~np.isclose(x["cfadj"].astype(float), 1.0),
               "nonstandard_contract_size": lambda x: x["contract_size"].astype(float) != 100.0}[reason](d)
        rejects[reason] = int(bad.sum())
        d = d[~bad]

    ratio = d["strike"] / spot
    if len(d) and not 0.2 < float(ratio.median()) < 5.0:
        raise ValueError(f"median strike/spot = {ratio.median():.3f}: the x1000 strike scaling or the spot is wrong")
    
    snap = pd.Timestamp(datetime.combine(d["date"].iloc[0].date(), SNAPSHOT_TIME)) if len(d) else None
    q = pd.DataFrame({"quote_time": snap, "underlying": underlying, "expiry": d["exdate"].to_numpy(), "strike": d["strike"].to_numpy(),
                      "cp_flag": d["cp_flag"].to_numpy(), "bid": d["best_bid"].astype(float).to_numpy(), "ask": d["best_offer"].astype(float).to_numpy(),
                      "volume": d["volume"].astype(float).to_numpy(), "open_interest": d["open_interest"].astype(float).to_numpy(), "spot": spot,
                      "spot_time": snap, "vendor_iv": d["impl_volatility"].to_numpy(), "vendor_delta": d["delta"].to_numpy(),
                      "exercise_style": d["exercise_style"].to_numpy(), "am_settlement": d["am_settlement"].to_numpy(dtype=bool)})
    validate_quotes(q[QUOTE_COLUMNS])

    return q[QUOTE_COLUMNS].reset_index(drop=True), rejects