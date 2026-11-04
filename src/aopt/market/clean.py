"""Quote cleaning with a reject log. Thresholds are fixed in CleanConfig before any plot is looked at."""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
import numpy as np
import pandas as pd
from .dividends import dividends_for_expiry
from .iv import Contract, bs_iv, bs_vega
from .timeutil import expiry_timestamp, year_fraction

# order matters: each row is counted under the first filter it fails
REASONS = ("bid_nonpositive", "crossed_market", "wide_spread", "low_open_interest", "below_intrinsic", "above_upper_bound",
           "maturity_out_of_range", "moneyness_out_of_range", "am_settled", "no_bs_iv", "low_vega")


@dataclass(frozen=True)
class CleanConfig:
    """Stated in advance thresholds (docs/hypotheses_step5.md); the spread threshold is varied only in the sensitivity table."""
    max_rel_spread: float = 0.10  # (ask - bid) / mid
    min_open_interest: float = 10.0
    min_days: float = 7.0
    max_days: float = 365.0
    min_moneyness: float = 0.8  # K / S
    max_moneyness: float = 1.2
    min_vega: float = 0.5  # price change per 1.00 of volatility (= 0.005 per vol point), at the BS IV of the mid
    reject_am_settled: bool = True


def add_model_inputs(q: pd.DataFrame, curve, distributions: pd.DataFrame, div_scale: float = 1.0, r_shift_bp: float = 0.0, spot_shift: float = 0.0) -> pd.DataFrame:
    """Add mid, T (ACT/365), r (continuous, linear in days), the per-expiry dividend tuple and shifted spot S_model."""
    d = q.copy()
    d["mid"] = 0.5 * (d["bid"] + d["ask"])
    ts = [expiry_timestamp(e.date(), am) for e, am in zip(d["expiry"], d["am_settlement"])]
    d["T"] = [year_fraction(qt.to_pydatetime(), t) for qt, t in zip(d["quote_time"], ts)]
    days = (d["expiry"].dt.normalize() - d["quote_time"].dt.normalize()).dt.days.to_numpy(float)
    d["days"] = days
    d["r"] = curve.shifted(r_shift_bp).rate_at(days) if r_shift_bp else curve.rate_at(days)
    d["dividends"] = [dividends_for_expiry(distributions, qt.to_pydatetime(), t, div_scale) for qt, t in zip(d["quote_time"], ts)]
    d["S_model"] = d["spot"] * (1.0 + spot_shift)

    return d


def contract_of(row) -> Contract:
    return Contract(S=float(row["S_model"]), K=float(row["strike"]), T=float(row["T"]), r=float(row["r"]), cp_flag=row["cp_flag"], dividends=row["dividends"])


def clean_quotes(d: pd.DataFrame, cfg: CleanConfig = CleanConfig(), load_rejects: dict[str, int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the filters in order to a table from add_model_inputs. Returns (kept, log) with log columns reason, count, remaining."""
    is_put = (d["cp_flag"] == "P").to_numpy()
    intrinsic = np.where(is_put, np.maximum(d["strike"] - d["S_model"], 0.0), np.maximum(d["S_model"] - d["strike"], 0.0))
    upper = np.where(is_put, d["strike"], d["S_model"])
    rel_spread = (d["ask"] - d["bid"]) / d["mid"].where(d["mid"] > 0)
    moneyness = d["strike"] / d["S_model"]
    tests: dict[str, np.ndarray] = {
        "bid_nonpositive": (d["bid"] <= 0).to_numpy(),
        "crossed_market": (d["ask"] < d["bid"]).to_numpy(),
        "wide_spread": ~(rel_spread <= cfg.max_rel_spread).to_numpy(),
        "low_open_interest": (d["open_interest"] < cfg.min_open_interest).to_numpy(),
        "below_intrinsic": (d["mid"] < intrinsic).to_numpy(),
        "above_upper_bound": (d["mid"] > upper).to_numpy(),
        "maturity_out_of_range": ((d["days"] < cfg.min_days) | (d["days"] > cfg.max_days)).to_numpy(),
        "moneyness_out_of_range": ((moneyness < cfg.min_moneyness) | (moneyness > cfg.max_moneyness)).to_numpy(),
        "am_settled": d["am_settlement"].to_numpy(bool) if cfg.reject_am_settled else np.zeros(len(d), bool),
    }
    alive = np.ones(len(d), dtype=bool)
    log = [dict(reason=k, count=int(v)) for k, v in (load_rejects or {}).items()]
    n_loaded = len(d) + sum((load_rejects or {}).values())
    remaining = len(d)
    log = [dict(**r, remaining=None) for r in log]
    for reason in REASONS[:-2]:
        hit = alive & tests[reason]
        alive &= ~hit
        remaining -= int(hit.sum())
        log.append(dict(reason=reason, count=int(hit.sum()), remaining=remaining))

    kept = d[alive].copy()

    # model-based filters need a BS IV at the mid; computed only for rows that passed everything else
    ivs, vegas = [], []
    for _, row in kept.iterrows():
        c = contract_of(row)
        iv, _ = bs_iv(float(row["mid"]), c)
        ivs.append(iv)
        vegas.append(bs_vega(c, iv) if math.isfinite(iv) else math.nan)

    kept["bs_iv_screen"], kept["vega_screen"] = ivs, vegas
    no_iv = kept["bs_iv_screen"].isna()
    log.append(dict(reason="no_bs_iv", count=int(no_iv.sum()), remaining=remaining - int(no_iv.sum())))
    kept = kept[~no_iv]
    low = kept["vega_screen"] < cfg.min_vega
    log.append(dict(reason="low_vega", count=int(low.sum()), remaining=len(kept) - int(low.sum())))
    kept = kept[~low]  # the original row index is kept so rejected rows can be traced
    out = pd.DataFrame(log)
    out.attrs["n_loaded"] = n_loaded

    return kept, out


def spread_sensitivity(d: pd.DataFrame, thresholds=(0.05, 0.10, 0.20), cfg: CleanConfig = CleanConfig()) -> pd.DataFrame:
    """Contracts surviving the full cleaning at each spread threshold (everything else held fixed)."""
    rows = []
    for t in thresholds:
        kept, _ = clean_quotes(d, replace(cfg, max_rel_spread=t))
        rows.append(dict(max_rel_spread=t, surviving=len(kept), puts=int((kept["cp_flag"] == "P").sum()), calls=int((kept["cp_flag"] == "C").sum())))

    return pd.DataFrame(rows)