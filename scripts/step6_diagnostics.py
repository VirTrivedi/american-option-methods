"""Step 6: Exploratory (Post-Hoc) diagnostics of the Step 5 results."""
from __future__ import annotations
import math
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle
import matplotlib.pyplot as plt
import step5_market as m
from aopt.market.clean import add_model_inputs, clean_quotes, contract_of
from aopt.market.iv import Contract, batch_iv, bs_vega
from aopt.market.loader import latest_raw
from aopt.market.timeutil import EX_DATE_TIME, year_fraction
from aopt.pde import solve_pde

STATUS = "exploratory (post-hoc)"
ROOT, TAB, FIG, DERIVED = m.ROOT, m.ROOT / "results" / "tables", m.ROOT / "results" / "figures", m.DERIVED_DIR
GRIDS = (400, 800, 1600)
SMALL_SPREAD = 0.05  # parity regression
PARITY_MNY = (0.95, 1.05)
WINGS = [("K/S < 0.93", -np.inf, 0.93), ("0.93 <= K/S <= 1.03", 0.93, 1.03), ("K/S > 1.03", 1.03, np.inf)]
OFFSET_MNY_MAX = 0.9
STAGES = ("discretization", "grid", "level", "split", "parity", "bound", "dollar", "wings", "offset")
VP = 100.0  # IV decimals -> vol points
IV_COLS = [f"{k}{M}" for M in GRIDS for k in ("am", "eu")] + ["am_bid", "am_ask", "bs", "half"]


def wing_of(x: float) -> str:
    return WINGS[0][0] if x < 0.93 else (WINGS[2][0] if x > 1.03 else WINGS[1][0])


def save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.copy()
    df["status"] = STATUS
    df.to_csv(TAB / f"diag_{name}.csv", index=False)

    return df


def solve(kept: pd.DataFrame, kind: str, price_col: str, M: int, workers: int) -> np.ndarray:
    contracts = [contract_of(r) for _, r in kept.iterrows()]
    prices = [float(p) for p in kept[price_col]]

    return np.array([x[0] for x in batch_iv(kind, prices, contracts, M, M, workers)])


def load(workers: int, force: bool = False) -> tuple[dict[str, tuple[pd.DataFrame, pd.Timestamp | None]], object, dict[str, str]]:
    """Cleaned contracts and their IVs on every grid. The solved table is cached in data/derived/diag_ivs_<name>.csv (per-contract, never committed)."""
    kv = m.parse_hypotheses(m.HYPOTHESES)
    unders, curve, _ = m.load_real(kv)
    DERIVED.mkdir(parents=True, exist_ok=True)
    out = {}
    for u in unders:
        kept, _ = clean_quotes(add_model_inputs(u.quotes, curve, u.distributions), m.CFG, u.load_rejects)
        ex_date = pd.Timestamp(u.distributions["ex_date"].iloc[0]) if len(u.distributions) else None
        cache = DERIVED / f"diag_ivs_{u.name}.csv"
        d = kept.copy()
        d["moneyness"] = d["strike"] / d["S_model"]
        d["side"] = "no dividend" if ex_date is None else np.where(d["expiry"] > ex_date, "after ex-date", "before ex-date")
        if cache.exists() and not force:
            iv = pd.read_csv(cache)
            assert len(iv) == len(d), "stale cache: rerun with --force"
            for c in IV_COLS:
                d[c] = iv[c].to_numpy()
        else:
            t0 = time.perf_counter()
            for M in GRIDS:
                d[f"am{M}"] = solve(d, "american", "mid", M, workers)
                d[f"eu{M}"] = solve(d, "european", "mid", M, workers) if ex_date is not None else np.nan
            d["am_bid"], d["am_ask"] = solve(d, "american", "bid", 400, workers), solve(d, "american", "ask", 400, workers)
            d["bs"] = solve(d, "bs", "mid", 400, workers)
            d["half"] = 0.5 * (d["am_ask"] - d["am_bid"])
            d[IV_COLS].to_csv(cache, index=False)
            print(f"{u.name}: solved IVs on grids {GRIDS} in {time.perf_counter() - t0:.0f}s")

        d["headline"] = d["bs"].notna() & d["am400"].notna() & d["half"].notna()  # the contracts used in the headline statistics (both IVs and the interval exist)
        d = m.buckets(d)
        out[u.name] = (d, ex_date)

    return out, curve, kv


def discretization(data: dict) -> None:
    rows, rows_e = [], []
    for name, (d, ex_date) in data.items():
        h = d[d.headline & d["am800"].notna() & d["am1600"].notna()].copy()
        e48, e81, e41 = (h["am400"] - h["am800"]).abs(), (h["am800"] - h["am1600"]).abs(), (h["am400"] - h["am1600"]).abs()
        order_all = math.log2(e48.median() / e81.median())
        pair = (e48 > 1e-6) & (e81 > 1e-6)
        per_contract = np.log2(e48[pair] / e81[pair])
        p = order_all if 1.0 <= order_all <= 3.0 else 2.0
        limit = h["am1600"] + (h["am1600"] - h["am800"]) / (2 ** p - 1)  # Richardson limit at the observed order
        err400 = (h["am400"] - limit).abs()
        rows.append(dict(underlying=name, n=len(h), n_headline=int(d.headline.sum()), max_abs_iv_1600_800_vp=VP * e81.max(), median_abs_iv_1600_800_vp=VP * e81.median(),
                         max_abs_iv_800_400_vp=VP * e48.max(), median_abs_iv_800_400_vp=VP * e48.median(), max_abs_iv_1600_400_vp=VP * e41.max(), median_abs_iv_1600_400_vp=VP * e41.median(),
                         empirical_order_from_medians=order_all, median_per_contract_order=float(per_contract.median()), n_per_contract_order=int(pair.sum()),
                         median_half_width_vp=VP * h["half"].median(), p05_half_width_vp=VP * h["half"].quantile(0.05),
                         ratio_max_1600_800_to_median_half_width=e81.max() / h["half"].median(), ratio_max_1600_400_to_median_half_width=e41.max() / h["half"].median(),
                         frac_1600_800_above_own_half_width=float((e81 > h["half"]).mean()), frac_800_400_above_own_half_width=float((e48 > h["half"]).mean()),
                         frac_1600_400_above_own_half_width=float((e41 > h["half"]).mean()), richardson_order_used=p, max_abs_iv_400_vs_richardson_vp=VP * err400.max(),
                         median_abs_iv_400_vs_richardson_vp=VP * err400.median(), frac_400_vs_richardson_above_own_half_width=float((err400 > h["half"]).mean())))
        h = h.assign(e81=e81, e41=e41, above=(e81 > h["half"]), expiry_s=h["expiry"].dt.date.astype(str))
        for key, g in h.groupby("expiry_s"):
            rows_e.append(dict(underlying=name, expiry=key, side=g["side"].iloc[0], n=len(g), max_abs_iv_1600_800_vp=VP * g.e81.max(), median_abs_iv_1600_800_vp=VP * g.e81.median(),
                               median_half_width_vp=VP * g["half"].median(), frac_1600_800_above_own_half_width=float(g.above.mean())))

    pd.set_option("display.width", 250, "display.max_columns", 40)
    t = save(pd.DataFrame(rows), "discretization")
    save(pd.DataFrame(rows_e), "discretization_by_expiry")
    print("DISCRETIZATION (vol points):\n" + t.drop(columns="status").T.to_string())


def level(data: dict) -> None:
    """Median American mid IV (M=N=400) at K/S 0.98-1.02, by underlying and expiry: the vol level of the snapshot. Context for the limitations; no comparison with other dates is made."""
    rows = []
    for name, (d, _) in data.items():
        a = d[d.headline & (d.moneyness >= 0.98) & (d.moneyness <= 1.02)]
        rows.append(dict(underlying=name, expiry="all", days="all", n=len(a), median_atm_iv_vp=VP * a["am400"].median()))
        for e, g in a.groupby(a["expiry"].dt.date.astype(str)):
            rows.append(dict(underlying=name, expiry=e, days=int(g["days"].iloc[0]), n=len(g), median_atm_iv_vp=VP * g["am400"].median()))

    tab = save(pd.DataFrame(rows), "snapshot_level")
    print("\nSNAPSHOT VOL LEVEL (median ATM American IV, vol points):\n" + tab.drop(columns="status").round(2).to_string(index=False))


def split(data: dict) -> None:
    d, ex_date = data["SPY"]
    h = d[d.headline & d["am800"].notna() & d["am1600"].notna() & d["eu400"].notna() & d["eu1600"].notna() & d["eu800"].notna()]
    for M in GRIDS:
        h = h.assign(**{f"div_model_{M}": VP * (h["bs"] - h[f"eu{M}"]), f"early_exercise_{M}": VP * (h[f"eu{M}"] - h[f"am{M}"]), f"total_{M}": VP * (h["bs"] - h[f"am{M}"])})

    cols = [f"{k}_{M}" for M in GRIDS for k in ("div_model", "early_exercise", "total")]
    tab = h.groupby(["side", "cp_flag", "mny_bucket"], observed=True).agg(n=("total_400", "size"), **{f"median_{c}_vp": (c, "median") for c in cols}).reset_index()
    tab["cp_flag"] = tab["cp_flag"].map({"P": "put", "C": "call"})
    tab = save(tab, "dividend_split")
    assert np.allclose(h["total_1600"], h["div_model_1600"] + h["early_exercise_1600"]), "decomposition must be exact per contract"
    print("\nDIVIDEND-MODEL SPLIT, SPY (median vol points; total = div_model + early_exercise per contract):\n"
          + tab[["side", "cp_flag", "mny_bucket", "n", "median_div_model_400_vp", "median_early_exercise_400_vp", "median_total_400_vp", "median_div_model_1600_vp", "median_early_exercise_1600_vp", "median_total_1600_vp"]].round(3).to_string(index=False))


def ols(K: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Slope, intercept and the 2x2 covariance matrix [[var b, cov], [cov, var a]] for y = a + b K."""
    X = np.column_stack([K, np.ones_like(K)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = float(resid @ resid) / (len(y) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)

    return float(beta[0]), float(beta[1]), cov


def pair_table(data: dict) -> pd.DataFrame:
    """One row per call-put pair (same underlying, expiry, strike) with the model early-exercise premium of each leg, computed at the IV of the out-of-the-money leg
    (American PDE at M=N=400 minus European PDE on the same spot-jump dividend model). Used by the parity regression and the dollar-space gap."""
    rows = []
    for name, (dd, ex) in data.items():
        c, p = dd[dd.cp_flag == "C"], dd[dd.cp_flag == "P"]
        pr = c.merge(p, on=["expiry", "strike"], suffixes=("_c", "_p"))
        for _, r in pr.iterrows():
            S, K, T, rate, divs = r.S_model_c, r.strike, r.T_c, r.r_c, r.dividends_c
            sigma = r.am400_p if K / S < 1.0 else r.am400_c  # IV of the out-of-the-money leg
            cc, pc = Contract(S, K, T, rate, "C", divs), Contract(S, K, T, rate, "P", divs)
            prem_c = prem_p = vega_c = math.nan
            if math.isfinite(sigma):
                prem = lambda con: solve_pde(con.params(sigma), 400, 400, american_method="penalty").price - solve_pde(con.params(sigma), 400, 400, european=True).price
                prem_c, prem_p, vega_c = prem(cc), prem(pc), bs_vega(cc, sigma)

            rows.append(dict(underlying=name, expiry=r.expiry, strike=K, S=S, T=T, r=rate, days=int(r.days_c), moneyness=K / S, wing=wing_of(K / S),
                             side="no dividend" if ex is None else ("after ex-date" if r.expiry > ex else "before ex-date"), pv_dividends=cc.params(0.2).dividend_pv(0.0),
                             sigma=sigma, prem_c=prem_c, prem_p=prem_p, vega_c=vega_c, mid_c=r.mid_c, mid_p=r.mid_p,
                             band=0.5 * ((r.ask_p - r.bid_p) + (r.ask_c - r.bid_c)), rel_c=(r.ask_c - r.bid_c) / r.mid_c, rel_p=(r.ask_p - r.bid_p) / r.mid_p))

    pt = pd.DataFrame(rows)
    pt["parity_eur"] = pt["S"] - pt["pv_dividends"] - pt["strike"] * np.exp(-pt["r"] * pt["T"])

    return pt


PARITY_VARIANTS = (("registered set (K/S 0.95-1.05, raw)", PARITY_MNY, "raw"),
                   ("model premium negligible (both legs < $0.01, K/S 0.85-1.05)", (0.85, 1.05), "negligible"),
                   ("model premium subtracted (K/S 0.85-1.05)", (0.85, 1.05), "subtract"))
NEGLIGIBLE = 0.01  # dollars


def parity(data: dict, date: str, pt: pd.DataFrame) -> None:
    """Per-expiry OLS of (C_mid - P_mid) on K. The raw regression mixes American puts, whose early-exercise value grows with K, into a European identity; the two
    other variants remove that (restrict to strikes where the model premium is negligible, or subtract the model premium of both legs)."""
    rows = []
    for name in data:
        raw = pd.read_csv(latest_raw(m.RAW_DIR, "opprcd", name, date), parse_dates=["exdate"])
        base = pt[(pt.underlying == name) & (pt.rel_c <= SMALL_SPREAD) & (pt.rel_p <= SMALL_SPREAD)]
        for label, (lo, hi), kind in PARITY_VARIANTS:
            pr = base[(base.moneyness >= lo) & (base.moneyness <= hi)]
            if kind != "raw":
                pr = pr[pr.prem_c.notna() & pr.prem_p.notna()]

            if kind == "negligible":
                pr = pr[(pr.prem_c.abs() < NEGLIGIBLE) & (pr.prem_p.abs() < NEGLIGIBLE)]

            for expiry, g in pr.groupby("expiry"):
                if len(g) < 3:
                    continue

                S, T, r, days, pv = float(g.S.iloc[0]), float(g["T"].iloc[0]), float(g.r.iloc[0]), int(g.days.iloc[0]), float(g.pv_dividends.iloc[0])
                y, K = (g.mid_c - g.mid_p).to_numpy(float), g.strike.to_numpy(float)
                if kind == "subtract":
                    y = y - (g.prem_c - g.prem_p).to_numpy(float)

                b, a, cov = ols(K, y)
                se_b, se_a = math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1])
                df_impl = -b
                rate_impl = -math.log(df_impl) / T if df_impl > 0 else math.nan
                se_rate = se_b / df_impl / T if df_impl > 0 else math.nan
                fwd_impl = a / df_impl if df_impl > 0 else math.nan  # forward-type intercept: escrowed spot compounded at the implied rate
                jac = np.array([a / b ** 2, -1.0 / b]) if df_impl > 0 else np.array([math.nan, math.nan])  # d(a / -b) / d(b, a); delta method with the full covariance
                se_fwd = float(math.sqrt(jac @ cov @ jac)) if df_impl > 0 else math.nan
                a0 = y + K * math.exp(-r * T)  # slope fixed at the model discount factor
                vend = raw.loc[raw.exdate == expiry, "forward_price"]
                rows.append(dict(variant=label, underlying=name, expiry=str(expiry.date()), days=days, n_strikes=len(g), spot_close=S, pv_dividends=pv, model_escrowed_spot=S - pv,
                                 free_intercept=a, se_free_intercept=se_a, free_slope=b, se_free_slope=se_b, implied_discount_factor=df_impl, model_discount_factor=math.exp(-r * T),
                                 implied_rate=rate_impl, se_implied_rate=se_rate, model_rate=r, implied_minus_model_rate_bp=1e4 * (rate_impl - r),
                                 implied_forward=fwd_impl, se_implied_forward=se_fwd, model_forward=(S - pv) * math.exp(r * T), close_forward=S * math.exp(r * T),
                                 vendor_forward_nonnull=int(vend.notna().sum()), vendor_forward=float(vend.median()) if vend.notna().any() else math.nan,
                                 fixed_slope_intercept=float(a0.mean()), se_fixed_slope_intercept=float(a0.std(ddof=1) / math.sqrt(len(a0))),
                                 offset_vs_close_dollars=float(a0.mean()) - S, offset_vs_model_escrowed_spot_dollars=float(a0.mean()) - (S - pv),
                                 offset_pct_of_spot=100 * (float(a0.mean()) - (S - pv)) / S, free_slope_below_minus_one=bool(b < -1.0), z_slope_vs_minus_one=(b + 1.0) / se_b,
                                 mean_abs_model_premium_in_fit=float(np.nanmean(np.abs(g.prem_c) + np.abs(g.prem_p))) if g.prem_c.notna().any() else math.nan))

    tab = save(pd.DataFrame(rows), "parity")
    print("\nPARITY REGRESSION (C_mid - P_mid on K; both legs spread <= 5%; three variants):\n"
          + tab[["variant", "underlying", "expiry", "n_strikes", "free_slope", "se_free_slope", "z_slope_vs_minus_one", "implied_rate", "se_implied_rate", "model_rate", "free_intercept", "se_free_intercept",
                 "model_escrowed_spot", "vendor_forward_nonnull", "fixed_slope_intercept", "se_fixed_slope_intercept", "offset_vs_model_escrowed_spot_dollars"]].round(4).to_string(index=False))


def bound(data: dict, curve) -> None:
    d, ex_date = data["SPY"]
    qt = d["quote_time"].iloc[0].to_pydatetime()
    t_d = year_fraction(qt, datetime.combine(ex_date.date(), EX_DATE_TIME))
    r_d = float(curve.rate_at(t_d * 365.0))
    calls = d[(d.cp_flag == "C") & (d.expiry > ex_date)].copy()
    calls["bound"] = calls["S_model"] - calls["strike"] * math.exp(-r_d * t_d)
    calls["ask_minus_bound"], calls["bid_minus_bound"] = calls["ask"] - calls["bound"], calls["bid"] - calls["bound"]
    calls["expiry_s"] = calls["expiry"].dt.date.astype(str)
    calls[["expiry_s", "strike", "moneyness", "bid", "ask", "bound", "ask_minus_bound", "bid_minus_bound"]].to_csv(DERIVED / "diag_model_free_bound_contracts.csv", index=False)
    rows = []
    for keys, g in [(("all", "all"), calls)] + [(("expiry", k), x) for k, x in calls.groupby("expiry_s")] + [(("moneyness", str(k)), x) for k, x in calls.groupby("mny_bucket", observed=True)]:
        rows.append(dict(by=keys[0], group=keys[1], n_calls=len(g), n_ask_below_bound=int((g.ask_minus_bound < 0).sum()), n_bid_below_bound=int((g.bid_minus_bound < 0).sum()),
                         min_ask_minus_bound=g.ask_minus_bound.min(), median_ask_minus_bound=g.ask_minus_bound.median(), t_d=t_d, r_td=r_d))

    for (e, b), g in calls.groupby(["expiry_s", "mny_bucket"], observed=True):  # expiry x moneyness counts
        rows.append(dict(by="expiry x moneyness", group=f"{e} {b}", n_calls=len(g), n_ask_below_bound=int((g.ask_minus_bound < 0).sum()), n_bid_below_bound=int((g.bid_minus_bound < 0).sum()),
                         min_ask_minus_bound=g.ask_minus_bound.min(), median_ask_minus_bound=g.ask_minus_bound.median(), t_d=t_d, r_td=r_d))

    tab = save(pd.DataFrame(rows), "model_free_bound")
    print(f"\nMODEL-FREE BOUND S - K exp(-r t_d) for SPY calls after the ex-date (t_d = {t_d:.4f} y, r = {r_d:.4f}):\n" + tab[tab.by.isin(["all", "expiry", "moneyness"])].drop(columns=["status", "t_d", "r_td"]).round(3).to_string(index=False))


def dollar_gap(pt: pd.DataFrame, data: dict) -> None:
    """Call-put gap in dollars: market (C - P) minus the American model's (C - P) at the IV of the out-of-the-money leg, set against the two quoted half-spreads."""
    dol = pt.dropna(subset=["sigma"]).copy()
    dol["market_deviation"], dol["model_deviation"] = (dol.mid_c - dol.mid_p) - dol.parity_eur, dol.prem_c - dol.prem_p
    dol["market_minus_model"] = dol["market_deviation"] - dol["model_deviation"]
    dol["implied_call_iv_shift_vp"] = VP * dol["market_minus_model"] / dol["vega_c"]  # the call-price gap, if it were all in the call, expressed in IV
    dol.drop(columns=["S", "T", "r"]).to_csv(DERIVED / "diag_dollar_gap_contracts.csv", index=False)
    agg = lambda g: pd.Series(dict(n=len(g), mean_market_deviation=g.market_deviation.mean(), mean_model_deviation=g.model_deviation.mean(), mean_market_minus_model=g.market_minus_model.mean(),
                                   se_market_minus_model=g.market_minus_model.std(ddof=1) / math.sqrt(len(g)) if len(g) > 1 else math.nan, median_market_minus_model=g.market_minus_model.median(),
                                   mean_abs_market_minus_model=g.market_minus_model.abs().mean(), mean_quoted_half_spread_sum=g.band.mean(), median_abs_gap_over_half_spread_sum=(g.market_minus_model.abs() / g.band).median(),
                                   frac_gap_within_half_spread_sum=float((g.market_minus_model.abs() <= g.band).mean()), median_call_vega=g.vega_c.median(),
                                   median_implied_call_iv_shift_vp=g.implied_call_iv_shift_vp.median(), median_abs_implied_call_iv_shift_vp=g.implied_call_iv_shift_vp.abs().median()))
    by_wing = dol.groupby(["underlying", "side", "wing"]).apply(agg, include_groups=False).reset_index()
    by_exp = dol[dol.wing == "K/S < 0.93"].groupby(["underlying", "side", "expiry", "wing"]).apply(agg, include_groups=False).reset_index()
    by_exp["expiry"] = by_exp["expiry"].dt.date.astype(str)
    tab = save(pd.concat([by_wing.assign(expiry="all"), by_exp]), "dollar_gap")
    print("\nCALL-PUT GAP IN DOLLARS vs the American model, against the quoted half-spreads (deep ITM calls = K/S < 0.93):\n"
          + tab.drop(columns="status").round(3).to_string(index=False))


def grid_checks(data: dict) -> None:
    """The registered Step 5 checks (the rows of step5_hypothesis_checks.csv) recomputed with the American IV at M=N=1600 and with the Richardson IV 1600 + (1600 - 800) / 3.
    The registered 400-grid verdicts are untouched; this adds columns. Half-widths stay the registered M=N=400 ones (bid/ask IVs were solved only there)."""
    reg = pd.read_csv(TAB / "step5_hypothesis_checks.csv").set_index("quantity")
    rows = []
    for H, name, quantity, threshold, test in [
            ("H1", "ATM", "median put diff, ATM (K/S 0.98-1.02)", "> 0.006", lambda v: v > 0.006),
            ("H1", "OTM", "median put diff, K/S <= 0.95", "> 0.003", lambda v: v > 0.003),
            ("H1", "ITM", "fraction of ITM puts (K/S >= 1.05, n={n}) with diff > half-width", "not above half-width at all (0)", lambda v: v == 0),
            ("H2", "H2", "max |BS - American| for calls 0.9<=K/S<=1.1 (n={n}); expected ~0 only if q=0", "> 0.001 (0.1 vol points)", lambda v: v > 1e-3),
            ("H3", "H3neg", "strikes with market premium negative and entirely below zero by the band (of {n})", "any far outside the band", lambda v: v > 0),
            ("H3", "H3mean", "mean market premium", "negative", lambda v: v < 0),
            ("H4", "H4", "mean |call-put IV gap|: American minus BS (negative = improvement)", ">= 0 (no improvement)", lambda v: v >= 0)]:
        for u in ("AMZN", "SPY"):
            if (name.startswith("H3") or name == "H2") and u != "AMZN" or name == "H4" and u != "SPY":
                continue

            d = data[u][0]
            h = d[d.headline & d["am800"].notna() & d["am1600"].notna()].copy()
            h["amR"] = h["am1600"] + (h["am1600"] - h["am800"]) / 3.0
            vals, n_used = {}, None
            for col in ("am400", "am1600", "amR"):
                diff = h["bs"] - h[col]
                puts, calls = h[h.cp_flag == "P"], h[h.cp_flag == "C"]
                if name == "ATM":
                    v = diff[puts.index][(puts.moneyness >= 0.98) & (puts.moneyness <= 1.02)].median()
                elif name == "OTM":
                    v = diff[puts.index][puts.moneyness <= 0.95].median()
                elif name == "ITM":
                    itm = puts[puts.moneyness >= 1.05]
                    v, n_used = float((diff[itm.index].abs() > itm["half"]).mean()), len(itm)
                elif name == "H2":
                    mc = calls[(calls.moneyness >= 0.9) & (calls.moneyness <= 1.1)]
                    v, n_used = float(diff[mc.index].abs().max()), len(mc)
                elif name.startswith("H3"):
                    prem = {M: m.analysis_b(d, u, None, None, quiet=True, grid=M) for M in (400, 800, 1600)}
                    t = prem[400] if col == "am400" else (prem[1600] if col == "am1600" else prem[1600].assign(model_premium=prem[1600].model_premium + (prem[1600].model_premium - prem[800].model_premium) / 3.0))
                    t = t.assign(market_minus_model=t.market_premium - t.model_premium)
                    if name == "H3neg":
                        neg = t[(t.market_premium < 0) & (t.market_premium + t.band_halfwidth < 0)]
                        v, n_used = float(len(neg)), len(t)
                    else:
                        v = float(t.market_premium.mean())
                else:  # H4
                    c, p = d[d.cp_flag == "C"], d[d.cp_flag == "P"]
                    pr = c.merge(p, on=["expiry", "strike"], suffixes=("_c", "_p"))
                    pr["amR_c"] = pr["am1600_c"] + (pr["am1600_c"] - pr["am800_c"]) / 3.0
                    pr["amR_p"] = pr["am1600_p"] + (pr["am1600_p"] - pr["am800_p"]) / 3.0
                    pr = pr.dropna(subset=["bs_c", "bs_p", f"{col}_c", f"{col}_p"])
                    v, n_used = float((pr[f"{col}_c"] - pr[f"{col}_p"]).abs().mean() - (pr["bs_c"] - pr["bs_p"]).abs().mean()), len(pr)

                vals[col] = float(v)

            q = f"{u} " + quantity.format(n=n_used)
            r = reg.loc[q] if q in reg.index else None
            rows.append(dict(hypothesis=H, quantity=q, pre_registered_surprise_threshold=threshold, value_registered_400=float(r["value"]) if r is not None else math.nan,
                             surprise_registered_400=bool(r["surprise"]) if r is not None else math.nan, value_400_recomputed=vals["am400"], value_1600=vals["am1600"], value_richardson=vals["amR"],
                             surprise_400_recomputed=bool(test(vals["am400"])), surprise_1600=bool(test(vals["am1600"])), surprise_richardson=bool(test(vals["amR"]))))

    # method row: the registered discretization requirement, on the headline set
    for u in ("SPY",):
        d = data[u][0]
        h = d[d.headline & d["am800"].notna() & d["am1600"].notna()]
        half = float(h["half"].median())
        v400, v1600, vR = float((h.am400 - h.am1600).abs().max()), float((h.am1600 - h.am800).abs().max()), float((h.am1600 - h.am800).abs().max() / 3.0)
        rows.append(dict(hypothesis="method", quantity=f"{u} max IV change vs the next finer answer (400 vs 1600; 1600 vs 800; 1600 vs Richardson); median half-width {half:.5f}",
                         pre_registered_surprise_threshold="must be much smaller than the median half-width (flagged if above)", value_registered_400=math.nan, surprise_registered_400=True,
                         value_400_recomputed=v400, value_1600=v1600, value_richardson=vR, surprise_400_recomputed=bool(v400 > half), surprise_1600=bool(v1600 > half), surprise_richardson=bool(vR > half)))

    tab = pd.DataFrame(rows)
    tab["verdict_changes_vs_registered"] = ((tab.surprise_1600 != tab.surprise_registered_400) | (tab.surprise_richardson != tab.surprise_registered_400))
    tab = save(tab, "grid_checks")
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    print("\nREGISTERED CHECKS ON THE 400 / 1600 / RICHARDSON IV (surprise = the registered 'would surprise me' condition held):\n" + tab.drop(columns="status").round(5).to_string(index=False))


def wings(data: dict) -> None:
    d, ex_date = data["SPY"]
    c, p = d[d.cp_flag == "C"], d[d.cp_flag == "P"]
    pr = c.merge(p, on=["expiry", "strike"], suffixes=("_c", "_p"))
    pr["moneyness"] = pr["strike"] / pr["S_model_c"]
    pr["side"] = np.where(pr["expiry"] > ex_date, "after ex-date", "before ex-date")
    rows = []
    for M in (400, 1600):
        pp = pr.dropna(subset=["bs_c", "bs_p", f"am{M}_c", f"am{M}_p"])
        pp = pp.assign(gap_bs=pp["bs_c"] - pp["bs_p"], gap_am=pp[f"am{M}_c"] - pp[f"am{M}_p"])
        pp = pp.assign(wing=[wing_of(x) for x in pp["moneyness"]])
        for label in [w[0] for w in WINGS] + ["all wings"]:
            for side in ("all", "before ex-date", "after ex-date"):
                g = pp if label == "all wings" else pp[pp.wing == label]
                if side != "all":
                    g = g[g.side == side]
                if g.empty:
                    continue

                rows.append(dict(grid=M, wing=label, side=side, n_pairs=len(g), mean_abs_gap_bs_vp=VP * g.gap_bs.abs().mean(), mean_abs_gap_american_vp=VP * g.gap_am.abs().mean(),
                                 mean_signed_gap_bs_vp=VP * g.gap_bs.mean(), mean_signed_gap_american_vp=VP * g.gap_am.mean(), frac_american_smaller=float((g.gap_am.abs() < g.gap_bs.abs()).mean())))

    tab = save(pd.DataFrame(rows), "h4_wings")
    print("\nSPY H4 BY WING (vol points; IV gap = call IV - put IV at the same strike and expiry):\n" + tab.drop(columns="status").round(3).to_string(index=False))


def offset() -> None:
    """AMZN premium: original row, and a second row with the median (market - model) premium at K/S < 0.9 subtracted from the market premium (exploratory)."""
    t = pd.read_csv(DERIVED / "step5_b_premium_AMZN.csv", parse_dates=["expiry"])
    low = t[t.moneyness < OFFSET_MNY_MAX]
    off = float(low["market_minus_model"].median())
    t = t.assign(market_adjusted=t["market_premium"] - off)
    t["adj_minus_model"] = t["market_adjusted"] - t["model_premium"]
    t["adj_within_band"] = t["adj_minus_model"].abs() <= t["band_halfwidth"]
    groups = [("all", t)] + [(f"K/S {lo}-{hi}", t[(t.moneyness > lo) & (t.moneyness <= hi)]) for lo, hi in ((0.0, 0.9), (0.9, 0.95), (0.95, 1.05), (1.05, 9.0))]
    rows = [dict(group=k, n=len(g), offset_used=off, offset_n=len(low), offset_iqr_low=float(low["market_minus_model"].quantile(0.25)), offset_iqr_high=float(low["market_minus_model"].quantile(0.75)),
                 mean_market_minus_model=g.market_minus_model.mean(), mean_adjusted_minus_model=g.adj_minus_model.mean(), median_adjusted_minus_model=g.adj_minus_model.median(),
                 frac_within_band_original=g.within_band.mean(), frac_within_band_adjusted=g.adj_within_band.mean()) for k, g in groups]
    tab = save(pd.DataFrame(rows), "amzn_offset")
    print(f"\nAMZN PREMIUM OFFSET (median (market - model) over {len(low)} strikes with K/S < {OFFSET_MNY_MAX} = {off:+.4f}):\n" + tab.drop(columns="status").round(3).to_string(index=False))

    expiries = sorted(t["expiry"].unique())
    fig, axes = plt.subplots(2, len(expiries), figsize=(3.4 * len(expiries), 6.6), squeeze=False, sharey="row")
    for j, e in enumerate(expiries):
        s = t[t.expiry == e].sort_values("moneyness")
        for i, (col, label) in enumerate((("market_premium", "parity premium (market)"), ("market_adjusted", f"parity premium minus {off:.3f} offset (exploratory, circular)"))):
            ax = axes[i, j]
            ax.fill_between(s["moneyness"], s[col] - s["band_halfwidth"], s[col] + s["band_halfwidth"], color=figstyle.PALETTE["grey"], alpha=0.6, label="+/- sum of the two half-spreads")
            ax.plot(s["moneyness"], s[col], "o", ms=3, color=figstyle.PALETTE["orange"], label=label)
            ax.plot(s["moneyness"], s["model_premium"], "-", color=figstyle.PALETTE["blue"], label="model premium")
            ax.set_title(f"{pd.Timestamp(e).date()}", fontsize=8)
            ax.set_xlabel("K / S")
            if j == 0:
                ax.set_ylabel("put early-exercise premium ($)")
                ax.legend(fontsize=5.5)

    axes[0, 0].annotate("row 1: as registered", (0.02, 0.97), xycoords="axes fraction", fontsize=7, va="top", color="#555555")
    axes[1, 0].annotate("row 2: EXPLORATORY (post-hoc); offset estimated from the same data (circular)", (0.02, 0.97), xycoords="axes fraction", fontsize=7, va="top", color="#B00020")
    fig.suptitle("AMZN: put early-exercise premium, parity estimate vs model", fontsize=9)
    figstyle.save(fig, FIG / "step6_amzn_premium_two_row.png", f"source: WRDS OptionMetrics Ivy DB, snapshot 2025-05-15 16:00 ET close; N={len(t)} strikes (call-put pairs); row-2 offset = median over {len(low)} strikes with K/S < {OFFSET_MNY_MAX}, estimated from the same data (circular)")


def main() -> None:
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="*", help=f"any of {STAGES}; default all")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--force", action="store_true", help="re-solve the IV cache")
    args = ap.parse_args()
    stages = args.stages or list(STAGES)
    if bad := [x for x in stages if x not in STAGES]:
        raise SystemExit(f"unknown stage(s) {bad}; choose from {STAGES}")

    TAB.mkdir(parents=True, exist_ok=True)
    data, curve, kv = load(args.workers, args.force)
    if "discretization" in stages:
        discretization(data)

    if "level" in stages:
        level(data)

    if "split" in stages:
        split(data)

    if "grid" in stages:
        grid_checks(data)

    pt = pair_table(data) if {"parity", "dollar"} & set(stages) else None
    if "parity" in stages:
        parity(data, kv["snapshot_date"], pt)

    if "bound" in stages:
        bound(data, curve)

    if "dollar" in stages:
        dollar_gap(pt, data)

    if "wings" in stages:
        wings(data)

    if "offset" in stages:
        offset()

    rep = os.environ.get("REPLICATION_DATE")
    print(f"\nreplication (A.6): {'REPLICATION_DATE=' + rep + ' is set but a replication run needs raw data pulled for that date and `step5_market.py` pointed at it; not run here' if rep else 'REPLICATION_DATE is not set, skipped'}")


if __name__ == "__main__":
    main()