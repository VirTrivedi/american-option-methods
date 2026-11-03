"""Step 5 POST-HOC diagnostics:
    1. IV discretization check on ALL cleaned contracts (M=N=400 vs 1600), both underlyings.
    2. SPY put/call BS-vs-American gap split into dividend-model difference (BS escrowed vs European spot-jump) and early exercise
    (European vs American spot-jump), by before/after the ex-date, at M=N=400 and 1600.
    3. Parity regression per expiry: intercept (implied escrowed spot) and slope (implied discount factor) vs the 16:00 close, the model and
    OptionMetrics' forward_price.
    4. SPY: model-free lower bound S - K*exp(-r*t_d) for calls after the ex-date, and the call-put gap in dollars (market vs model)."""
from __future__ import annotations
import math
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import step5_market as m
from aopt.market.clean import add_model_inputs, clean_quotes, contract_of
from aopt.market.iv import batch_iv
from aopt.market.loader import latest_raw
from aopt.pde import solve_pde

WORKERS = 6
ROOT = m.ROOT
FIG, TAB = ROOT / "results" / "figures", ROOT / "results" / "tables"
PARITY_MNY = (0.95, 1.05)  # near the money, where early-exercise premia are small


def solve_iv(kept: pd.DataFrame, kind: str, price_col: str, M: int) -> np.ndarray:
    contracts = [contract_of(r) for _, r in kept.iterrows()]

    return np.array([x[0] for x in batch_iv(kind, [float(p) for p in kept[price_col]], contracts, M, M, WORKERS)])


def discretization_and_split(name: str, kept: pd.DataFrame, ex_date: pd.Timestamp | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = kept.copy()
    d["moneyness"] = d["strike"] / d["S_model"]
    d["am400"], d["am1600"] = solve_iv(d, "american", "mid", 400), solve_iv(d, "american", "mid", 1600)
    d["am_bid"], d["am_ask"] = solve_iv(d, "american", "bid", 400), solve_iv(d, "american", "ask", 400)
    d["bs"] = solve_iv(d, "bs", "mid", 400)
    d["eu400"], d["eu1600"] = solve_iv(d, "european", "mid", 400), solve_iv(d, "european", "mid", 1600)
    d["half"] = 0.5 * (d["am_ask"] - d["am_bid"])
    ok = d.dropna(subset=["am400", "am1600"])
    diff = (ok["am400"] - ok["am1600"]).abs()
    rows = [dict(underlying=name, n=len(ok), max_abs_iv_diff=diff.max(), p95_abs_iv_diff=diff.quantile(0.95), median_abs_iv_diff=diff.median(),
                 median_half_width=ok["half"].median(), p05_half_width=ok["half"].quantile(0.05), ratio_max_to_median_half_width=diff.max() / ok["half"].median(),
                 frac_contracts_with_diff_above_half_width=float((diff > ok["half"]).mean()))]
    disc = pd.DataFrame(rows)
    print(disc.to_string(index=False))
    d["side"] = "no dividend" if ex_date is None else np.where(d["expiry"] > ex_date, "after ex-date", "before ex-date")
    b = m.buckets(d.dropna(subset=["am400", "am1600", "bs", "eu400", "eu1600"]))
    b = b.assign(div_model_400=b["bs"] - b["eu400"], early_ex_400=b["eu400"] - b["am400"], total_400=b["bs"] - b["am400"],
                 div_model_1600=b["bs"] - b["eu1600"], early_ex_1600=b["eu1600"] - b["am1600"], total_1600=b["bs"] - b["am1600"], grid=(b["am400"] - b["am1600"]).abs())
    tab = b.groupby(["side", "cp_flag", "mny_bucket"], observed=True).agg(n=("total_400", "size"), **{c: (c, "median") for c in (
        "div_model_400", "early_ex_400", "total_400", "div_model_1600", "early_ex_1600", "total_1600")}, median_grid_diff=("grid", "median")).reset_index()
    tab.insert(0, "underlying", name)
    d.drop(columns=["dividends"]).to_csv(TAB / f"step5_diag_ivs_{name}.csv", index=False)
    
    return disc, tab


def parity(name: str, kept: pd.DataFrame, raw_fwd: pd.DataFrame, model_prem: pd.DataFrame | None) -> pd.DataFrame:
    c, p = kept[kept.cp_flag == "C"], kept[kept.cp_flag == "P"]
    pr = c.merge(p, on=["expiry", "strike"], suffixes=("_c", "_p"))
    pr = pr[(pr.strike / pr.S_model_c >= PARITY_MNY[0]) & (pr.strike / pr.S_model_c <= PARITY_MNY[1])]
    rows = []
    for expiry, g in pr.groupby("expiry"):
        if len(g) < 4:
            continue
        
        S, T, r = float(g.S_model_c.iloc[0]), float(g.T_c.iloc[0]), float(g.r_c.iloc[0])
        divs = g.dividends_c.iloc[0]
        from aopt.params import OptionParams
        pv = OptionParams(S=S, K=1.0, T=T, r=r, q=0.0, sigma=0.2, option_type="call", dividends=divs).dividend_pv(0.0)
        y = (g.mid_c - g.mid_p).to_numpy()
        K = g.strike.to_numpy()
        slope, intercept = np.polyfit(K, y, 1)
        a0 = float(np.mean(y + K * math.exp(-r * T)))  # intercept with the slope fixed at -exp(-rT)
        fwd_vendor = float(raw_fwd[raw_fwd.exdate == expiry]["forward_price"].median()) if len(raw_fwd) else math.nan
        row = dict(underlying=name, expiry=expiry.date(), days=int(g.days_c.iloc[0]), n_strikes=len(g), spot_close=S, pv_dividends=pv,
                   model_escrowed_spot=S - pv, implied_escrowed_spot_fixed_slope=a0, offset_dollars=a0 - (S - pv), offset_pct_of_spot=100 * (a0 - (S - pv)) / S,
                   free_slope=slope, model_slope=-math.exp(-r * T), implied_rate_from_slope=-math.log(-slope) / T if slope < 0 else math.nan, model_rate=r,
                   free_intercept=intercept, implied_forward_fixed_slope=a0 * math.exp(r * T), model_forward=(S - pv) * math.exp(r * T), vendor_forward=fwd_vendor)
        if model_prem is not None:  # AMZN: add back the model put premium (calls are European at q = 0) so early exercise does not masquerade as an offset
            mp = model_prem[model_prem.expiry == expiry].set_index("strike")["model_premium"]
            adj = g.strike.map(mp).fillna(0.0).to_numpy()
            row["offset_dollars_after_model_premium"] = float(np.mean(y + adj + K * math.exp(-r * T))) - (S - pv)
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def bound_and_dollars(name: str, kept: pd.DataFrame, ivs: pd.DataFrame, ex_ts, curve) -> tuple[pd.DataFrame, pd.DataFrame]:
    from datetime import datetime
    from aopt.market.timeutil import EX_DATE_TIME, year_fraction
    d = kept.join(ivs[["am400", "bs"]].set_axis(kept.index))
    rows_b = []
    if ex_ts is not None:
        qt = d["quote_time"].iloc[0].to_pydatetime()
        t_d = year_fraction(qt, datetime.combine(ex_ts.date(), EX_DATE_TIME))
        r_d = float(curve.rate_at(t_d * 365.0))
        calls = d[(d.cp_flag == "C") & (d.expiry > ex_ts)]
        bound = calls["S_model"] - calls["strike"] * math.exp(-r_d * t_d)  # hold to the ex-date and exercise cum-dividend
        eur_bound = calls["S_model"] - calls.apply(lambda r: sum(a * math.exp(-r["r"] * t) for t, a in r["dividends"]), axis=1) - calls["strike"] * np.exp(-calls["r"] * calls["T"])
        tab = pd.DataFrame(dict(expiry=calls["expiry"].dt.date, strike=calls["strike"], moneyness=calls["strike"] / calls["S_model"], bid=calls["bid"], ask=calls["ask"],
                                bound=bound, ask_minus_bound=calls["ask"] - bound, bid_minus_bound=calls["bid"] - bound, ask_minus_european_bound=calls["ask"] - eur_bound))
        rows_b = tab
        print(f"  {name}: model-free bound S - K exp(-r t_d) (t_d = {t_d:.4f}y) on {len(tab)} calls after the ex-date: ask below bound for {(tab.ask_minus_bound < 0).sum()}, "
              f"by more than $0.01 for {(tab.ask_minus_bound < -0.01).sum()}; min (ask - bound) = {tab.ask_minus_bound.min():+.3f}; bid below bound for {(tab.bid_minus_bound < 0).sum()}")
    
    c, p = d[d.cp_flag == "C"], d[d.cp_flag == "P"]
    pr = c.merge(p, on=["expiry", "strike"], suffixes=("_c", "_p"))
    rows = []
    for _, r in pr.iterrows():
        S, K, T, rate, divs = r.S_model_c, r.strike, r.T_c, r.r_c, r.dividends_c
        sigma = r.am400_p if K / S < 1.0 else r.am400_c  # IV of the out-of-the-money leg, least affected by early exercise
        if not math.isfinite(sigma):
            continue
    
        from aopt.market.iv import Contract
        cc, pc = Contract(S, K, T, rate, "C", divs), Contract(S, K, T, rate, "P", divs)
        prem = lambda con: (solve_pde(con.params(sigma), 400, 400, american_method="penalty").price - solve_pde(con.params(sigma), 400, 400, european=True).price)
        euro_parity = S - cc.params(sigma).dividend_pv(0.0) - K * math.exp(-rate * T)
        rows.append(dict(underlying=name, expiry=r.expiry.date(), strike=K, moneyness=K / S, side="no dividend" if ex_ts is None else ("after ex-date" if r.expiry > ex_ts else "before ex-date"),
                         market_deviation=(r.mid_c - r.mid_p) - euro_parity, model_deviation=prem(cc) - prem(pc)))
    
    dol = pd.DataFrame(rows)
    dol["market_minus_model"] = dol["market_deviation"] - dol["model_deviation"]
    
    return (pd.DataFrame(rows_b) if len(rows_b) else pd.DataFrame()), dol


def main() -> None:
    kv = m.parse_hypotheses(m.HYPOTHESES)
    unders, curve, _ = m.load_real(kv)
    date = kv["snapshot_date"]
    discs, tabs, pars, bounds, dols = [], [], [], [], []
    for u in unders:
        kept, _ = clean_quotes(add_model_inputs(u.quotes, curve, u.distributions), m.CFG, u.load_rejects)
        ex_date = pd.Timestamp(u.distributions["ex_date"].iloc[0]) if len(u.distributions) else None
        print(f"{u.name}: {len(kept)} contracts; ex-date {ex_date}")
        disc, tab = discretization_and_split(u.name, kept, ex_date)
        ivs = pd.read_csv(TAB / f"step5_diag_ivs_{u.name}.csv", parse_dates=["expiry"])
        raw = pd.read_csv(latest_raw(m.RAW_DIR, "opprcd", u.name, date), parse_dates=["exdate"])
        prem = pd.read_csv(TAB / f"step5_b_premium_{u.name}.csv", parse_dates=["expiry"]) if (TAB / f"step5_b_premium_{u.name}.csv").exists() else None
        pars.append(parity(u.name, kept, raw, prem))
        bnd, dol = bound_and_dollars(u.name, kept, ivs, ex_date, curve)
        discs.append(disc); tabs.append(tab); dols.append(dol)
        if len(bnd):
            bnd.insert(0, "underlying", u.name); bounds.append(bnd)
    
    pd.concat(discs).to_csv(TAB / "step5_diag_discretization.csv", index=False)
    pd.concat(tabs).to_csv(TAB / "step5_diag_dividend_split.csv", index=False)
    par = pd.concat(pars); par.to_csv(TAB / "step5_diag_parity.csv", index=False)
    if bounds:
        pd.concat(bounds).to_csv(TAB / "step5_diag_model_free_bound.csv", index=False)
    
    dol = pd.concat(dols); dol.to_csv(TAB / "step5_diag_dollar_gap.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print("\nDIVIDEND-MODEL SPLIT (median, IV decimals; split by side of the ex-date):\n" + pd.concat(tabs)[lambda x: x.underlying == "SPY"].round(5).to_string(index=False))
    print("\nPARITY REGRESSION (near the money):\n" + par.drop(columns=["underlying"]).round(4).to_string(index=False))
    g = dol.groupby(["underlying", "side", pd.cut(dol.moneyness, m.MNY_BINS).astype(str)], observed=True).agg(
        n=("strike", "size"), market=("market_deviation", "mean"), model=("model_deviation", "mean"), diff=("market_minus_model", "mean")).round(3)
    print("\nCALL-PUT DEVIATION FROM EUROPEAN PARITY IN DOLLARS (market vs model, mean):\n" + g.to_string())
    g.to_csv(TAB / "step5_diag_dollar_gap_by_bucket.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, color in (("AMZN", "#4C78A8"), ("SPY", "#F58518")):
        s = par[par.underlying == name]
        axes[0].plot(s["days"], s["offset_dollars"], "o-", color=color, label=f"{name}: implied - model escrowed spot ($)")
        axes[1].plot(s["days"], 100 * (s["implied_rate_from_slope"] - s["model_rate"]), "o-", color=color, label=name)
    
    axes[0].axhline(0, color="k", lw=0.8); axes[1].axhline(0, color="k", lw=0.8)
    axes[0].set_xlabel("days to expiry"); axes[0].set_ylabel("parity intercept offset ($)"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("days to expiry"); axes[1].set_ylabel("implied minus model rate (percentage points)")
    axes[0].set_title("post-hoc: spot-level offset (constant => timing)", fontsize=9); axes[1].set_title("post-hoc: slope-implied rate error", fontsize=9)
    for ax in axes: 
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout(); fig.savefig(FIG / "step5_diag_parity.png", dpi=130)


if __name__ == "__main__":
    main()