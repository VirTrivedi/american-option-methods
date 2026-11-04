"""Step 5: sanity check of the pricing model against one market snapshot."""
from __future__ import annotations
import argparse
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle
import numpy as np
import pandas as pd
from aopt.bs import bs_price
from aopt.market.clean import CleanConfig, add_model_inputs, clean_quotes, contract_of, spread_sensitivity
from aopt.market.dividends import load_distributions
from aopt.market.iv import american_iv, batch_iv
from aopt.market.loader import build_quotes, latest_raw, load_option_prices, underlying_close
from aopt.market.rates import ZeroCurve, curve_from_zerocd
from aopt.market.schema import QUOTE_COLUMNS
from aopt.market.synthetic import make_chain
from aopt.pde import solve_pde

# configuration
ROOT = Path(__file__).resolve().parents[1]
HYPOTHESES = ROOT / "docs" / "hypotheses_step5.md"
RAW_DIR = ROOT / "data" / "raw"
DERIVED_DIR = ROOT / "data" / "derived"  # per-contract tables carry vendor quotes
CFG = CleanConfig()
MNY_BINS = [0.8, 0.9, 0.95, 1.05, 1.1, 1.2]
DAY_BINS = [0, 45, 90, 182, 366]
BOOTSTRAP_B, BOOTSTRAP_SEED = 1000, 20240315
SENSITIVITY = [("base", {}), ("r -50bp", dict(r_shift_bp=-50.0)), ("r +50bp", dict(r_shift_bp=50.0)), ("spot -0.05%", dict(spot_shift=-5e-4)),
               ("spot +0.05%", dict(spot_shift=5e-4)), ("dividends x0.9", dict(div_scale=0.9)), ("dividends x1.1", dict(div_scale=1.1))]
DISCRETIZATION_SAMPLE = 30
C = {"bs": figstyle.PALETTE["orange"], "am": figstyle.PALETTE["blue"], "band": figstyle.PALETTE["grey"], "vendor": figstyle.PALETTE["green"]}


@dataclass
class Underlying:
    name: str
    quotes: pd.DataFrame
    distributions: pd.DataFrame
    load_rejects: dict[str, int]
    pull_stamp: str = "synthetic"


def parse_hypotheses(path: Path) -> dict[str, str]:
    """key: value lines inside docs/hypotheses_step5.md; refuses to proceed while any snapshot field is TBD or missing."""
    kv = {m.group(1): m.group(2).strip() for line in path.read_text().splitlines() if (m := re.match(r"^(\w+):\s*([^#]*?)\s*(#.*)?$", line))}
    if kv.get("status", "CONFIRMED") != "CONFIRMED":
        raise SystemExit(f"{path} has status {kv.get('status')!r}: set `status: CONFIRMED` before running real data")

    todo = [k for k in ("snapshot_date", "ticker_nondividend", "ticker_dividend", "min_days", "max_days", "expiry_targets_days", "cash_distr_types") if kv.get(k, "TBD") == "TBD"]
    if todo:
        raise SystemExit(f"{path}: still TBD or missing: {todo}")

    return kv


def select_expiries(quotes: pd.DataFrame, targets: list[int], name: str) -> pd.DataFrame:
    """Keep, for each target (days after the snapshot), the nearest listed expiry not already chosen (stated in advance rule)."""
    days = (quotes["expiry"].dt.normalize() - quotes["quote_time"].dt.normalize()).dt.days
    listed = sorted(set(zip(quotes["expiry"], days)), key=lambda x: x[1])
    chosen: list = []
    for t in targets:
        best = min((e for e in listed if e[0] not in {c[0] for c in chosen}), key=lambda e: abs(e[1] - t), default=None)
        if best is not None:
            chosen.append(best)

    print(f"{name}: expiries kept {[(str(e.date()), d) for e, d in sorted(chosen, key=lambda x: x[1])]} of {len(listed)} listed")

    return quotes[quotes["expiry"].isin([c[0] for c in chosen])].reset_index(drop=True)


def load_real(kv: dict[str, str]) -> tuple[list[Underlying], ZeroCurve, datetime]:
    date = kv["snapshot_date"]
    cash_types = {t.strip() for t in kv["cash_distr_types"].split(",")}
    curve_path = sorted(RAW_DIR.glob(f"zerocd_{date}_*.csv"))[-1]
    curve = curve_from_zerocd(pd.read_csv(curve_path))
    out = []
    for ticker in (kv["ticker_nondividend"], kv["ticker_dividend"]):
        prices = load_option_prices(pd.read_csv(latest_raw(RAW_DIR, "opprcd", ticker, date)), pd.read_csv(latest_raw(RAW_DIR, "opinfd", ticker, date)))
        px = pd.read_csv(latest_raw(RAW_DIR, "secprd", ticker, date))
        secids = prices["secid"].unique()
        if len(secids) != 1:
            raise SystemExit(f"{ticker}: opprcd contains secids {secids}")
        
        spot = underlying_close(px, int(secids[0]), prices["date"].iloc[0])
        quotes, rej = build_quotes(prices, spot, ticker)
        quotes = select_expiries(quotes, [int(x) for x in kv["expiry_targets_days"].split(",")], ticker)
        earnings = kv.get("next_earnings_date")
        if ticker == kv["ticker_nondividend"] and earnings and (quotes["expiry"] >= pd.Timestamp(earnings)).any():
            raise SystemExit(f"{ticker}: a selected expiry is on or after the next earnings date {earnings}; lower max_days / the targets")
        
        if (quotes["exercise_style"] != "A").any():
            raise SystemExit(f"{ticker}: non-American contracts present; the American model must not be applied to them")
        
        dist = load_distributions(pd.read_csv(latest_raw(RAW_DIR, "distrd", ticker, date), parse_dates=["ex_date"]), cash_types)
        last_expiry = quotes["expiry"].max()
        in_window = dist[(dist["ex_date"] > quotes["quote_time"].iloc[0].normalize()) & (dist["ex_date"] <= last_expiry)]
        if ticker == kv["ticker_nondividend"] and not dist.empty:
            raise SystemExit(f"{ticker} is declared non-dividend but distrd lists {len(dist)} distribution(s) in the window; pick another ticker")
        
        if ticker == kv["ticker_dividend"] and len(in_window) != 1:
            raise SystemExit(f"{ticker}: stated in advance rule needs exactly one ex-date in the window, distrd lists {len(in_window)}; choose another snapshot date")
        
        out.append(Underlying(ticker, quotes, dist, rej, latest_raw(RAW_DIR, "opprcd", ticker, date).stem.split("_")[-1]))

    return out, curve, out[0].quotes["quote_time"].iloc[0].to_pydatetime()


def load_synthetic() -> tuple[list[Underlying], ZeroCurve, datetime]:
    fixtures = ROOT / "tests" / "fixtures"
    qt = datetime(2024, 3, 15, 16, 0)
    curve = ZeroCurve(np.array([30.0, 365.0]), np.array([0.045, 0.05]))
    div = pd.read_csv(fixtures / "synthetic_chain.csv", parse_dates=["quote_time", "spot_time", "expiry"])[QUOTE_COLUMNS]
    nodiv = make_chain(curve=curve, rel_spread=0.02, noise_rel=0.002, tick=0.01, seed=11, strikes_over_spot=(0.9, 0.95, 1.0, 1.05, 1.1), expiry_days=(35, 91, 182), underlying="SYN_ND")[QUOTE_COLUMNS]
    dist = pd.read_csv(fixtures / "synthetic_distributions.csv", parse_dates=["ex_date"])

    return [Underlying("SYN_ND", nodiv, load_distributions(pd.DataFrame(), None), {}), Underlying("SYN_DIV", div, dist, {})], curve, qt


def iv_table(kept: pd.DataFrame, workers: int, kinds=("bs", "american"), mid_only: bool = False) -> pd.DataFrame:
    """IVs at bid/mid/ask for every kept contract, under European BS (escrowed dividends) and the American PDE model."""
    contracts = [contract_of(r) for _, r in kept.iterrows()]
    cols = ("mid",) if mid_only else ("bid", "mid", "ask")
    out = kept[["underlying", "expiry", "strike", "cp_flag", "spot", "S_model", "days", "T", "r", "bid", "ask", "mid", "vendor_iv"]].copy()
    out["moneyness"] = out["strike"] / out["S_model"]
    for kind in kinds:
        prices = [float(kept[c].iloc[i]) for c in cols for i in range(len(kept))]
        res = batch_iv(kind, prices, contracts * len(cols), workers=workers)
        n = len(kept)
        for j, c in enumerate(cols):
            out[f"{kind}_{c}"] = [x[0] for x in res[j * n:(j + 1) * n]]
            if c == "mid":
                out[f"{kind}_reason"] = [x[1] for x in res[j * n:(j + 1) * n]]

        if not mid_only:
            out[f"{kind}_half"] = 0.5 * (out[f"{kind}_ask"] - out[f"{kind}_bid"])

    return out


MIN_CELL = 5  # heatmap cells with fewer contracts are masked
PROVENANCE = ""  # set in main(): source and snapshot, stamped on every figure


def derived_dir(tabdir: Path) -> Path:
    """Where per-contract tables go: data/derived (gitignored) for the real run, next to the other synthetic outputs otherwise."""
    out = DERIVED_DIR if tabdir == ROOT / "results" / "tables" else tabdir
    out.mkdir(parents=True, exist_ok=True)

    return out


def buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Add ordered categorical bucket columns (so tables and heatmaps are in numeric, not alphabetical, order)."""
    df = df.copy()
    for col, src, bins in (("mny_bucket", "moneyness", MNY_BINS), ("maturity_bucket", "days", DAY_BINS)):
        cut = pd.cut(df[src], bins)
        df[col] = pd.Categorical(cut.astype(str), categories=[str(i) for i in cut.cat.categories], ordered=True)

    return df


def analysis_a(t: pd.DataFrame, name: str, figdir: Path, tabdir: Path) -> pd.DataFrame:
    d = buckets(t.dropna(subset=["bs_mid", "american_mid", "american_half"]))
    d["diff"] = d["bs_mid"] - d["american_mid"]
    d["exceeds_half_width"] = d["diff"].abs() > d["american_half"]
    tab = d.groupby(["cp_flag", "mny_bucket", "maturity_bucket"], observed=True).agg(n=("diff", "size"), median_diff=("diff", "median"), 
                    mean_half_width=("american_half", "mean"), frac_exceeds_half_width=("exceeds_half_width", "mean")).reset_index()
    tab.to_csv(tabdir / f"step5_a_bs_vs_american_{name}.csv", index=False)
    d.drop(columns=["spot"]).to_csv(derived_dir(tabdir) / f"step5_iv_contracts_{name}.csv", index=False)
    print(f"  (a) {name}: contracts with both IVs {len(d)}; fraction |BS-American| > American half-width: puts {d[d.cp_flag == 'P']['exceeds_half_width'].mean():.3f}, "
          f"calls {d[d.cp_flag == 'C']['exceeds_half_width'].mean():.3f}; median diff puts {d[d.cp_flag == 'P']['diff'].median():+.4f}, calls {d[d.cp_flag == 'C']['diff'].median():+.4f}")
    expiries = sorted(d["expiry"].unique())
    fig, axes = plt.subplots(2, len(expiries), figsize=(3.6 * len(expiries), 6.4), squeeze=False, sharey="row")
    for j, e in enumerate(expiries):
        for i, cp in enumerate(("P", "C")):
            s = d[(d.expiry == e) & (d.cp_flag == cp)].sort_values("moneyness")
            ax = axes[i, j]
            ax.fill_between(s["moneyness"], s["american_bid"], s["american_ask"], color=C["band"], alpha=0.6, label="bid-ask IV band (American)")
            ax.plot(s["moneyness"], s["bs_mid"], "o-", ms=3, color=C["bs"], label="BS IV (mid)")
            ax.plot(s["moneyness"], s["american_mid"], "s-", ms=3, color=C["am"], label="American IV (mid)")
            ax.set_title(f"{'put' if cp == 'P' else 'call'}, {pd.Timestamp(e).date()}", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("K / S")

    axes[0, 0].legend(fontsize=6)
    axes[0, 0].set_ylabel("implied vol")
    axes[1, 0].set_ylabel("implied vol")
    fig.suptitle(f"{name}: IV smiles, BS vs American (sanity check, not a mispricing search)", fontsize=9)
    figstyle.save(fig, figdir / f"step5_a_smiles_{name}.png", f"{PROVENANCE}; N={len(d)} contracts")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, cp in zip(axes[:2], ("P", "C")):
        s = d[d.cp_flag == cp]
        ax.errorbar(s["moneyness"], s["diff"], yerr=s["american_half"], fmt="o", ms=3, color=C["am"], ecolor=C["band"], alpha=0.8, label="BS - American, +/- bid-ask half-width")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{name} {'puts' if cp == 'P' else 'calls'}: IV difference vs noise band", fontsize=9)
        ax.set_xlabel("K / S")
        ax.set_ylabel("BS IV - American IV")
        ax.grid(True, alpha=0.3)

    ax.legend(fontsize=7)
    heatmap(axes[2], d[d.cp_flag == "P"], "diff", f"{name}: median put BS - American IV (vol decimals)")
    figstyle.save(fig, figdir / f"step5_a_difference_{name}.png", f"{PROVENANCE}; N={len(d)} contracts")

    return d


def heatmap(ax, puts: pd.DataFrame, col: str, title: str) -> None:
    """Median of `col` by moneyness bucket (y, ascending upwards) and maturity bucket (x); each cell prints its contract count and cells with n < MIN_CELL are masked."""
    pt = puts.pivot_table(index="mny_bucket", columns="maturity_bucket", values=col, aggfunc="median", observed=False, dropna=False)
    cnt = puts.pivot_table(index="mny_bucket", columns="maturity_bucket", values=col, aggfunc="size", observed=False, dropna=False).fillna(0).astype(int)
    vals = np.ma.masked_where(cnt.to_numpy() < MIN_CELL, pt.to_numpy(float))
    im = ax.imshow(vals, aspect="auto", origin="lower")
    ax.set_xticks(range(pt.shape[1]), [f"{c} d" for c in pt.columns], fontsize=7, rotation=20)
    ax.set_yticks(range(pt.shape[0]), pt.index, fontsize=7)
    ax.set_xlabel("days to expiry"); ax.set_ylabel("K / S")
    ax.grid(False)
    for i in range(cnt.shape[0]):
        for j in range(cnt.shape[1]):
            ax.text(j, i, f"n={cnt.iat[i, j]}", ha="center", va="center", fontsize=7, color="w" if cnt.iat[i, j] >= MIN_CELL else "#888888")

    ax.set_title(f"{title}\ncells with n < {MIN_CELL} masked", fontsize=8)
    ax.figure.colorbar(im, ax=ax)


def summarize_premium(d: pd.DataFrame, name: str, tabdir: Path) -> None:
    """Aggregate view of the strike-level parity premium (the strike-level table itself stays out of the repository)."""
    d = d.assign(group=pd.cut(d["moneyness"], [0.0, 0.95, 1.05, 9.0], labels=["OTM (K/S<=0.95)", "ATM (0.95-1.05)", "ITM (K/S>1.05)"]).astype(str))
    rows = []
    for kind, key in (("all", None), ("expiry", d["expiry"].dt.date.astype(str)), ("moneyness", d["group"])):
        groups = [("all", d)] if key is None else list(d.groupby(key))
        for label, g in groups:
            rows.append(dict(by=kind, group=label, n=len(g), mean_market_premium=g.market_premium.mean(), mean_model_premium=g.model_premium.mean(), mean_market_minus_model=g.market_minus_model.mean(),
                             median_market_minus_model=g.market_minus_model.median(), frac_within_band=g.within_band.mean()))

    pd.DataFrame(rows).to_csv(tabdir / f"step5_b_premium_summary_{name}.csv", index=False)


def pairs(t: pd.DataFrame) -> pd.DataFrame:
    """One row per (underlying, expiry, strike) with both legs present."""
    c, p = t[t.cp_flag == "C"], t[t.cp_flag == "P"]
    m = c.merge(p, on=["underlying", "expiry", "strike"], suffixes=("_c", "_p"))

    return m


def analysis_b(raw_inputs: pd.DataFrame, name: str, figdir: Path, tabdir: Path, scenario: dict | None = None, quiet: bool = False, grid: int = 400) -> pd.DataFrame:
    """Parity premium vs model premium for a non-dividend underlying (premium test)."""
    if any(len(x) for x in raw_inputs["dividends"]):
        raise SystemExit(f"{name}: analysis (b) requires no dividends in the window")
    
    m = pairs(raw_inputs)
    rows = []
    for _, r in m.iterrows():
        S, K, T, rate = r["S_model_c"], r["strike"], r["T_c"], r["r_c"]
        market = r["mid_p"] - (r["mid_c"] - S + K * math.exp(-rate * T))
        band = 0.5 * ((r["ask_p"] - r["bid_p"]) + (r["ask_c"] - r["bid_c"]))  # worst case propagation of both half-spreads (linear)
        from aopt.market.iv import Contract, bs_iv
        cc = Contract(S, K, T, rate, "C")
        sig, why = bs_iv(r["mid_c"], cc)
        if not math.isfinite(sig):
            continue

        pc = Contract(S, K, T, rate, "P")
        model = solve_pde(pc.params(sig), grid, grid, american_method="penalty").price - float(bs_price(S, K, T, rate, 0.0, sig, "put"))
        rows.append(dict(underlying=name, expiry=r["expiry"], strike=K, moneyness=K / S, days=r["days_c"], market_premium=market, band_halfwidth=band, model_premium=model,
                         market_minus_model=market - model, within_band=abs(market - model) <= band))
        
    d = pd.DataFrame(rows)
    if quiet:
        return d
    
    d.to_csv(derived_dir(tabdir) / f"step5_b_premium_{name}.csv", index=False)
    summarize_premium(d, name, tabdir)
    print(f"  (b) {name}: {len(d)} strikes; mean market premium {d.market_premium.mean():+.4f}, mean model premium {d.model_premium.mean():+.4f}, "
          f"mean (market-model) {d.market_minus_model.mean():+.4f}; fraction within propagated band {d.within_band.mean():.3f}")
    expiries = sorted(d["expiry"].unique())
    fig, axes = plt.subplots(1, len(expiries), figsize=(3.6 * len(expiries), 4), squeeze=False, sharey=True)
    for ax, e in zip(axes[0], expiries):
        s = d[d.expiry == e].sort_values("moneyness")
        ax.fill_between(s["moneyness"], s["market_premium"] - s["band_halfwidth"], s["market_premium"] + s["band_halfwidth"], color=C["band"], alpha=0.6, label="parity premium +/- sum of the two half-spreads (worst case)")
        ax.plot(s["moneyness"], s["market_premium"], "o", ms=3, color=C["bs"], label="parity premium (market)")
        ax.plot(s["moneyness"], s["model_premium"], "-", color=C["am"], label="model premium")
        ax.set_title(str(pd.Timestamp(e).date()), fontsize=8)
        ax.set_xlabel("K / S")
        ax.grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("early-exercise premium of the put")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"{name}: put early-exercise premium, parity estimate vs model (assumes no borrow cost, calls not exercised early, synchronized quotes)", fontsize=8)
    figstyle.save(fig, figdir / f"step5_b_premium_{name}.png", f"{PROVENANCE}; N={len(d)} strikes (call-put pairs)")

    return d


def analysis_c(t: pd.DataFrame, name: str, figdir: Path, tabdir: Path, quiet: bool = False) -> pd.DataFrame:
    m = pairs(t).dropna(subset=["bs_mid_c", "bs_mid_p", "american_mid_c", "american_mid_p"])
    m = m.assign(moneyness=m["strike"] / m["S_model_c"])
    m["gap_bs"], m["gap_american"] = m["bs_mid_c"] - m["bs_mid_p"], m["american_mid_c"] - m["american_mid_p"]
    m["days"] = m["days_c"]
    if quiet:
        return m
    
    b = buckets(m)
    tab = b.groupby("mny_bucket", observed=True).agg(n=("gap_bs", "size"), mean_gap_bs=("gap_bs", "mean"), mean_abs_gap_bs=("gap_bs", lambda x: x.abs().mean()),
                    mean_gap_american=("gap_american", "mean"), mean_abs_gap_american=("gap_american", lambda x: x.abs().mean())).reset_index()
    tab.to_csv(tabdir / f"step5_c_call_put_gap_{name}.csv", index=False)
    print(f"  (c) {name}: {len(m)} pairs; mean |call IV - put IV|: BS {m.gap_bs.abs().mean():.4f}, American {m.gap_american.abs().mean():.4f}; "
          f"American gap smaller in {(m.gap_american.abs() < m.gap_bs.abs()).mean():.3f} of pairs")
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(m["moneyness"], m["gap_bs"], "o", ms=4, color=C["bs"], label="BS (escrowed forward)")
    ax.plot(m["moneyness"], m["gap_american"], "s", ms=4, color=C["am"], label="American, discrete dividends")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("K / S")
    ax.set_ylabel("call IV - put IV at the same strike and expiry")
    ax.set_title(f"{name}: call-put IV consistency", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    figstyle.save(fig, figdir / f"step5_c_call_put_gap_{name}.png", f"{PROVENANCE}; N={len(m)} call-put pairs")

    return m


def summary_stats(per: dict[str, dict[str, pd.DataFrame]]) -> dict[str, float]:
    """The handful of outputs tracked in the sensitivity table and the bootstrap."""
    out: dict[str, float] = {}
    for name, parts in per.items():
        a, b, c = parts.get("a"), parts.get("b"), parts.get("c")
        if a is not None and len(a):
            puts = a[a.cp_flag == "P"]
            out[f"{name}: median put IV diff (BS - American)"] = float((puts["bs_mid"] - puts["american_mid"]).median())

        if b is not None and len(b):
            out[f"{name}: mean (market - model) put premium"] = float(b["market_minus_model"].mean())
            out[f"{name}: mean model put premium"] = float(b["model_premium"].mean())

        if c is not None and len(c):
            out[f"{name}: mean |call-put IV gap| American"] = float(c["gap_american"].abs().mean())
            out[f"{name}: mean |call-put IV gap| BS"] = float(c["gap_bs"].abs().mean())

    return out


def analysis_d(prepared: dict[str, tuple[pd.DataFrame, pd.DataFrame]], curve: ZeroCurve, quote_dists: dict[str, pd.DataFrame], workers: int, tabdir: Path) -> None:
    """Rerun (a)-(c) at mid prices with shifted inputs for the contracts that survived the base cleaning."""
    rows, base = [], None
    for label, kw in SENSITIVITY:
        per: dict[str, dict[str, pd.DataFrame]] = {}
        for name, (kept_idx_rows, _) in prepared.items():
            q = kept_idx_rows[QUOTE_COLUMNS]
            shifted = add_model_inputs(q, curve, quote_dists[name], **kw)
            t = iv_table(shifted, workers, mid_only=True)
            parts: dict[str, pd.DataFrame] = {}
            t["american_half"], t["american_bid"], t["american_ask"] = np.nan, np.nan, np.nan
            parts["a"] = t.dropna(subset=["bs_mid", "american_mid"])
            if all(len(x) == 0 for x in shifted["dividends"]):
                parts["b"] = analysis_b(shifted, name, Path("."), Path("."), quiet=True)
            else:
                parts["c"] = analysis_c(t, name, Path("."), Path("."), quiet=True)
            per[name] = parts

        stats = summary_stats(per)
        if base is None:
            base = stats

        for k, v in stats.items():
            rows.append(dict(scenario=label, statistic=k, value=v, change_vs_base=v - base[k]))

    df = pd.DataFrame(rows)
    # the "model effect" each output is compared with: the BS-American IV difference itself; the model premium; for the call-put gap the
    # improvement of the American gap over the BS gap
    effect: dict[str, float] = {}
    for k in base:
        name = k.split(":")[0]
        if "median put IV diff" in k:
            effect[k] = abs(base[k])
        elif "market - model" in k or "model put premium" in k:
            effect[k] = abs(base.get(f"{name}: mean model put premium", np.nan))
        else:
            effect[k] = abs(base.get(f"{name}: mean |call-put IV gap| BS", np.nan) - base.get(f"{name}: mean |call-put IV gap| American", np.nan))

    df["base_value"] = df["statistic"].map(base)
    df["model_effect"] = df["statistic"].map(effect)
    df["change_relative_to_model_effect"] = df["change_vs_base"].abs() / df["model_effect"].replace(0, np.nan)
    df.to_csv(tabdir / "step5_d_sensitivity.csv", index=False)
    print("  (d) sensitivity (change vs base, and as a multiple of the model effect: BS-American IV gap / model premium / BS-vs-American call-put gap improvement):")
    for _, r in df[df.scenario != "base"].iterrows():
        print(f"      {r.scenario:16s} {r.statistic:55s} {r.change_vs_base:+.5f}  ({r.change_relative_to_model_effect:.2f}x of effect {r.model_effect:.4f})")


def analysis_vendor(a_tables: dict[str, pd.DataFrame], tabdir: Path) -> None:
    rows, summary = [], []
    for name, t in a_tables.items():
        d = buckets(t.dropna(subset=["vendor_iv", "american_mid"]))
        if d.empty:
            continue

        d["abs_diff_am"] = (d["american_mid"] - d["vendor_iv"]).abs()
        d["abs_diff_bs"] = (d["bs_mid"] - d["vendor_iv"]).abs()
        for keys in (["cp_flag", "mny_bucket"], ["cp_flag", "maturity_bucket"]):
            g = d.groupby(keys, observed=True).agg(n=("abs_diff_am", "size"), median_abs_diff_american=("abs_diff_am", "median"), p95_abs_diff_american=("abs_diff_am", lambda x: x.quantile(0.95)),
                          median_abs_diff_bs=("abs_diff_bs", "median"), p95_abs_diff_bs=("abs_diff_bs", lambda x: x.quantile(0.95))).reset_index()
            g.insert(0, "grouping", "+".join(keys[1:]))
            g.insert(0, "underlying", name)
            rows.append(g)

        mine = ((d["american_mid"] >= d["american_bid"]) & (d["american_mid"] <= d["american_ask"])).mean()
        vend = ((d["vendor_iv"] >= d["american_bid"]) & (d["vendor_iv"] <= d["american_ask"])).mean()
        vend_bs = ((d["vendor_iv"] >= d["bs_bid"]) & (d["vendor_iv"] <= d["bs_ask"])).mean()
        summary.append(dict(underlying=name, n=len(d), median_signed_vendor_minus_american=(d["vendor_iv"] - d["american_mid"]).median(),
                            median_signed_vendor_minus_bs=(d["vendor_iv"] - d["bs_mid"]).median(), median_abs_diff_american=d["abs_diff_am"].median(), p95_abs_diff_american=d["abs_diff_am"].quantile(0.95),
                            median_abs_diff_bs=d["abs_diff_bs"].median(), p95_abs_diff_bs=d["abs_diff_bs"].quantile(0.95),
                            frac_my_mid_inside_my_interval=mine, frac_vendor_inside_my_american_interval=vend, frac_vendor_inside_my_bs_interval=vend_bs))
    if rows:
        pd.concat(rows).to_csv(tabdir / "step5_e_vendor_by_bucket.csv", index=False)
        s = pd.DataFrame(summary)
        s.to_csv(tabdir / "step5_e_vendor_summary.csv", index=False)
        print("  (e) vendor IV comparison:\n" + s.to_string(index=False))
    else:
        print("  (e) no vendor IV available")


def analysis_bootstrap(a_tables: dict[str, pd.DataFrame], b_tables: dict[str, pd.DataFrame], c_tables: dict[str, pd.DataFrame], B: int, tabdir: Path) -> None:
    """Resample contracts within expiry. Contracts at neighbouring strikes are not independent, so these CIs are too narrow."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    specs = []
    for name, t in a_tables.items():
        p = t[t.cp_flag == "P"].assign(diff=lambda x: x["bs_mid"] - x["american_mid"], exc=lambda x: (x["bs_mid"] - x["american_mid"]).abs() > x["american_half"])
        specs += [(f"{name}: median put IV diff (BS - American)", p, "diff", np.median), (f"{name}: fraction |BS - American| > half-width (puts)", p, "exc", np.mean)]

    for name, t in b_tables.items():
        specs.append((f"{name}: mean (market - model) put premium", t, "market_minus_model", np.mean))
    
    for name, t in c_tables.items():
        specs += [(f"{name}: mean |call-put gap| American", t.assign(a=t["gap_american"].abs()), "a", np.mean), (f"{name}: mean |call-put gap| BS", t.assign(a=t["gap_bs"].abs()), "a", np.mean)]
    
    rows = []
    for label, d, col, fn in specs:
        d = d.dropna(subset=[col])
        groups = [g[col].to_numpy(float) for _, g in d.groupby("expiry")]
        draws = [fn(np.concatenate([rng.choice(g, g.size) for g in groups])) for _ in range(B)]
        rows.append(dict(statistic=label, n=len(d), estimate=float(fn(d[col].to_numpy(float))), ci_low=float(np.quantile(draws, 0.025)), ci_high=float(np.quantile(draws, 0.975))))
    
    df = pd.DataFrame(rows)
    df.to_csv(tabdir / "step5_f_bootstrap.csv", index=False)
    print("  (f) bootstrap 95% CIs (contracts within an expiry are NOT independent; intervals are too narrow):\n" + df.to_string(index=False))


def hypothesis_checks(a_tables, b_tables, c_tables, disc: float | None, tabdir: Path) -> None:
    """Evaluate the stated in advance thresholds of docs/hypotheses_step5.md section 3 (units: IV in decimals, 0.01 = one vol point)."""
    rows = []

    def add(h: str, what: str, value: float, threshold: str, surprising: bool) -> None:
        rows.append(dict(hypothesis=h, quantity=what, value=f"{value:.5f}", pre_registered_surprise_threshold=threshold, surprise=surprising))

    for name, t in a_tables.items():
        d = t.assign(diff=t["bs_mid"] - t["american_mid"])
        puts, calls = d[d.cp_flag == "P"], d[d.cp_flag == "C"]
        atm = puts[(puts.moneyness >= 0.98) & (puts.moneyness <= 1.02)]["diff"].median()
        otm = puts[puts.moneyness <= 0.95]["diff"].median()
        itm = puts[puts.moneyness >= 1.05]
        add("H1", f"{name} median put diff, ATM (K/S 0.98-1.02)", atm, "> 0.006", bool(atm > 0.006))
        add("H1", f"{name} median put diff, K/S <= 0.95", otm, "> 0.003", bool(otm > 0.003))
        add("H1", f"{name} fraction of ITM puts (K/S >= 1.05, n={len(itm)}) with diff > half-width", float((itm["diff"].abs() > itm["american_half"]).mean()) if len(itm) else math.nan,
            "not above half-width at all (0)", bool(len(itm) and (itm["diff"].abs() > itm["american_half"]).mean() == 0))
        mid_calls = calls[(calls.moneyness >= 0.9) & (calls.moneyness <= 1.1)]
        if name not in b_tables:  # H2 is about the q = 0 stock only; with dividends the American call legitimately differs from BS
            continue
        
        add("H2", f"{name} max |BS - American| for calls 0.9<=K/S<=1.1 (n={len(mid_calls)}); expected ~0 only if q=0", float(mid_calls["diff"].abs().max()), "> 0.001 (0.1 vol points)", bool(mid_calls["diff"].abs().max() > 1e-3))
    
    for name, t in b_tables.items():
        neg = t[(t.market_premium < 0) & (t.market_premium + t.band_halfwidth < 0)]
        add("H3", f"{name} strikes with market premium negative and entirely below zero by the band (of {len(t)})", float(len(neg)), "any far outside the band", bool(len(neg) > 0))
        add("H3", f"{name} mean market premium", float(t.market_premium.mean()), "negative", bool(t.market_premium.mean() < 0))
    
    for name, m in c_tables.items():
        add("H4", f"{name} mean |call-put IV gap|: American minus BS (negative = improvement)", float(m.gap_american.abs().mean() - m.gap_bs.abs().mean()), ">= 0 (no improvement)", bool(m.gap_american.abs().mean() >= m.gap_bs.abs().mean()))
    
    if disc is not None:
        half = float(np.nanmedian(a_tables[next(iter(c_tables))]["american_half"])) if c_tables else math.nan
        add("method", f"max |IV(400) - IV(1600)| in the discretization check (SPY); median half-width {half:.5f}", disc, "must be much smaller than the median half-width (here: > half-width flagged)", bool(disc > half))
    
    pd.DataFrame(rows).to_csv(tabdir / "step5_hypothesis_checks.csv", index=False)
    print("  hypothesis checks against the stated in advance thresholds:\n" + pd.DataFrame(rows).to_string(index=False))


def discretization_check(kept: pd.DataFrame, t: pd.DataFrame, tabdir: Path) -> float:
    """American IV at M=N=400 vs 1600 on a sample, in IV units, against the typical bid-ask half-width."""
    idx = np.linspace(0, len(kept) - 1, min(DISCRETIZATION_SAMPLE, len(kept))).astype(int)
    diffs = []
    for i in idx:
        row = kept.iloc[i]
        c = contract_of(row)
        a, _ = american_iv(float(row["mid"]), c, 400, 400)
        b, _ = american_iv(float(row["mid"]), c, 1600, 1600)
        if math.isfinite(a) and math.isfinite(b):
            diffs.append(abs(a - b))
    
    half = t["american_half"].median()
    pd.DataFrame([dict(sample=len(diffs), max_abs_iv_diff_400_vs_1600=max(diffs), median_abs_iv_diff=float(np.median(diffs)), median_bid_ask_iv_half_width=half,
                       ratio_max_diff_to_half_width=max(diffs) / half)]).to_csv(tabdir / "step5_discretization_check.csv", index=False)
    print(f"  discretization check ({len(diffs)} contracts): max |IV(400) - IV(1600)| = {max(diffs):.2e}; median bid-ask IV half-width {half:.2e} (ratio {max(diffs) / half:.3f})")

    return max(diffs)


def main() -> None:
    global PROVENANCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="run the pipeline on the synthetic fixture (outputs under results/synthetic/)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    args = ap.parse_args()
    if args.synthetic:
        unders, curve, quote_time = load_synthetic()
        base = ROOT / "results" / "synthetic"
        nd_name, div_name = "SYN_ND", "SYN_DIV"
        print("SYNTHETIC DATA: pipeline demo only, no market data")
        PROVENANCE = "source: synthetic fixture (no market data)"
    else:
        kv = parse_hypotheses(HYPOTHESES)
        unders, curve, quote_time = load_real(kv)
        base = ROOT / "results"
        nd_name, div_name = kv["ticker_nondividend"], kv["ticker_dividend"]
        print(f"data source: WRDS OptionMetrics Ivy DB; snapshot {kv['snapshot_date']} (quote time {quote_time}); pull stamps {[u.pull_stamp for u in unders]}")
        PROVENANCE = f"source: WRDS OptionMetrics Ivy DB, snapshot {kv['snapshot_date']} 16:00 ET close"
    
    figdir, tabdir = base / "figures", base / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)

    disc_value = None
    prepared, a_tables, b_tables, c_tables, waterfall, rejects, sens_rows = {}, {}, {}, {}, [], [], []
    dists = {}
    for u in unders:
        inputs = add_model_inputs(u.quotes, curve, u.distributions)
        kept, log = clean_quotes(inputs, CFG, u.load_rejects)
        n_loaded = log.attrs["n_loaded"]
        log.insert(0, "underlying", u.name)
        rejects.append(log)
        waterfall.append(dict(underlying=u.name, loaded_rows=n_loaded, **{f"after_{r.reason}": r.remaining for r in log.itertuples() if r.remaining is not None}))
        sens = spread_sensitivity(inputs, cfg=CFG)
        sens.insert(0, "underlying", u.name)
        sens_rows.append(sens)
        print(f"{u.name}: {n_loaded} loaded -> {len(kept)} contracts after cleaning ({len(kept[kept.cp_flag == 'P'])} puts)")
        print(log.dropna(subset=["remaining"]).to_string(index=False))
        prepared[u.name] = (kept, log)
        dists[u.name] = u.distributions
    
    pd.concat(rejects).to_csv(tabdir / "step5_reject_log.csv", index=False)
    pd.DataFrame(waterfall).to_csv(tabdir / "step5_data_summary.csv", index=False)
    pd.concat(sens_rows).to_csv(tabdir / "step5_cleaning_threshold_sensitivity.csv", index=False)
    print("cleaning-threshold sensitivity (spread 5%, 10%, 20%):\n" + pd.concat(sens_rows).to_string(index=False))

    for name, (kept, _) in prepared.items():
        t = iv_table(kept, args.workers)
        for kind in ("bs", "american"):
            print(f"{name}: {kind} IV at mid: {t[f'{kind}_reason'].value_counts().to_dict()}")
        
        a = analysis_a(t, name, figdir, tabdir)
        a_tables[name] = a
        if name == nd_name:
            b_tables[name] = analysis_b(kept, name, figdir, tabdir)
        
        if name == div_name:
            c_tables[name] = analysis_c(t, name, figdir, tabdir)
        
        if name == div_name:
            disc_value = discretization_check(kept, t, tabdir)
    
    per = {n: {"a": a_tables[n], **({"b": b_tables[n]} if n in b_tables else {}), **({"c": c_tables[n]} if n in c_tables else {})} for n in a_tables}
    analysis_d(prepared, curve, dists, args.workers, tabdir)
    analysis_vendor(a_tables, tabdir)
    analysis_bootstrap(a_tables, b_tables, c_tables, args.bootstrap, tabdir)
    hypothesis_checks(a_tables, b_tables, c_tables, disc_value, tabdir)


if __name__ == "__main__":
    main()