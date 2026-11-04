"""Quantitative pre-registration for Step 5: BS-vs-American IV gaps measured with my own pricer on synthetic chains with
AMZN-like (no dividends, vol ~30%) and SPY-like (one quarterly dividend, vol ~16%) inputs. No market data is used."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from aopt.market.clean import add_model_inputs, contract_of
from aopt.market.dividends import load_distributions
from aopt.market.iv import american_iv, bs_iv
from aopt.market.rates import ZeroCurve
from aopt.market.synthetic import distributions_table, make_chain

QT = datetime(2026, 5, 13, 16, 0)
RATE = 0.04
CURVE = ZeroCurve(np.array([14.0, 365.0]), np.array([RATE, RATE]))
EXPIRY_DAYS = (14, 28, 42, 56, 65)  # the stated in advance targets (2 to ~9 weeks)
STRIKES = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15)
AMZN_LIKE = dict(base=0.30, skew=-0.10, smile=0.30, term=0.0)
SPY_LIKE = dict(base=0.16, skew=-0.20, smile=0.40, term=0.0)
SPY_DIVIDEND = ((36, 0.28),)  # one ex-date 36 days out, 0.28% of spot (about one quarterly SPY dividend)
OUT = Path(__file__).resolve().parents[1] / "results" / "tables"
MNY = [0.8, 0.9, 0.95, 1.05, 1.1, 1.2]


def gaps(name: str, surface: dict, divs: tuple) -> pd.DataFrame:
    chain = make_chain(strikes_over_spot=STRIKES, expiry_days=EXPIRY_DAYS, curve=CURVE, dividends_abs=divs, surface=surface, quote_time=QT, underlying=name)
    dist = distributions_table(QT, divs) if divs else load_distributions(pd.DataFrame(), None)
    d = add_model_inputs(chain, CURVE, dist)
    rows = []
    for _, r in d.iterrows():
        c = contract_of(r)
        am, _ = american_iv(r["true_mid"], c)
        bs, _ = bs_iv(r["true_mid"], c)
        rows.append(dict(underlying=name, cp=r.cp_flag, days=int(r.days), moneyness=r.strike / r.spot, am_iv=am, bs_iv=bs, gap_bs_minus_am=bs - am,
                         price=r.true_mid, premium=r.true_mid - r.true_mid_european))
        
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    amzn, spy = gaps("AMZN_like", AMZN_LIKE, ()), gaps("SPY_like", SPY_LIKE, SPY_DIVIDEND)
    allg = pd.concat([amzn, spy])
    allg.to_csv(OUT / "step5_predictions_gaps.csv", index=False)
    pd.set_option("display.width", 200)
    for name, g in (("AMZN_like", amzn), ("SPY_like", spy)):
        for cp in ("P", "C"):
            piv = g[g.cp == cp].pivot_table(index="moneyness", columns="days", values="gap_bs_minus_am") * 100.0
            print(f"\n{name} {'puts' if cp == 'P' else 'calls'}: BS IV - American IV, vol points (rows K/S, columns days)\n{piv.round(2).to_string()}")

    # H4: call-put IV gap at the same strike/expiry, BS vs American, on SPY-like inputs (true model = American, so the American gap is grid error)
    m = spy[spy.cp == "C"].merge(spy[spy.cp == "P"], on=["days", "moneyness"], suffixes=("_c", "_p"))
    m["gap_bs"] = (m.bs_iv_c - m.bs_iv_p) * 100
    m["gap_am"] = (m.am_iv_c - m.am_iv_p) * 100
    m["after_ex_date"] = m.days > 36
    print("\nSPY_like call IV - put IV, vol points: mean |gap| by group")
    print(m.groupby("after_ex_date")[["gap_bs", "gap_am"]].agg(lambda x: x.abs().mean()).round(3).to_string())
    print(m.pivot_table(index="moneyness", columns="days", values="gap_bs").round(2).to_string())
    m.to_csv(OUT / "step5_predictions_call_put.csv", index=False)
    print("\nAMZN_like call gap max |BS - American| (vol points):", round(float(amzn[amzn.cp == "C"].gap_bs_minus_am.abs().max() * 100), 3))
    print("AMZN_like put premium (price units, spot=100) by K/S and days:\n", amzn[amzn.cp == "P"].pivot_table(index="moneyness", columns="days", values="premium").round(3).to_string())


if __name__ == "__main__":
    main()