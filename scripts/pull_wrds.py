"""Pull one OptionMetrics Ivy DB snapshot from WRDS into data/raw/."""
from __future__ import annotations
import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# column names
OPPRCD_COLUMNS = ("secid, date, symbol, optionid, exdate, last_date, cp_flag, strike_price, best_bid, best_offer, volume, open_interest, "
                   "impl_volatility, delta, ss_flag, am_settlement, contract_size, cfadj, forward_price")


def secid_for(db, ticker: str, snapshot: date) -> int:
    """secid of the security with this ticker on the snapshot date (optionm.secnmd name history)."""
    df = db.raw_sql("select secid, ticker, issuer, effect_date from optionm.secnmd where ticker = %(t)s and effect_date <= %(d)s "
                    "order by effect_date desc limit 1", params={"t": ticker, "d": snapshot})
    if df.empty:
        raise SystemExit(f"no secid found for ticker {ticker} on {snapshot}")
    
    print(f"{ticker}: secid {int(df['secid'].iloc[0])} ({df['issuer'].iloc[0]})")

    return int(df["secid"].iloc[0])


def pull(db, ticker: str, snapshot: date, min_days: int, max_days: int, margin_days: int, out: Path, stamp: str) -> None:
    year, secid = snapshot.year, secid_for(db, ticker, snapshot)
    first, last = snapshot + timedelta(days=min_days), snapshot + timedelta(days=max_days)
    opts = db.raw_sql(f"select {OPPRCD_COLUMNS} from optionm.opprcd{year} where secid = %(s)s and date = %(d)s and exdate between %(a)s and %(b)s",
                      params={"s": secid, "d": snapshot, "a": first, "b": last}, date_cols=["date", "exdate"])
    px = db.raw_sql(f"select * from optionm.secprd{year} where secid = %(s)s and date = %(d)s", params={"s": secid, "d": snapshot}, date_cols=["date"])
    distr = db.raw_sql("select * from optionm.distrd where secid = %(s)s and ex_date between %(a)s and %(b)s",
                       params={"s": secid, "a": snapshot, "b": last + timedelta(days=margin_days)}, date_cols=["ex_date"])
    info = db.raw_sql("select * from optionm.opinfd where secid = %(s)s", params={"s": secid})
    for kind, df in (("opprcd", opts), ("secprd", px), ("distrd", distr), ("opinfd", info)):
        path = out / f"{kind}_{ticker}_{snapshot}_{stamp}.csv"
        df.to_csv(path, index=False)
        print(f"  wrote {len(df):>6} rows -> {path.name}")
    
    if opts.empty or px.empty:
        raise SystemExit(f"{ticker}: empty option or price pull; check the date (trading day?) and the expiry window")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, type=date.fromisoformat, help="snapshot date YYYY-MM-DD")
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--min-days", type=int, default=14, help="shortest expiry to pull, calendar days after the snapshot")
    ap.add_argument("--max-days", type=int, default=190, help="longest expiry to pull")
    ap.add_argument("--margin-days", type=int, default=10, help="distributions are pulled this long past the last expiry")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "raw")
    ap.add_argument("--username", default=None, help="WRDS username (not a secret); the password comes from ~/.pgpass")
    args = ap.parse_args()
    try:
        import wrds
    except ImportError:
        raise SystemExit("pip install wrds (or `pip install -e .[wrds]`) first")
    
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    db = wrds.Connection(wrds_username=args.username) if args.username else wrds.Connection()
    try:
        curve = db.raw_sql("select date, days, rate from optionm.zerocd where date = %(d)s order by days", params={"d": args.date}, date_cols=["date"])
        if curve.empty:
            raise SystemExit(f"no zero curve for {args.date}")
    
        path = args.out / f"zerocd_{args.date}_{stamp}.csv"
        curve.to_csv(path, index=False)
        print(f"zero curve: {len(curve)} rows -> {path.name}")
        for ticker in args.tickers:
            pull(db, ticker, args.date, args.min_days, args.max_days, args.margin_days, args.out, stamp)
    finally:
        db.close()


if __name__ == "__main__":
    main()