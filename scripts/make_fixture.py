"""Write the tiny synthetic chain fixture used by the tests and by `step5_market.py --synthetic`."""
from datetime import datetime
from pathlib import Path
import numpy as np
from aopt.market.rates import ZeroCurve
from aopt.market.synthetic import distributions_table, make_chain

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
DIVIDENDS = ((50, 0.8), (140, 0.8))  # (days from the quote date, amount)
SEED = 20240315


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    curve = ZeroCurve(np.array([30.0, 365.0]), np.array([0.045, 0.05]))
    chain = make_chain(dividends_abs=DIVIDENDS, curve=curve, rel_spread=0.02, noise_rel=0.002, tick=0.01, seed=SEED, strikes_over_spot=(0.9, 0.95, 1.0, 1.05, 1.1), expiry_days=(35, 91, 182))
    chain.to_csv(OUT / "synthetic_chain.csv", index=False)
    distributions_table(datetime(2024, 3, 15, 16), DIVIDENDS).to_csv(OUT / "synthetic_distributions.csv", index=False)
    print(f"wrote {len(chain)} rows to {OUT}")


if __name__ == "__main__":
    main()