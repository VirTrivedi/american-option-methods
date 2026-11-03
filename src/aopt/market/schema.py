"""Standard quote schema and its validation."""
from __future__ import annotations
import numpy as np
import pandas as pd

QUOTE_COLUMNS = ["quote_time", "underlying", "expiry", "strike", "cp_flag", "bid", "ask", "volume", "open_interest", "spot",
                 "spot_time", "vendor_iv", "vendor_delta", "exercise_style", "am_settlement"]
# vendor_iv / vendor_delta are NaN when the vendor has no value; am_settlement is bool (True = AM-settled expiry)


def validate_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """Raise ValueError on any schema violation; returns df unchanged."""
    missing = [c for c in QUOTE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"quote table is missing columns {missing}")
    
    if df.empty:
        raise ValueError("quote table is empty")
    
    for col in ("quote_time", "spot_time", "expiry"):
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            raise ValueError(f"column {col} must be datetime64, got {df[col].dtype}")
    
    for col in ("strike", "bid", "ask", "spot", "volume", "open_interest"):
        if not np.isfinite(df[col].astype(float)).all():
            raise ValueError(f"column {col} contains NaN or inf")
    
    if not set(df["cp_flag"]) <= {"C", "P"}:
        raise ValueError(f"cp_flag must be 'C' or 'P', got {sorted(set(df['cp_flag']))}")
    
    if not set(df["exercise_style"]) <= {"A", "E"}:
        raise ValueError(f"exercise_style must be 'A' or 'E', got {sorted(set(df['exercise_style']))}")
    
    if df["am_settlement"].dtype != bool:
        raise ValueError("am_settlement must be a bool column")
    
    if (df["strike"] <= 0).any() or (df["spot"] <= 0).any():
        raise ValueError("strike and spot must be positive")
    
    if (df["bid"] < 0).any() or (df["ask"] < 0).any():
        raise ValueError("bid and ask must be non-negative (crossed or zero quotes are left to the cleaner)")
    
    if (df["spot_time"] > df["quote_time"]).any():
        raise ValueError("spot_time is after quote_time")
    
    if (df["expiry"].dt.normalize() < df["quote_time"].dt.normalize()).any():
        raise ValueError("expiry is before quote_time")

    return df