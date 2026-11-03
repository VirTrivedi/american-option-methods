"""ACT/365 time to expiry. All timestamps are naive and in US/Eastern wall-clock time."""
from __future__ import annotations
from datetime import date, datetime, time

PM_CLOSE = time(16, 0)  # PM-settled expiries stop trading at the 16:00 ET close
AM_OPEN = time(9, 30)  # AM-settled expiries settle on the opening prices of the expiry date
EX_DATE_TIME = time(9, 30)  # the spot drops by the dividend at the open of the ex-date


def expiry_timestamp(expiry: date, am_settled: bool) -> datetime:
    """Moment the option stops existing: 09:30 ET for AM-settled, 16:00 ET otherwise."""
    return datetime.combine(expiry, AM_OPEN if am_settled else PM_CLOSE)


def year_fraction(start: datetime, end: datetime) -> float:
    """ACT/365 year fraction between two timestamps."""
    return (end - start).total_seconds() / (365.0 * 86400.0)