"""Option contract and market parameters."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal
import numpy as np


OptionType = Literal["put", "call"]
SPOT_FLOOR_REL = 1e-8


@dataclass(frozen=True)
class OptionParams:
    """Inputs for a simple option on an asset paying a continuous dividend yield.

    Attributes:
        S: spot price (> 0).
        K: strike (> 0).
        T: time to maturity in years (> 0).
        r: continuously compounded risk-free rate.
        q: continuous dividend yield.
        sigma: volatility (> 0).
        option_type: "put" or "call".
        dividends: discrete cash dividends as ((ex_time, amount), ...) with 0 < ex_time < T and amount > 0."""
    S: float
    K: float
    T: float
    r: float
    q: float
    sigma: float
    option_type: OptionType
    dividends: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        for name in ("S", "K", "T", "r", "q", "sigma"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
            
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
            
        for name in ("S", "K", "T", "sigma"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
            
        if self.option_type not in ("put", "call"):
            raise ValueError(f"option_type must be 'put' or 'call', got {self.option_type!r}")
        
        divs = tuple(sorted((float(t), float(d)) for t, d in self.dividends))
        for t, d in divs:
            if not (math.isfinite(t) and math.isfinite(d)) or not 0.0 < t < self.T or d <= 0.0:
                raise ValueError(f"dividends need 0 < ex_time < T={self.T} and amount > 0, got ({t}, {d})")
            
        if any(a[0] == b[0] for a, b in zip(divs, divs[1:])):
            raise ValueError("dividend ex-times must be distinct")
        
        object.__setattr__(self, "dividends", divs)

    def ex_dividend(self, s, amount: float):
        """Spot just after paying ``amount``: max(s - amount, SPOT_FLOOR_REL * K)."""
        return np.maximum(np.asarray(s, dtype=float) - amount, SPOT_FLOOR_REL * self.K)

    def dividend_pv(self, t: float = 0.0, inclusive: bool = False) -> float:
        """Present value at time t of dividends paid after t (also at t itself if ``inclusive``)."""
        eps = 1e-9 * self.T
        lo = t - eps if inclusive else t + eps

        return float(sum(d * math.exp(-self.r * (te - t)) for te, d in self.dividends if te > lo))

    def payoff(self, s):
        """Intrinsic value at spot price(s) ``s`` (scalar or ndarray)."""
        sign = 1.0 if self.option_type == "call" else -1.0
        
        return np.maximum(sign * (np.asarray(s, dtype=float) - self.K), 0.0)