"""Option contract and market parameters."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal
import numpy as np


OptionType = Literal["put", "call"]


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
        option_type: "put" or "call"."""

    S: float
    K: float
    T: float
    r: float
    q: float
    sigma: float
    option_type: OptionType

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

    def payoff(self, s):
        """Intrinsic value at spot price(s) ``s`` (scalar or ndarray)."""
        sign = 1.0 if self.option_type == "call" else -1.0
        return np.maximum(sign * (np.asarray(s, dtype=float) - self.K), 0.0)