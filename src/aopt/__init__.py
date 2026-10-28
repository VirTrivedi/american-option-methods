"""aopt: American option pricing methods."""

from .interface import price
from .params import OptionParams

__all__ = ["OptionParams", "price"]
