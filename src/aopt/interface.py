"""Uniform entry point for the American option pricers."""
from __future__ import annotations
from typing import Any
from .params import OptionParams
from .pde import solve_pde
from .reference import reference_price
from .tree import tree_price


METHODS = ("tree", "tree_reference", "pde", "lsm")


def price(params: OptionParams, method: str = "tree", **kwargs: Any) -> float:
    """Price an American option (CRR tree, extrapolated tree, pde, or lsm)."""
    if method == "tree":
        return tree_price(params, **kwargs)
    if method == "tree_reference":
        return reference_price(params, **kwargs).price
    if method == "pde":
        return solve_pde(params, **kwargs).price
    if method == "lsm":
        raise NotImplementedError("method 'lsm' is not implemented yet")
    raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")