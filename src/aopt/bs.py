"""Black-Scholes-Merton European prices and deltas with continuous dividend yield."""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr
from .params import OptionType


def d1_d2(S, K, T, r, q, sigma):
    S, K, T, r, q, sigma = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma)))
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return S, K, T, r, q, d1, d1 - vol_sqrt_t


def bs_price(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike, sigma: ArrayLike, option_type: OptionType) -> NDArray[np.float64]:
    """European option price; requires T, sigma > 0."""
    S, K, T, r, q, d1, d2 = d1_d2(S, K, T, r, q, sigma)
    fwd_s, disc_k = S * np.exp(-q * T), K * np.exp(-r * T)
    if option_type == "call":
        return fwd_s * ndtr(d1) - disc_k * ndtr(d2)
    if option_type == "put":
        return disc_k * ndtr(-d2) - fwd_s * ndtr(-d1)
    raise ValueError(f"option_type must be 'put' or 'call', got {option_type!r}")


def bs_delta(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike, sigma: ArrayLike, option_type: OptionType) -> NDArray[np.float64]:
    """European option delta dV/dS."""
    _, _, T, _, q, d1, _ = d1_d2(S, K, T, r, q, sigma)
    if option_type == "call":
        return np.exp(-q * T) * ndtr(d1)
    if option_type == "put":
        return -np.exp(-q * T) * ndtr(-d1)
    raise ValueError(f"option_type must be 'put' or 'call', got {option_type!r}")


def bs_gamma(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike, sigma: ArrayLike) -> NDArray[np.float64]:
    """European option gamma d2V/dS2."""
    S, _, T, _, q, d1, _ = d1_d2(S, K, T, r, q, sigma)
    sigma = np.asarray(sigma, dtype=float)
    pdf = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    return np.exp(-q * T) * pdf / (S * sigma * np.sqrt(T))