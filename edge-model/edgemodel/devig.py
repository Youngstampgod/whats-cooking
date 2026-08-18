"""Removing the vig: turning quoted prices into a probability estimate.

The coin in the parable arrives with p = 0.55 stamped on it.  A sportsbook
market arrives with the margin baked in, so the two sides of a game sum to
more than 100%.  Before you can ask "do I have an edge?" you have to strip the
margin out and recover what the market actually believes.

Four standard methods are implemented.  They disagree most on lopsided markets
(heavy favourites, longshots), which is exactly where retail models most often
think they have found an edge and have instead found a de-vig artefact.
"""

from __future__ import annotations

from typing import Sequence

from .odds import decimal_to_prob, prob_to_decimal

__all__ = ["devig", "no_vig_prices", "DEVIG_METHODS", "shin_z"]

DEVIG_METHODS = ("multiplicative", "additive", "power", "shin")

_TOL = 1e-12
_MAX_ITER = 200


def _bisect(fn, lo: float, hi: float, target: float = 0.0) -> float:
    """Solve fn(x) = target on [lo, hi] for a monotone fn."""
    f_lo = fn(lo) - target
    f_hi = fn(hi) - target
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if f_lo * f_hi > 0.0:
        # No sign change: return whichever endpoint is closer to the target.
        return lo if abs(f_lo) < abs(f_hi) else hi
    for _ in range(_MAX_ITER):
        mid = 0.5 * (lo + hi)
        f_mid = fn(mid) - target
        if abs(f_mid) < _TOL or (hi - lo) < _TOL:
            return mid
        if f_lo * f_mid < 0.0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def _multiplicative(raw: Sequence[float]) -> list[float]:
    """Scale every implied probability down by the same factor."""
    total = sum(raw)
    return [r / total for r in raw]


def _additive(raw: Sequence[float]) -> list[float]:
    """Subtract the margin equally from every outcome."""
    n = len(raw)
    excess = (sum(raw) - 1.0) / n
    out = [r - excess for r in raw]
    if any(p <= 0.0 for p in out):
        # Degenerate on very lopsided markets; fall back rather than emit junk.
        return _multiplicative(raw)
    return out


def _power(raw: Sequence[float]) -> list[float]:
    """Raise every implied probability to a common power k so they sum to 1.

    Because probabilities are below 1, k > 1 shrinks longshots harder than
    favourites, which matches the empirical shape of bookmaker margin.
    """

    def total(k: float) -> float:
        return sum(r ** k for r in raw)

    k = _bisect(total, 1.0, 50.0, target=1.0)
    return [r ** k for r in raw]


def shin_z(raw: Sequence[float]) -> float:
    """Fit Shin's insider-trading parameter z for a market.

    z is the implied fraction of money coming from bettors who know the
    outcome.  It is what forces the book to widen prices on longshots.
    """
    pi = sum(raw)

    def total(z: float) -> float:
        return sum(_shin_prob(r, z, pi) for r in raw)

    return _bisect(total, 0.0, 1.0 - 1e-9, target=1.0)


def _shin_prob(r: float, z: float, pi: float) -> float:
    if z <= 0.0:
        return r / (pi ** 0.5)
    if z >= 1.0:
        return r * r / pi
    disc = z * z + 4.0 * (1.0 - z) * (r * r) / pi
    return ((disc ** 0.5) - z) / (2.0 * (1.0 - z))


def _shin(raw: Sequence[float]) -> list[float]:
    pi = sum(raw)
    z = shin_z(raw)
    probs = [_shin_prob(r, z, pi) for r in raw]
    total = sum(probs)
    # Renormalise away the residual left by the numerical root find.
    return [p / total for p in probs]


_METHODS = {
    "multiplicative": _multiplicative,
    "additive": _additive,
    "power": _power,
    "shin": _shin,
}


def devig(decimal_prices: Sequence[float], method: str = "multiplicative") -> list[float]:
    """Strip the margin from a complete market, returning fair probabilities.

    ``decimal_prices`` must cover every outcome of the market — both sides of a
    spread, all three of a soccer 1X2 — otherwise the margin cannot be located.
    """
    if method not in _METHODS:
        raise ValueError(
            f"Unknown de-vig method {method!r}, expected one of {DEVIG_METHODS}"
        )
    prices = [float(d) for d in decimal_prices]
    if len(prices) < 2:
        raise ValueError(
            "De-vigging needs every outcome in the market, got "
            f"{len(prices)} price(s)"
        )
    raw = [decimal_to_prob(d) for d in prices]
    if sum(raw) < 1.0:
        raise ValueError(
            f"Prices sum to {sum(raw):.4f} < 1: that is an arbitrage, not a "
            "market with margin. Check the inputs."
        )
    return _METHODS[method](raw)


def no_vig_prices(
    decimal_prices: Sequence[float], method: str = "multiplicative"
) -> list[float]:
    """The same market re-quoted with zero margin — the 'fair line'."""
    return [prob_to_decimal(p) for p in devig(decimal_prices, method=method)]
