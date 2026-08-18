"""Odds arithmetic: the translation layer between how books quote a price and
how the five formulas need it expressed.

Every formula in this library wants two numbers: a probability ``p`` and net
odds ``b`` (profit per unit staked on a win).  Sportsbooks quote neither.  They
quote American (-110), decimal (1.909) or fractional prices that already have
the house margin baked in.  This module converts between them and measures the
margin.
"""

from __future__ import annotations

from math import isfinite

__all__ = [
    "american_to_decimal",
    "decimal_to_american",
    "decimal_to_prob",
    "prob_to_decimal",
    "prob_to_american",
    "american_to_prob",
    "net_odds",
    "decimal_from_net",
    "parse_odds",
    "overround",
    "hold",
    "breakeven_prob",
    "fair_decimal",
]


def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds. ``-110 -> 1.9091``."""
    a = float(american)
    if -100.0 < a < 100.0:
        raise ValueError(
            f"American odds must be <= -100 or >= +100, got {american!r}"
        )
    if a > 0:
        return 1.0 + a / 100.0
    return 1.0 + 100.0 / abs(a)


def decimal_to_american(decimal_odds: float) -> float:
    """Convert decimal odds to American odds. ``1.9091 -> -110``."""
    d = float(decimal_odds)
    if d <= 1.0:
        raise ValueError(f"Decimal odds must be > 1.0, got {decimal_odds!r}")
    if d >= 2.0:
        return (d - 1.0) * 100.0
    return -100.0 / (d - 1.0)


def decimal_to_prob(decimal_odds: float) -> float:
    """Implied probability of a decimal price, margin included."""
    d = float(decimal_odds)
    if d <= 1.0:
        raise ValueError(f"Decimal odds must be > 1.0, got {decimal_odds!r}")
    return 1.0 / d


def prob_to_decimal(p: float) -> float:
    """Fair decimal price for a probability (zero margin)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"Probability must be in (0, 1), got {p!r}")
    return 1.0 / p


def prob_to_american(p: float) -> float:
    """Fair American price for a probability (zero margin)."""
    return decimal_to_american(prob_to_decimal(p))


def american_to_prob(american: float) -> float:
    """Implied probability of an American price, margin included."""
    return decimal_to_prob(american_to_decimal(american))


def net_odds(decimal_odds: float) -> float:
    """``b`` in the Kelly/expectancy formulas: profit per unit risked."""
    return float(decimal_odds) - 1.0


def decimal_from_net(b: float) -> float:
    """Inverse of :func:`net_odds`."""
    if b <= 0.0:
        raise ValueError(f"Net odds must be > 0, got {b!r}")
    return 1.0 + float(b)


def parse_odds(value, fmt: str = "auto") -> float:
    """Read a price in whatever notation it arrived in, return decimal odds.

    ``fmt`` may be ``"auto"``, ``"american"``, ``"decimal"`` or ``"fractional"``.
    Auto-detection uses the fact that American prices are always outside
    (-100, +100) while decimal prices sit between 1 and 100.
    """
    if isinstance(value, str):
        text = value.strip()
        if "/" in text:
            num, _, den = text.partition("/")
            return 1.0 + float(num) / float(den)
        if fmt == "auto" and text and text[0] in "+-":
            return american_to_decimal(float(text))
        value = float(text)

    v = float(value)
    if not isfinite(v):
        raise ValueError(f"Odds must be finite, got {value!r}")

    if fmt == "american":
        return american_to_decimal(v)
    if fmt == "decimal":
        if v <= 1.0:
            raise ValueError(f"Decimal odds must be > 1.0, got {value!r}")
        return v
    if fmt == "fractional":
        return 1.0 + v
    if fmt != "auto":
        raise ValueError(f"Unknown odds format {fmt!r}")

    if abs(v) >= 100.0:
        return american_to_decimal(v)
    if v > 1.0:
        return v
    raise ValueError(
        f"Ambiguous odds {value!r}: decimal odds must exceed 1.0 and American "
        "odds must be at least 100 in magnitude. Pass fmt= to disambiguate."
    )


def overround(decimal_prices) -> float:
    """Total implied probability minus 1 — the book's margin on a market.

    A two-way market priced -110/-110 sums to 1.0476, i.e. a 4.76% overround.
    """
    total = sum(decimal_to_prob(d) for d in decimal_prices)
    return total - 1.0


def hold(decimal_prices) -> float:
    """The book's expected revenue per dollar handled on a balanced book.

    Distinct from the overround: -110/-110 is a 4.76% overround but only a
    4.55% hold, because the margin is collected on the larger, grossed-up pool.
    """
    total = sum(decimal_to_prob(d) for d in decimal_prices)
    return (total - 1.0) / total


def breakeven_prob(decimal_odds: float) -> float:
    """Win rate you need just to break even at this price.

    At -110 this is 52.38%.  Every edge calculation is measured against this
    line, not against 50%.
    """
    return decimal_to_prob(decimal_odds)


def fair_decimal(p: float) -> float:
    """Alias of :func:`prob_to_decimal`, reads better at call sites."""
    return prob_to_decimal(p)
