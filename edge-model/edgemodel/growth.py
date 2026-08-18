"""Formula 5 — Geometric growth: why the average is a lie.

    g ~= mu - sigma^2 / 2

There are two averages hiding behind the word.  A thousand copies of you
placing the bet have a mean ending bankroll that grows at any stake size,
because a handful of wildly lucky copies drag it up.  You are not a thousand
copies.  You walk one path, and that path compounds at the geometric rate:
the average *minus half the variance*.

Variance is not the discomfort on the way to the return.  It is a subtraction
from it.
"""

from __future__ import annotations

from math import exp, log

from .expectancy import expectancy
from .kelly import kelly_fraction
from .volatility import bet_variance

__all__ = [
    "log_growth_rate",
    "growth_approx",
    "growth_curve",
    "zero_growth_fraction",
    "doubling_time",
    "ensemble_vs_time",
    "growth_at_kelly_multiple",
]


def log_growth_rate(f: float, p: float, b: float = 1.0) -> float:
    """``g(f) = p*ln(1 + b*f) + q*ln(1 - f)`` — exact compounding rate per bet.

    This is the number that ends the bet-size argument.  Not the average dollar
    outcome: the rate your money actually compounds along the path you live.
    """
    f = float(f)
    if f < 0.0:
        raise ValueError(f"Bet fraction must be >= 0, got {f!r}")
    if f >= 1.0:
        return float("-inf")  # One loss and you are done.
    q = 1.0 - p
    return p * log(1.0 + b * f) + q * log(1.0 - f)


def growth_approx(f: float, p: float, b: float = 1.0) -> float:
    """``g ~= mu - sigma^2/2`` applied to a stake of fraction f.

    Staking f scales both the mean and the standard deviation of the bet by f,
    so ``g ~= f*mu - f^2*sigma^2/2``.  On the 55% coin at f = 0.10 this gives
    0.00505 against an exact 0.00501 — and maximising it reproduces Kelly as
    ``f = mu / sigma^2``.  Formulas 2, 4 and 5 are three views of one quantity.
    """
    mu = expectancy(p, b)
    var = bet_variance(p, b)
    return f * mu - 0.5 * f * f * var


def growth_at_kelly_multiple(multiple: float, p: float, b: float = 1.0) -> float:
    """Exact growth rate when staking k times the Kelly fraction."""
    return log_growth_rate(kelly_fraction(p, b) * multiple, p, b)


def growth_curve(p: float, b: float = 1.0, fractions=None) -> list[tuple[float, float]]:
    """(stake fraction, growth rate) pairs — the curve you cannot unsee."""
    if fractions is None:
        fractions = [i / 100.0 for i in range(0, 100)]
    return [(f, log_growth_rate(f, p, b)) for f in fractions]


def zero_growth_fraction(p: float, b: float = 1.0) -> float:
    """The stake at which growth crosses back through zero.

    Everything you risk beyond this point is compounding you backwards on a bet
    you are right about.  It sits at almost exactly twice Kelly (0.1983 against
    a Kelly of 0.10 for the coin), which is where the "2x Kelly buys nothing"
    rule comes from.
    """
    f_star = kelly_fraction(p, b)
    if f_star <= 0.0:
        return 0.0
    lo, hi = f_star, 1.0 - 1e-12
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if log_growth_rate(mid, p, b) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    return 0.5 * (lo + hi)


def doubling_time(growth_rate: float) -> float:
    """Bets required to double the bankroll at a given growth rate."""
    if growth_rate <= 0.0:
        return float("inf")
    return log(2.0) / growth_rate


def ensemble_vs_time(f: float, p: float, b: float = 1.0, n: int = 1000) -> dict:
    """The two averages, side by side, after n bets.

    ``mean`` is the ensemble average across every parallel copy of you —
    ``(1 + f*mu)^n``, which rises at *any* stake size and is the number that
    looks magnificent in a pitch deck.  ``median`` is the typical single path,
    ``exp(n*g)``.  Over-bet far enough and the mean goes to infinity while the
    median goes to zero, on the same bet, at the same time.
    """
    mu = expectancy(p, b)
    g = log_growth_rate(f, p, b)
    mean = (1.0 + f * mu) ** n
    median = exp(g * n) if g > -700.0 / max(n, 1) else 0.0
    return {
        "bets": n,
        "fraction": f,
        "growth_rate": g,
        "ensemble_mean": mean,
        "median_path": median,
        "mean_over_median": (mean / median) if median > 0 else float("inf"),
    }
