"""Formula 4 — The Kelly criterion: exactly how much should you bet?

    f* = (b*p - q) / b

Formula 3 said bet size decides survival.  Kelly says which size turns survival
into growth.  For the 55% even-money coin the answer is 10% of the bankroll —
and 10% of the *new* number after every settled bet, not 10% forever.

The practical content of this module is mostly about why you should stake less
than f* in a real sportsbook, and about the one adjustment that actually
matters: your p is estimated, the coin's was given.
"""

from __future__ import annotations

from math import log, sqrt

from .expectancy import expectancy
from .odds import decimal_to_prob, net_odds
from .volatility import bet_variance, norm_cdf, norm_ppf

__all__ = [
    "kelly_fraction",
    "kelly_from_odds",
    "fractional_kelly",
    "kelly_stake",
    "kelly_quantile",
    "probability_edge_is_real",
    "kelly_variance_form",
    "growth_fraction_of_max",
    "implied_kelly_multiple",
]


def kelly_fraction(p: float, b: float = 1.0) -> float:
    """``f* = (b*p - q) / b`` — the growth-optimal fraction of bankroll.

    Returns 0 for a non-positive edge: Kelly's answer to a bad bet is not a
    small bet, it is no bet.  Note the numerator is exactly the expectancy from
    formula 1, so ``f* = edge / b``: the optimal stake is your edge divided by
    the price.  Longer prices demand smaller stakes for the same edge.
    """
    b = float(b)
    if b <= 0.0:
        raise ValueError(f"Net odds must be > 0, got {b!r}")
    q = 1.0 - p
    f = (b * p - q) / b
    return max(0.0, f)


def kelly_from_odds(p: float, decimal_odds: float) -> float:
    """Kelly fraction straight from a quoted decimal price."""
    return kelly_fraction(p, net_odds(decimal_odds))


def fractional_kelly(p: float, b: float = 1.0, multiple: float = 1.0) -> float:
    """Stake a fixed multiple of full Kelly — the standard practitioner move.

    Half Kelly gives up 25% of the growth rate to remove about 87% of the
    drawdown risk (see :func:`edgemodel.ruin.drawdown_probability`).  That is
    the trade nearly every professional takes, and it is not timidity: the
    growth curve is flat near its peak and the risk curve is not.
    """
    return kelly_fraction(p, b) * max(0.0, float(multiple))


def kelly_stake(
    bankroll: float,
    p: float,
    decimal_odds: float,
    multiple: float = 1.0,
    cap_fraction: float | None = None,
    round_to: float | None = None,
) -> float:
    """Currency stake for a bet, with the caps a real staking plan needs."""
    f = fractional_kelly(p, net_odds(decimal_odds), multiple)
    if cap_fraction is not None:
        f = min(f, float(cap_fraction))
    stake = max(0.0, float(bankroll) * f)
    if round_to:
        stake = round(stake / round_to) * round_to
    return stake


def kelly_variance_form(p: float, b: float = 1.0) -> float:
    """Kelly as ``edge / variance`` — the continuous approximation.

    ``f ~= mu / sigma^2`` is the form that generalises to portfolios and makes
    the link to formula 5 explicit: the optimal stake is the edge divided by
    the very variance that formula 5 subtracts from your growth.  For the coin
    it gives 0.101 against an exact 0.100.
    """
    var = bet_variance(p, b)
    if var <= 0.0:
        return 0.0
    return max(0.0, expectancy(p, b) / var)


def probability_edge_is_real(
    p_mean: float, p_sd: float, decimal_odds: float
) -> float:
    """P(the true win probability actually clears the breakeven price).

    The single most clarifying number for a sports model.  A 54% estimate at
    -110 (breakeven 52.4%) with a 2-point standard error is only a 79% chance
    of being a real edge at all — and a 21% chance you are the gambler holding
    a coin tilted the wrong way without knowing it.
    """
    breakeven = decimal_to_prob(decimal_odds)
    if p_sd <= 0.0:
        return 1.0 if p_mean > breakeven else 0.0
    return 1.0 - norm_cdf((breakeven - p_mean) / p_sd)


def kelly_quantile(
    p_mean: float, p_sd: float, decimal_odds: float, quantile: float = 0.25
) -> float:
    """Kelly sized off a conservative quantile of your probability estimate.

    A correction to the usual folklore.  Averaging growth over parameter
    uncertainty does *not* move the optimum: ``g(f, p)`` is linear in p, so the
    expectation is just ``g(f, E[p])`` and the posterior mean is still
    growth-optimal.  Uncertainty does something else — it widens the
    distribution of *realised* growth, and it puts real mass on the region
    where your true edge is negative and full Kelly is a leveraged bet on a
    losing proposition.

    So the honest adjustment is not to average, it is to size off a downside
    quantile of p.  Betting the 25th percentile means the stake is right or
    conservative three times in four, and the shortfall when you are wrong is
    bounded.  This is where fractional Kelly comes from in practice, and it
    also explains why it should be *tighter* on markets you model worse rather
    than uniformly.
    """
    if p_sd <= 0.0:
        return kelly_from_odds(p_mean, decimal_odds)
    p_conservative = p_mean + norm_ppf(quantile) * p_sd
    p_conservative = min(max(p_conservative, 1e-9), 1.0 - 1e-9)
    return kelly_from_odds(p_conservative, decimal_odds)


def growth_fraction_of_max(multiple: float) -> float:
    """Share of peak growth retained at k times Kelly, in the small-edge limit.

    ``g(k) / g(1) = k * (2 - k)``.  Half Kelly keeps 75% of the growth, quarter
    Kelly keeps 44%, and double Kelly keeps exactly none.  This is the
    quadratic that makes fractional Kelly cheap and over-betting ruinous.
    """
    k = float(multiple)
    return k * (2.0 - k)


def implied_kelly_multiple(stake_fraction: float, p: float, b: float = 1.0) -> float:
    """How many Kelly units a given stake actually represents.

    Point it at your current staking plan.  Flat-staking 5% of a bankroll on a
    1% edge at -110 is over three times Kelly, which is past the zero-growth
    wall — the honest diagnosis of most losing systems with a real edge.
    """
    f_star = kelly_fraction(p, b)
    if f_star <= 0.0:
        return float("inf") if stake_fraction > 0 else 0.0
    return float(stake_fraction) / f_star
