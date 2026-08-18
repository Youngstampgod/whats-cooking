"""Formula 1 — Expectancy: is there an edge at all?

    E = (W x A) - (L x B)

The entrance ticket.  A positive number here says you are allowed to play; it
says nothing whatsoever about how much to bet or whether you will survive long
enough to collect.  Everything in the other four modules exists because this
one cannot answer those questions.
"""

from __future__ import annotations

from .odds import decimal_to_prob, net_odds

__all__ = [
    "expectancy",
    "edge",
    "expected_value",
    "breakeven_win_rate",
    "edge_over_market",
    "required_win_rate",
    "expectancy_general",
]


def expectancy_general(
    win_rate: float, avg_win: float, loss_rate: float | None = None, avg_loss: float = 1.0
) -> float:
    """``E = (W x A) - (L x B)`` in its raw form.

    Used when wins and losses have varying sizes (part-settled bets, pushes
    folded into the average, a mixed book of prices).  ``loss_rate`` defaults
    to ``1 - win_rate``.
    """
    W = float(win_rate)
    L = float(loss_rate) if loss_rate is not None else 1.0 - W
    return (W * float(avg_win)) - (L * float(avg_loss))


def expectancy(p: float, b: float = 1.0) -> float:
    """Expected profit per dollar staked on a single binary bet.

    ``p`` is the true win probability, ``b`` the net odds.  This is the
    even-money form ``E = (0.55 x 1) - (0.45 x 1) = +0.10`` generalised to any
    price: a bet is only as good as the price you got on it.
    """
    _check_prob(p)
    return p * float(b) - (1.0 - p)


#: The Kelly numerator and the per-dollar expectancy are the same quantity.
edge = expectancy


def expected_value(stake: float, p: float, decimal_odds: float) -> float:
    """Expected profit in currency for a specific stake at a specific price."""
    return float(stake) * expectancy(p, net_odds(decimal_odds))


def breakeven_win_rate(decimal_odds: float) -> float:
    """The win rate at which expectancy is exactly zero.

    At -110 this is 52.38%.  Beating a coin flip is not an edge; beating this
    number is.
    """
    return decimal_to_prob(decimal_odds)


def required_win_rate(decimal_odds: float, target_roi: float) -> float:
    """Win rate needed to hit a target return per dollar staked."""
    b = net_odds(decimal_odds)
    return (target_roi + 1.0) / (b + 1.0)


def edge_over_market(p_model: float, p_market: float) -> float:
    """Probability points by which the model disagrees with the fair market.

    Reported separately from expectancy because they answer different
    questions: this one is "how far from consensus am I?", expectancy is "does
    the price I can actually get pay me for it?"
    """
    _check_prob(p_model)
    _check_prob(p_market)
    return p_model - p_market


def _check_prob(p: float) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Probability must be in [0, 1], got {p!r}")
