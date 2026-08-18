"""Formula 3 — Risk of ruin: can the noise kill you before the edge pays?

    R = (q / p) ^ N

Ruin is a race between the edge and the variance, and the thing that sets the
odds of that race is not the coin — it is how much you bet.  Measure the
bankroll in units, where one unit is one bet.

    q/p = 0.82  ->  N = 4 units: ruin 45%   ·   N = 20 units: ruin under 2%

Same coin, same edge, same side.  The only variable is bet size.

Two regimes are covered, because they behave completely differently:

* **Flat betting** — a fixed stake every time.  Ruin is a real absorbing state
  and the formulas above apply.
* **Proportional (Kelly) betting** — stake a fraction of the current bankroll.
  You mathematically never reach zero, so ruin is replaced by *drawdown*: the
  probability of ever being down to some fraction of your peak.
"""

from __future__ import annotations

from math import exp, log

__all__ = [
    "risk_of_ruin_even_money",
    "risk_of_ruin",
    "lundberg_exponent",
    "units_for_ruin_target",
    "max_bet_fraction_for_ruin",
    "drawdown_probability",
    "drawdown_for_probability",
    "ruin_table",
]


def risk_of_ruin_even_money(p: float, units: float) -> float:
    """``R = (q/p)^N`` — the classic gambler's ruin on an even-money bet.

    ``units`` is bankroll divided by stake: how many losing bets in a row the
    bankroll can absorb.  Fractional values are allowed and interpolate
    sensibly.
    """
    q = 1.0 - p
    if p <= q:
        return 1.0
    if units <= 0.0:
        return 1.0
    return (q / p) ** units


def lundberg_exponent(p: float, b: float = 1.0) -> float:
    """The decay rate of ruin probability per unit of bankroll.

    Solves ``E[exp(-L*X)] = 1`` for L > 0, where X is the per-unit result of
    one bet.  Ruin then falls off as ``exp(-L * units)``.  For an even-money
    bet this collapses exactly to ``L = ln(p/q)``, which reproduces
    ``R = (q/p)^N``; for any other price it is the correct generalisation.
    """
    q = 1.0 - p
    mu = p * b - q
    if mu <= 0.0:
        return 0.0  # No positive root: ruin is certain.

    def f(lam: float) -> float:
        return p * exp(-lam * b) + q * exp(lam) - 1.0

    lo, hi = 1e-12, 1.0
    for _ in range(200):
        if f(hi) > 0.0:
            break
        hi *= 2.0
        if hi > 1e6:
            return 0.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-14:
            break
    return 0.5 * (lo + hi)


def risk_of_ruin(p: float, b: float = 1.0, units: float = 20.0) -> float:
    """Probability of ever going broke while flat-betting at any price.

    Exact for even money, and the standard Lundberg bound elsewhere (it is
    conservative — real ruin is slightly lower — because it ignores the fact
    that you overshoot zero rather than landing exactly on it).
    """
    if units <= 0.0:
        return 1.0
    lam = lundberg_exponent(p, b)
    if lam <= 0.0:
        return 1.0
    return min(1.0, exp(-lam * units))


def units_for_ruin_target(p: float, b: float = 1.0, target: float = 0.01) -> float:
    """How many units of bankroll you need to hold ruin under ``target``.

    This is the formula run backwards, and it is the one worth memorising: it
    turns a risk appetite directly into a bet size.
    """
    if not 0.0 < target < 1.0:
        raise ValueError(f"Target ruin must be in (0, 1), got {target!r}")
    lam = lundberg_exponent(p, b)
    if lam <= 0.0:
        return float("inf")
    return -log(target) / lam


def max_bet_fraction_for_ruin(p: float, b: float = 1.0, target: float = 0.01) -> float:
    """The largest flat stake, as a fraction of bankroll, meeting a ruin cap."""
    units = units_for_ruin_target(p, b, target)
    if units == float("inf") or units <= 0.0:
        return 0.0
    return 1.0 / units


def drawdown_probability(fraction: float, kelly_multiple: float = 1.0) -> float:
    """P(bankroll ever falls to ``fraction`` of *the level it started at*).

    Read that carefully, because it is the most misread number in staking.
    This measures decline from a fixed reference point — the bankroll you have
    today — not decline from a rolling high-water mark.  Those are wildly
    different quantities.  Bet quarter Kelly for long enough and your chance of
    ever being 50% below *today's* bankroll is under 1%, while your chance of
    at some point sitting 50% below a *future peak* approaches certainty,
    simply because the peak keeps climbing and every peak gets retraced.  Use
    this function for "how much of what I have now can I lose?", and a path
    simulation for peak-to-trough questions.

    Continuous-time result: ``P = fraction ^ ((2 - k) / k)`` where k is the
    multiple of full Kelly being staked.  It depends only on k, not on the size
    of the edge, which is why it is the single most useful number for choosing
    a staking plan.  It assumes an unlimited horizon, so it is a mild
    overstatement over any finite run of bets:

    * k = 1.0 (full Kelly)    -> a 50% chance of halving the bankroll at
      some point.  Correct, growth-maximal, and unbearable in practice.
    * k = 0.5 (half Kelly)    -> a 12.5% chance of the same drawdown, for 75%
      of the growth rate.
    * k = 0.25 (quarter Kelly)-> 0.2%, for 44% of the growth.
    * k >= 2.0                -> probability 1.  Every drawdown, however deep,
      happens eventually.  This is the same 2x-Kelly wall that formula 5 shows
      as zero growth.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"Drawdown fraction must be in (0, 1), got {fraction!r}")
    k = float(kelly_multiple)
    if k <= 0.0:
        return 0.0
    if k >= 2.0:
        return 1.0
    return fraction ** ((2.0 - k) / k)


def drawdown_for_probability(probability: float, kelly_multiple: float = 1.0) -> float:
    """Inverse: the drawdown depth you should expect to see with a given odds.

    "What is the worst drawdown I have a 1-in-10 chance of hitting?"
    """
    if not 0.0 < probability < 1.0:
        raise ValueError(f"Probability must be in (0, 1), got {probability!r}")
    k = float(kelly_multiple)
    if k <= 0.0:
        return 1.0
    if k >= 2.0:
        return 0.0
    return probability ** (k / (2.0 - k))


def ruin_table(p: float, b: float = 1.0, unit_counts=(2, 4, 10, 20, 50, 100)) -> list[tuple[float, float]]:
    """(units, ruin probability) pairs — the table that ends the bet-size argument."""
    return [(float(n), risk_of_ruin(p, b, n)) for n in unit_counts]
