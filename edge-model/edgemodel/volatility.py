"""Formula 2 — Volatility: how loud is the noise around the edge?

    sigma = sqrt( E[X^2] - mu^2 )

On the even-money coin the edge is ten cents and the swing on a single flip is
about a dollar.  The signal-to-noise ratio is 0.10, so the edge is inaudible
for hundreds of bets.  The functions here put a number on "hundreds" — how long
the roar lasts before the whisper wins.
"""

from __future__ import annotations

from math import erf, sqrt

from .expectancy import expectancy

__all__ = [
    "bet_variance",
    "bet_std",
    "signal_to_noise",
    "sharpe_after",
    "bets_to_significance",
    "prob_behind_after",
    "prob_below_after",
    "norm_cdf",
    "norm_ppf",
]


def bet_variance(p: float, b: float = 1.0) -> float:
    """Variance of the per-dollar result of one bet.

    ``X = +b`` with probability p, ``-1`` with probability q.
    """
    q = 1.0 - p
    second_moment = p * b * b + q
    mu = expectancy(p, b)
    return second_moment - mu * mu


def bet_std(p: float, b: float = 1.0) -> float:
    """Standard deviation of one bet's per-dollar result — the roar."""
    return sqrt(bet_variance(p, b))


def signal_to_noise(p: float, b: float = 1.0) -> float:
    """Edge divided by its own standard deviation — the per-bet Sharpe.

    The 55% coin scores 0.10.  That single number explains why streaks feel
    like information and are not.
    """
    sigma = bet_std(p, b)
    if sigma == 0.0:
        return float("inf")
    return expectancy(p, b) / sigma


def sharpe_after(n: int, p: float, b: float = 1.0) -> float:
    """Signal-to-noise after n independent bets: it grows only as sqrt(n)."""
    return signal_to_noise(p, b) * sqrt(max(n, 0))


def bets_to_significance(p: float, b: float = 1.0, z: float = 2.0) -> float:
    """How many bets before the edge is z standard errors clear of zero.

    The answer for the 55% coin is roughly 400 flips at z = 2.  Anyone drawing
    conclusions from fifty bets is reading noise.
    """
    snr = signal_to_noise(p, b)
    if snr <= 0.0:
        return float("inf")
    return (z / snr) ** 2


def prob_behind_after(n: int, p: float, b: float = 1.0) -> float:
    """Probability of being down money after n bets despite a positive edge.

    Uses the normal approximation to the sum of n flat bets.  For the 55% coin
    it is still 16% after a hundred flips — which is what a losing month looks
    like when nothing is wrong.
    """
    return prob_below_after(0.0, n, p, b)


def prob_below_after(threshold_units: float, n: int, p: float, b: float = 1.0) -> float:
    """Probability that cumulative profit after n flat bets is below a level."""
    if n <= 0:
        return 1.0 if threshold_units > 0.0 else 0.0
    mu = expectancy(p, b) * n
    sigma = bet_std(p, b) * sqrt(n)
    if sigma == 0.0:
        return 1.0 if mu < threshold_units else 0.0
    return norm_cdf((threshold_units - mu) / sigma)


def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


# Acklam's rational approximation to the inverse normal CDF; |error| < 1.15e-9.
_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
      1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
      6.680131188771972e01, -1.328068155288572e01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
      -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
      3.754408661907416e00)


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF, pure stdlib."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf needs p in (0, 1), got {p!r}")
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = sqrt(-2.0 * _log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
               ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    if p > p_high:
        q = sqrt(-2.0 * _log(1.0 - p))
        return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
                ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
           (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)


def _log(x: float) -> float:
    from math import log

    return log(x)
