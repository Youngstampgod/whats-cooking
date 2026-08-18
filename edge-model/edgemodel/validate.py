"""Proving the edge exists — before the variance gets a vote.

Formula 2 says the 55% coin needs ~400 flips before its edge clears two
standard errors.  A 2% sports edge at -110 needs closer to nine *thousand*
bets.  Almost nobody places nine thousand bets before deciding their model
works, which means almost every conclusion drawn from a betting record is
noise wearing a result's clothing.

There is one way out, and it is the reason closing line value is the metric
professionals actually track.  CLV measures the same edge against a far
quieter yardstick, so it converges in tens or hundreds of bets instead of
thousands.  If you beat the closing line consistently, you have an edge even
while your P&L is red; if you do not, you do not have one even while your P&L
is green.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Sequence

from .devig import devig
from .odds import decimal_to_prob, parse_odds
from .volatility import norm_cdf

__all__ = [
    "brier_score",
    "log_loss",
    "calibration_table",
    "clv_percent",
    "clv_probability",
    "ClvSummary",
    "summarize_clv",
    "bets_needed",
    "significance",
    "validation_comparison",
]


def brier_score(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """Mean squared error of probability forecasts. Lower is better.

    Always score against a baseline: the de-vigged closing line's Brier score
    on the same games.  Beating 0.25 (a coin) is meaningless; beating the
    market's own number is the whole game.
    """
    _check_pairs(probs, outcomes)
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def log_loss(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """Log score. Punishes confident errors far harder than Brier does."""
    _check_pairs(probs, outcomes)
    total = 0.0
    for p, o in zip(probs, outcomes):
        p = min(max(p, 1e-15), 1.0 - 1e-15)
        total += -(o * log(p) + (1 - o) * log(1.0 - p))
    return total / len(probs)


def calibration_table(
    probs: Sequence[float], outcomes: Sequence[int], buckets: int = 10
) -> list[dict]:
    """Predicted vs realised frequency, bucketed.

    A model that says 60% and wins 60% is calibrated.  A model that says 60%
    and wins 52% does not have a small edge, it has a negative one — and the
    only thing separating those two diagnoses is this table.
    """
    _check_pairs(probs, outcomes)
    rows: list[dict] = []
    for i in range(buckets):
        lo, hi = i / buckets, (i + 1) / buckets
        idx = [
            j for j, p in enumerate(probs)
            if (lo <= p < hi) or (i == buckets - 1 and p == 1.0)
        ]
        if not idx:
            continue
        pred = sum(probs[j] for j in idx) / len(idx)
        actual = sum(outcomes[j] for j in idx) / len(idx)
        rows.append({
            "bucket": f"{lo:.0%}-{hi:.0%}",
            "n": len(idx),
            "predicted": pred,
            "actual": actual,
            "error": pred - actual,
        })
    return rows


def clv_percent(bet_price, closing_price) -> float:
    """How much better your price was than the close, in percent.

    Beating the close by 2% at the same stake is worth roughly 2% of turnover
    in expectation, which is a bigger number than most bettors' entire edge.
    """
    bet = parse_odds(bet_price)
    close = parse_odds(closing_price)
    return bet / close - 1.0


def clv_probability(
    bet_price, closing_market: Sequence[float], method: str = "shin"
) -> float:
    """CLV in probability points against the de-vigged closing market.

    ``closing_market`` must contain every outcome at the close, with your side
    first.  This is the cleanest estimate of your true edge on the bet: the
    closing line is the sharpest probability anyone produced for that game, so
    the gap between it and the price you got is what you actually captured.
    """
    bet = parse_odds(bet_price)
    closes = [parse_odds(x) for x in closing_market]
    p_close = devig(closes, method=method)[0]
    return p_close - decimal_to_prob(bet)


@dataclass
class ClvSummary:
    n: int
    mean_clv: float
    sd_clv: float
    t_stat: float
    p_value: float
    beat_rate: float

    @property
    def verdict(self) -> str:
        if self.n < 30:
            return "too few bets to say anything"
        if self.t_stat > 2.0:
            return "edge confirmed against the closing line"
        if self.t_stat < -2.0:
            return "negative CLV: the market moves against you, this is not an edge"
        return "inconclusive: keep logging"


def summarize_clv(clv_values: Sequence[float]) -> ClvSummary:
    """Aggregate a CLV log into a verdict."""
    n = len(clv_values)
    if n == 0:
        return ClvSummary(0, 0.0, 0.0, 0.0, 1.0, 0.0)
    mean = sum(clv_values) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in clv_values) / (n - 1)
    else:
        var = 0.0
    sd = sqrt(var)
    t = (mean / (sd / sqrt(n))) if sd > 0 else 0.0
    return ClvSummary(
        n=n,
        mean_clv=mean,
        sd_clv=sd,
        t_stat=t,
        p_value=2.0 * (1.0 - norm_cdf(abs(t))),
        beat_rate=sum(1 for x in clv_values if x > 0) / n,
    )


def bets_needed(mean: float, sd: float, z: float = 2.0) -> float:
    """Sample size for a mean to clear z standard errors of zero.

    The formula behind every "how long is the long run?" argument:
    ``n = (z * sd / mean)^2``.
    """
    if mean <= 0.0 or sd <= 0.0:
        return float("inf")
    return (z * sd / mean) ** 2


def significance(observed_mean: float, sd: float, n: int) -> dict:
    """t-statistic and p-value for a realised betting record."""
    if n <= 0 or sd <= 0:
        return {"t_stat": 0.0, "p_value": 1.0, "standard_error": 0.0}
    se = sd / sqrt(n)
    t = observed_mean / se
    return {
        "t_stat": t,
        "p_value": 2.0 * (1.0 - norm_cdf(abs(t))),
        "standard_error": se,
    }


def validation_comparison(
    edge: float,
    sigma_pnl: float,
    mean_clv: float,
    sd_clv: float,
    z: float = 2.0,
) -> dict:
    """Bets needed to prove the edge via P&L versus via CLV.

    Typically a 50-to-100x difference.  That ratio is the entire argument for
    tracking CLV, and it is also why a bettor who only watches the bankroll
    cannot tell a real edge from a hot streak until long after it matters.
    """
    n_pnl = bets_needed(edge, sigma_pnl, z)
    n_clv = bets_needed(mean_clv, sd_clv, z)
    return {
        "bets_via_pnl": n_pnl,
        "bets_via_clv": n_clv,
        "speedup": (n_pnl / n_clv) if n_clv not in (0.0, float("inf")) else float("inf"),
    }


def _check_pairs(probs: Sequence[float], outcomes: Sequence[int]) -> None:
    if len(probs) != len(outcomes):
        raise ValueError(
            f"Got {len(probs)} probabilities and {len(outcomes)} outcomes"
        )
    if not probs:
        raise ValueError("Need at least one forecast")
    if any(o not in (0, 1, True, False) for o in outcomes):
        raise ValueError("Outcomes must be 0/1")
