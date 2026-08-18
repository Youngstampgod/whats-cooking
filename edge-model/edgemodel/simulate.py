"""Monte Carlo: watching the two averages come apart.

Formula 5 claims that the mean ending bankroll and the typical ending bankroll
are different numbers, and that over-betting drives them in opposite
directions.  This module runs the paths so you can see it happen rather than
take it on faith.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import median

from .growth import log_growth_rate
from .kelly import kelly_fraction

__all__ = ["SimResult", "simulate_flat", "simulate_kelly", "compare_stakes"]


@dataclass
class SimResult:
    label: str
    paths: int
    bets: int
    ruin_rate: float
    mean_terminal: float
    median_terminal: float
    p05_terminal: float
    p95_terminal: float
    mean_max_drawdown: float
    worst_drawdown: float
    growth_rate_realised: float

    def row(self) -> str:
        return (
            f"{self.label:<16} ruin {self.ruin_rate:6.2%}  "
            f"median {self.median_terminal:>12,.0f}  mean {self.mean_terminal:>14,.0f}  "
            f"p05 {self.p05_terminal:>10,.0f}  maxDD {self.mean_max_drawdown:6.1%}"
        )


def _summarise(label, terminals, drawdowns, ruined, bets, start) -> SimResult:
    n = len(terminals)
    ordered = sorted(terminals)
    med = median(terminals)
    from math import log

    realised = (log(med / start) / bets) if med > 0 else float("-inf")
    return SimResult(
        label=label,
        paths=n,
        bets=bets,
        ruin_rate=ruined / n,
        mean_terminal=sum(terminals) / n,
        median_terminal=med,
        p05_terminal=ordered[max(0, int(0.05 * n) - 1)],
        p95_terminal=ordered[min(n - 1, int(0.95 * n))],
        mean_max_drawdown=sum(drawdowns) / n,
        worst_drawdown=max(drawdowns),
        growth_rate_realised=realised,
    )


def simulate_flat(
    p: float,
    b: float = 1.0,
    stake: float = 100.0,
    bankroll: float = 1000.0,
    bets: int = 1000,
    paths: int = 5000,
    seed: int | None = 42,
    label: str | None = None,
) -> SimResult:
    """Fixed stake every bet, with a real absorbing barrier at zero."""
    rng = random.Random(seed)
    terminals, drawdowns = [], []
    ruined = 0
    for _ in range(paths):
        w = bankroll
        peak = w
        worst = 0.0
        for _ in range(bets):
            if w < stake:
                ruined += 1
                w = 0.0
                break
            w += stake * b if rng.random() < p else -stake
            peak = max(peak, w)
            worst = max(worst, 1.0 - w / peak)
        terminals.append(w)
        drawdowns.append(worst)
    return _summarise(
        label or f"flat {stake:g}", terminals, drawdowns, ruined, bets, bankroll
    )


def simulate_kelly(
    p: float,
    b: float = 1.0,
    fraction: float | None = None,
    kelly_multiple: float = 1.0,
    bankroll: float = 1000.0,
    bets: int = 1000,
    paths: int = 5000,
    ruin_threshold: float = 0.01,
    seed: int | None = 42,
    label: str | None = None,
) -> SimResult:
    """Proportional staking. 'Ruin' means falling below a fraction of the start,
    since proportional betting never mathematically reaches zero."""
    f = fraction if fraction is not None else kelly_fraction(p, b) * kelly_multiple
    rng = random.Random(seed)
    floor = bankroll * ruin_threshold
    terminals, drawdowns = [], []
    ruined = 0
    for _ in range(paths):
        w = bankroll
        peak = w
        worst = 0.0
        hit = False
        for _ in range(bets):
            w += w * f * b if rng.random() < p else -w * f
            peak = max(peak, w)
            worst = max(worst, 1.0 - w / peak)
            if w <= floor:
                hit = True
        if hit:
            ruined += 1
        terminals.append(w)
        drawdowns.append(worst)
    return _summarise(
        label or f"f={f:.3f}", terminals, drawdowns, ruined, bets, bankroll
    )


def compare_stakes(
    p: float,
    b: float = 1.0,
    multiples=(0.25, 0.5, 1.0, 1.5, 2.0, 3.0),
    bankroll: float = 1000.0,
    bets: int = 1000,
    paths: int = 3000,
    seed: int | None = 42,
) -> list[SimResult]:
    """Run the same edge at several Kelly multiples on identical coin flips."""
    out = []
    for k in multiples:
        f = kelly_fraction(p, b) * k
        out.append(
            simulate_kelly(
                p, b, fraction=f, bankroll=bankroll, bets=bets, paths=paths,
                seed=seed, label=f"{k:g}x kelly",
            )
        )
    return out
