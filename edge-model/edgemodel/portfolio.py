"""Simultaneous Kelly: sizing a slate, not a coin.

The parable flips one coin at a time and settles it before the next.  A real
betting card does not work that way.  Eight Sunday games kick off at once, and
"10% of bankroll" on each is 80% of bankroll at risk against the possibility
that all eight lose together.  Sequential Kelly applied bet-by-bet to a
simultaneous slate is one of the most common ways a positive-edge bettor ends
up over-staked without ever knowingly breaking a rule.

The fix is to maximise the same log-growth objective over the *joint* outcome
distribution:

    max over f of  sum over scenarios  P(w) * ln(1 + sum_i f_i * x_i(w))

which automatically shrinks each stake as the slate grows, and shrinks it
harder when the legs are correlated.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import product
from math import log, sqrt

from .kelly import kelly_fraction
from .odds import net_odds
from .volatility import norm_ppf

__all__ = [
    "Leg",
    "PortfolioResult",
    "simultaneous_kelly",
    "independent_scenarios",
    "correlated_scenarios",
    "optimize_growth",
]

_MAX_EXACT_LEGS = 14  # 2^14 = 16384 scenarios: still fast in pure Python.


@dataclass
class Leg:
    """One bet in a simultaneous slate."""

    name: str
    p: float
    decimal_odds: float
    group: str | None = None  # Legs sharing a group are correlated.

    @property
    def b(self) -> float:
        return net_odds(self.decimal_odds)


@dataclass
class PortfolioResult:
    fractions: list[float]
    growth_rate: float
    total_exposure: float
    scenarios_used: int
    standalone_fractions: list[float] = field(default_factory=list)

    @property
    def shrinkage(self) -> list[float]:
        """Ratio of the portfolio stake to the naive standalone Kelly stake."""
        out = []
        for f, s in zip(self.fractions, self.standalone_fractions):
            out.append(f / s if s > 0 else 0.0)
        return out


def independent_scenarios(legs: list[Leg]) -> tuple[list[float], list[list[float]]]:
    """Enumerate every joint win/lose combination for independent legs."""
    n = len(legs)
    if n > _MAX_EXACT_LEGS:
        raise ValueError(
            f"{n} legs is {2 ** n} scenarios; use correlated_scenarios() for "
            "Monte-Carlo sampling instead"
        )
    probs: list[float] = []
    payoffs: list[list[float]] = []
    for combo in product((True, False), repeat=n):
        pr = 1.0
        row = []
        for leg, won in zip(legs, combo):
            pr *= leg.p if won else (1.0 - leg.p)
            row.append(leg.b if won else -1.0)
        if pr > 0.0:
            probs.append(pr)
            payoffs.append(row)
    return probs, payoffs


def _cholesky(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = matrix[i][i] - s
                # Nudge a non-PSD correlation matrix back onto the cone.
                L[i][j] = sqrt(max(val, 1e-12))
            else:
                L[i][j] = (matrix[i][j] - s) / L[j][j]
    return L


def correlated_scenarios(
    legs: list[Leg],
    correlation: float | list[list[float]] = 0.0,
    draws: int = 20000,
    seed: int | None = 7,
) -> tuple[list[float], list[list[float]]]:
    """Sample joint outcomes under a Gaussian copula.

    Correlation is what same-game legs actually have: if the favourite covers
    the spread, the game very likely went over the total too.  Pass a single
    number to correlate every pair equally, a full matrix for real structure,
    or rely on ``Leg.group`` by passing a scalar — legs in different groups are
    left independent.
    """
    n = len(legs)
    if isinstance(correlation, (int, float)):
        rho = float(correlation)
        matrix = [
            [
                1.0
                if i == j
                else (rho if _same_group(legs[i], legs[j]) else 0.0)
                for j in range(n)
            ]
            for i in range(n)
        ]
    else:
        matrix = [list(row) for row in correlation]

    L = _cholesky(matrix)
    thresholds = [norm_ppf(min(max(leg.p, 1e-9), 1 - 1e-9)) for leg in legs]
    rng = random.Random(seed)

    # Antithetic pairs: every draw z is used alongside -z, which halves the
    # sampling error in the joint probabilities for free. That matters here
    # because the optimiser will happily exploit sampling noise in a scenario
    # set, and exploiting noise means over-staking.
    counts: dict[tuple[bool, ...], int] = {}
    total = 0
    for _ in range((draws + 1) // 2):
        z = [rng.gauss(0.0, 1.0) for _ in range(n)]
        for sign in (1.0, -1.0):
            corr = [sign * sum(L[i][k] * z[k] for k in range(i + 1)) for i in range(n)]
            # Win when the latent draw lands in the upper tail of mass p.
            key = tuple(corr[i] > -thresholds[i] for i in range(n))
            counts[key] = counts.get(key, 0) + 1
            total += 1

    probs = []
    payoffs = []
    for key, count in counts.items():
        probs.append(count / total)
        payoffs.append([leg.b if won else -1.0 for leg, won in zip(legs, key)])
    return probs, payoffs


def _same_group(a: Leg, b: Leg) -> bool:
    return a.group is not None and a.group == b.group


def _is_effectively_independent(legs: list[Leg], correlation) -> bool:
    """True when the correlation structure carries no off-diagonal weight."""
    n = len(legs)
    if isinstance(correlation, (int, float)):
        if correlation == 0.0:
            return True
        return not any(
            _same_group(legs[i], legs[j])
            for i in range(n)
            for j in range(i + 1, n)
        )
    return all(
        abs(correlation[i][j]) < 1e-12
        for i in range(n)
        for j in range(n)
        if i != j
    )


def _project(f: list[float], caps: list[float], total_cap: float) -> list[float]:
    """Euclidean projection onto {0 <= f_i <= cap_i, sum f_i <= total_cap}."""
    clipped = [min(max(x, 0.0), c) for x, c in zip(f, caps)]
    if sum(clipped) <= total_cap:
        return clipped

    def shifted_sum(theta: float) -> float:
        return sum(min(max(x - theta, 0.0), c) for x, c in zip(f, caps))

    lo, hi = 0.0, max(f) if f else 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if shifted_sum(mid) > total_cap:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            break
    theta = 0.5 * (lo + hi)
    return [min(max(x - theta, 0.0), c) for x, c in zip(f, caps)]


def optimize_growth(
    probs: list[float],
    payoffs: list[list[float]],
    caps: list[float] | None = None,
    total_cap: float = 0.5,
    iterations: int = 600,
) -> tuple[list[float], float]:
    """Maximise expected log growth over a scenario set (concave, so global)."""
    n = len(payoffs[0]) if payoffs else 0
    if n == 0:
        return [], 0.0
    caps = caps or [1.0] * n
    total_cap = min(total_cap, 0.95)  # Keep wealth strictly positive.

    f = [0.0] * n

    def objective(vec: list[float]) -> float:
        total = 0.0
        for pr, row in zip(probs, payoffs):
            w = 1.0 + sum(v * x for v, x in zip(vec, row))
            if w <= 1e-12:
                return float("-inf")
            total += pr * log(w)
        return total

    def gradient(vec: list[float]) -> list[float]:
        grad = [0.0] * n
        for pr, row in zip(probs, payoffs):
            w = 1.0 + sum(v * x for v, x in zip(vec, row))
            if w <= 1e-12:
                continue
            scale = pr / w
            for i, x in enumerate(row):
                grad[i] += scale * x
        return grad

    current = objective(f)
    step = 0.5
    for _ in range(iterations):
        grad = gradient(f)
        improved = False
        for _ in range(60):
            candidate = _project([fi + step * gi for fi, gi in zip(f, grad)], caps, total_cap)
            value = objective(candidate)
            if value > current + 1e-15:
                delta = max(abs(c - fi) for c, fi in zip(candidate, f))
                f, current = candidate, value
                improved = True
                if delta < 1e-12:
                    return f, current
                step *= 1.3
                break
            step *= 0.5
            if step < 1e-15:
                break
        if not improved:
            break
    return f, current


def simultaneous_kelly(
    legs: list[Leg],
    correlation: float | list[list[float]] = 0.0,
    kelly_multiple: float = 1.0,
    max_per_bet: float | None = None,
    total_cap: float = 0.5,
    draws: int = 40000,
    seed: int | None = 7,
) -> PortfolioResult:
    """Growth-optimal stakes for a slate that settles at the same time.

    ``kelly_multiple`` scales the whole solved vector, exactly as fractional
    Kelly scales a single bet.  Applying it after the optimisation rather than
    before keeps the relative sizing across the slate growth-optimal.
    """
    if not legs:
        return PortfolioResult([], 0.0, 0.0, 0)

    # Route on the effective correlation, not on whether groups were labelled:
    # legs tagged with a group but given rho = 0 are still independent, and
    # exact enumeration beats sampling whenever it is available.
    if _is_effectively_independent(legs, correlation) and len(legs) <= _MAX_EXACT_LEGS:
        probs, payoffs = independent_scenarios(legs)
    else:
        probs, payoffs = correlated_scenarios(legs, correlation, draws=draws, seed=seed)

    caps = [max_per_bet if max_per_bet is not None else 1.0] * len(legs)
    fractions, growth = optimize_growth(probs, payoffs, caps, total_cap)

    scaled = [f * kelly_multiple for f in fractions]
    standalone = [kelly_fraction(leg.p, leg.b) * kelly_multiple for leg in legs]
    return PortfolioResult(
        fractions=scaled,
        growth_rate=growth,
        total_exposure=sum(scaled),
        scenarios_used=len(probs),
        standalone_fractions=standalone,
    )
