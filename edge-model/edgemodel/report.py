"""Human-readable rendering of a slate decision."""

from __future__ import annotations

from .model import BetAnalysis, SlateResult

__all__ = ["format_bet", "format_slate"]


def _pct(x: float, places: int = 2) -> str:
    return f"{x * 100:.{places}f}%"


def format_bet(a: BetAnalysis, bankroll: float | None = None) -> str:
    """The full five-formula readout for one bet."""
    market = _pct(a.market_prob) if a.market_prob is not None else "n/a"
    lines = [
        f"{a.name}   {a.price_american:+.0f} ({a.price_decimal:.3f})   [{a.verdict}]",
        f"  1 expectancy   model {_pct(a.model_prob)} -> market {market} "
        f"-> blended {_pct(a.blended_prob)} vs breakeven {_pct(a.breakeven_prob)}",
        f"                 edge {a.edge:+.3%} per dollar "
        f"({a.edge_points * 100:+.2f} pts), {_pct(a.confidence, 1)} chance it is real",
        f"  2 volatility   sigma {a.sigma:.3f}, signal/noise {a.signal_to_noise:.4f}, "
        f"{a.bets_to_significance:,.0f} bets to prove it",
    ]
    if a.bet:
        lines += [
            f"  3 ruin         {a.units_held:.0f} units held -> "
            f"{_pct(a.ruin_flat, 3)} flat-bet ruin, "
            f"{_pct(a.drawdown_50, 1)} chance of ever halving from here",
            f"  4 kelly        full {_pct(a.kelly_full)} -> sized {_pct(a.stake_fraction)} "
            f"-> stake {a.stake:,.2f}",
            f"  5 growth       {a.growth_rate:+.4%} per bet "
            f"({_pct(a.growth_pct_of_max, 0)} of the peak rate), "
            f"EV {a.expected_profit:+,.2f}",
        ]
    for r in a.reasons:
        lines.append(f"  note           {r}")
    return "\n".join(lines)


def format_slate(result: SlateResult, bankroll: float) -> str:
    """One line per candidate, then the portfolio totals."""
    header = (
        f"{'bet':<26}{'price':>8}{'blend':>8}{'edge':>9}"
        f"{'conf':>7}{'stake':>10}{'units':>7}  verdict"
    )
    lines = [header, "-" * len(header)]
    for a in result.analyses:
        units = f"{a.units_held:.0f}" if a.stake_fraction > 0 else "-"
        lines.append(
            f"{a.name[:25]:<26}{a.price_american:>+8.0f}{_pct(a.blended_prob, 1):>8}"
            f"{a.edge:>+9.2%}{_pct(a.confidence, 0):>7}"
            f"{a.stake:>10,.0f}{units:>7}  {a.verdict}"
        )
    lines += [
        "-" * len(header),
        f"{len(result.bets)} of {len(result.analyses)} bets pass  |  "
        f"staked {result.total_stake:,.0f} of {bankroll:,.0f} "
        f"({result.total_exposure:.2%} exposure)",
        f"expected profit {result.expected_profit:+,.2f}  |  "
        f"portfolio growth {result.portfolio_growth:+.4%} per slate",
    ]
    return "\n".join(lines)
