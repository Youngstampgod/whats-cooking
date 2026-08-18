"""Command line interface: ``python -m edgemodel <command>``."""

from __future__ import annotations

import argparse
import sys
from math import copysign, sqrt

from .devig import DEVIG_METHODS, devig, no_vig_prices
from .expectancy import breakeven_win_rate, expectancy
from .growth import (
    ensemble_vs_time,
    growth_approx,
    log_growth_rate,
    zero_growth_fraction,
)
from .kelly import growth_fraction_of_max, kelly_fraction
from .model import Bet, EdgeModel, ModelConfig
from .odds import decimal_to_american, hold, net_odds, overround, parse_odds
from .report import format_bet, format_slate
from .ruin import (
    drawdown_probability,
    max_bet_fraction_for_ruin,
    risk_of_ruin,
    units_for_ruin_target,
)
from .simulate import compare_stakes
from .volatility import (
    bet_std,
    bets_to_significance,
    prob_behind_after,
    signal_to_noise,
)


def _bar(value: float, lo: float, hi: float, width: int = 34) -> str:
    """A terminal bar for the growth curve, on a signed square-root scale.

    Linear scaling makes the interesting part invisible: the peak growth rate
    is +0.5% while over-betting reaches -8.9%, so the entire profitable region
    collapses into one character. The square-root scale keeps the sign and the
    ordering exactly right while giving the small positive values room to show.
    Zero is marked with a pipe.
    """
    def t(x: float) -> float:
        return copysign(sqrt(abs(x)), x)

    lo_t, hi_t = t(lo), t(hi)
    span = hi_t - lo_t
    if span <= 0:
        return ""

    def cell(x: float) -> int:
        return max(0, min(width, int(round((t(x) - lo_t) / span * width))))

    zero, point = cell(0.0), cell(value)
    cells = [" "] * (width + 1)
    a, b = sorted((zero, point))
    for i in range(a, b + 1):
        cells[i] = "#"
    cells[zero] = "|"
    return "".join(cells)


def cmd_coin(args) -> int:
    """Walk the five formulas on the weighted coin, end to end."""
    p, b = args.p, args.b
    q = 1.0 - p
    f_star = kelly_fraction(p, b)
    print(f"A coin that lands heads {p:.0%} of the time, paying {b:g} to 1.\n")

    print("1  EXPECTANCY - is there an edge at all?")
    print(f"     E = ({p:.2f} x {b:g}) - ({q:.2f} x 1) = {expectancy(p, b):+.4f} per dollar")
    print(f"     Breakeven win rate at this price is {breakeven_win_rate(1 + b):.2%}.")
    print("     Verdict: allowed to play. That is all this number says.\n")

    print("2  VOLATILITY - how loud is the noise around the edge?")
    print(f"     sigma = {bet_std(p, b):.4f} per dollar, edge = {expectancy(p, b):.4f}")
    print(f"     signal / noise = {signal_to_noise(p, b):.4f}")
    print(f"     The edge does not clear two standard errors until "
          f"{bets_to_significance(p, b):,.0f} bets.")
    for n in (10, 50, 100, 500, 1000):
        print(f"       after {n:>5,} bets, P(still down money) = {prob_behind_after(n, p, b):.1%}")
    print()

    print("3  RISK OF RUIN - can the noise kill you before the edge pays?")
    print(f"     q/p = {q / p:.4f}")
    for units in (2, 4, 10, 20, 50):
        print(f"       bankroll = {units:>3} units ({1 / units:>5.1%} per bet): "
              f"ruin = {risk_of_ruin(p, b, units):7.2%}")
    print(f"     To hold ruin under 1% you need "
          f"{units_for_ruin_target(p, b, 0.01):.1f} units "
          f"({max_bet_fraction_for_ruin(p, b, 0.01):.2%} per bet).\n")

    print("4  KELLY - exactly how much should you bet?")
    print(f"     f* = ({b:g} x {p:.2f} - {q:.2f}) / {b:g} = {f_star:.4f} "
          f"-> stake {f_star:.1%} of bankroll")
    print("     and of the NEW bankroll after every settled bet, not the original.\n")

    print("5  GEOMETRIC GROWTH - why the average is a lie")
    curve = [(f, log_growth_rate(f, p, b)) for f in
             [i / 100 for i in range(1, 61)]]
    lo = min(g for _, g in curve)
    hi = max(g for _, g in curve)
    print(f"     g(f) = p*ln(1+bf) + q*ln(1-f)         peak at f = {f_star:.2f}")
    for f in (0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50):
        g = log_growth_rate(f, p, b)
        mark = "  <- Kelly" if abs(f - f_star) < 1e-9 else ""
        print(f"       f={f:.2f}  g={g:+.4%}  {_bar(g, lo, hi)}{mark}")
    print(f"     Growth returns to zero at f = {zero_growth_fraction(p, b):.4f} "
          f"(~2x Kelly). Past that you compound backwards.")
    print(f"     Approximation g ~= mu - sigma^2/2 at Kelly: "
          f"{growth_approx(f_star, p, b):+.4%} vs exact "
          f"{log_growth_rate(f_star, p, b):+.4%}\n")

    print("   The two averages after 1,000 bets, same coin, same edge:")
    for f in (f_star, 2 * f_star, 5 * f_star):
        r = ensemble_vs_time(f, p, b, 1000)
        print(f"     f={f:.2f}: ensemble mean {r['ensemble_mean']:>14,.0f}   "
              f"median path {r['median_path']:>14,.2f}")
    print("   The gambler optimises the first column. He does not live in it.")
    return 0


def cmd_devig(args) -> int:
    prices = [parse_odds(x) for x in args.prices]
    print(f"market: {', '.join(f'{decimal_to_american(d):+.0f}' for d in prices)}")
    print(f"overround {overround(prices):.2%}   hold {hold(prices):.2%}\n")
    print(f"{'method':<16}{'fair probabilities':<34}fair line")
    for method in DEVIG_METHODS:
        probs = devig(prices, method)
        fair = no_vig_prices(prices, method)
        print(f"{method:<16}"
              f"{'  '.join(f'{p:.4f}' for p in probs):<34}"
              f"{'  '.join(f'{decimal_to_american(d):+.0f}' for d in fair)}")
    spread = max(devig(prices, m)[0] for m in DEVIG_METHODS) - \
        min(devig(prices, m)[0] for m in DEVIG_METHODS)
    print(f"\nmethods disagree by {spread * 100:.2f} probability points on the first "
          f"outcome.\nIf your claimed edge is smaller than that, you have found a "
          f"de-vig artefact, not an edge.")
    return 0


def cmd_price(args) -> int:
    cfg = ModelConfig(
        bankroll=args.bankroll,
        model_weight=args.model_weight,
        kelly_multiple=args.kelly,
        prob_sd=args.sd,
        max_bet_fraction=args.max_bet,
    )
    model = EdgeModel(cfg)
    bet = Bet(
        name=args.name,
        model_prob=args.p,
        price=args.price,
        reference_prices=args.market,
    )
    print(format_bet(model.price_bet(bet)))
    return 0


def cmd_curve(args) -> int:
    p, b = args.p, args.b
    curve = [(f, log_growth_rate(f, p, b)) for f in
             [i / 200 for i in range(1, args.max_points + 1)]]
    lo, hi = min(g for _, g in curve), max(g for _, g in curve)
    f_star = kelly_fraction(p, b)
    print(f"growth rate by stake size   p={p} b={b}   Kelly={f_star:.4f}   "
          f"zero growth at {zero_growth_fraction(p, b):.4f}")
    for f, g in curve:
        if int(round(f * 200)) % args.every:
            continue
        mark = " <- Kelly" if abs(f - f_star) < 0.0025 else ""
        print(f"  f={f:5.3f}  g={g:+8.4%}  {_bar(g, lo, hi)}{mark}")
    return 0


def cmd_ruin(args) -> int:
    p, b = args.p, args.b
    print(f"risk of ruin, flat betting   p={p} b={b}")
    print(f"{'units':>8}{'per bet':>10}{'ruin':>12}")
    for n in (2, 3, 4, 5, 10, 20, 30, 50, 100):
        print(f"{n:>8}{1 / n:>10.1%}{risk_of_ruin(p, b, n):>12.4%}")
    print()
    print("drawdown risk under proportional (Kelly) staking, from today's bankroll:")
    print(f"{'multiple':>10}{'P(-50%)':>10}{'P(-75%)':>10}{'growth kept':>13}")
    for k in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        print(f"{k:>9.2f}x{drawdown_probability(0.5, k):>10.1%}"
              f"{drawdown_probability(0.25, k):>10.1%}"
              f"{growth_fraction_of_max(k):>13.0%}")
    return 0


def cmd_simulate(args) -> int:
    print(f"p={args.p} b={args.b}, {args.bets:,} bets, {args.paths:,} paths, "
          f"start {args.bankroll:,.0f}")
    print(f"{'stake':<16}{'ruin':>8}{'median end':>14}{'mean end':>16}"
          f"{'5th pct':>12}{'max DD':>9}")
    for r in compare_stakes(args.p, args.b, bankroll=args.bankroll,
                            bets=args.bets, paths=args.paths):
        print(f"{r.label:<16}{r.ruin_rate:>8.2%}{r.median_terminal:>14,.0f}"
              f"{r.mean_terminal:>16,.0f}{r.p05_terminal:>12,.0f}"
              f"{r.mean_max_drawdown:>9.1%}")
    print("\nthe mean column rises with stake size long after the median has "
          "collapsed.\nthat gap is formula 5.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edgemodel",
        description="A sports betting edge model built on five formulas.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("coin", help="walk all five formulas on the weighted coin")
    c.add_argument("--p", type=float, default=0.55)
    c.add_argument("--b", type=float, default=1.0)
    c.set_defaults(func=cmd_coin)

    d = sub.add_parser("devig", help="strip the margin out of a market")
    d.add_argument("prices", nargs="+", help="every outcome, e.g. -110 -110")
    d.set_defaults(func=cmd_devig)

    pr = sub.add_parser("price", help="price and size a single bet")
    pr.add_argument("--name", default="bet")
    pr.add_argument("--p", type=float, required=True, help="your model's probability")
    pr.add_argument("--price", required=True, help="the price you can get")
    pr.add_argument("--market", nargs="*", default=None,
                    help="reference market, every outcome, your side first")
    pr.add_argument("--bankroll", type=float, default=10000.0)
    pr.add_argument("--kelly", type=float, default=0.5)
    pr.add_argument("--model-weight", type=float, default=0.35)
    pr.add_argument("--sd", type=float, default=0.02)
    pr.add_argument("--max-bet", type=float, default=0.02)
    pr.set_defaults(func=cmd_price)

    cv = sub.add_parser("curve", help="growth rate against stake size")
    cv.add_argument("--p", type=float, default=0.55)
    cv.add_argument("--b", type=float, default=1.0)
    cv.add_argument("--max-points", type=int, default=100)
    cv.add_argument("--every", type=int, default=4)
    cv.set_defaults(func=cmd_curve)

    r = sub.add_parser("ruin", help="ruin and drawdown tables")
    r.add_argument("--p", type=float, default=0.55)
    r.add_argument("--b", type=float, default=1.0)
    r.set_defaults(func=cmd_ruin)

    s = sub.add_parser("simulate", help="monte carlo the bankroll paths")
    s.add_argument("--p", type=float, default=0.55)
    s.add_argument("--b", type=float, default=1.0)
    s.add_argument("--bets", type=int, default=1000)
    s.add_argument("--paths", type=int, default=2000)
    s.add_argument("--bankroll", type=float, default=1000.0)
    s.set_defaults(func=cmd_simulate)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        # Bad odds and impossible markets are user errors, not crashes.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
