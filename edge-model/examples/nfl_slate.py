"""A Sunday slate through all five formulas.

The coin arrives with 0.55 stamped on it. A football game does not. Everything
that makes sports betting harder than the parable happens between the model's
output and the stake:

  * the quoted price contains the book's margin, so it overstates probability
  * your model's number has error bars, and you bet exactly where it disagrees
    most with the market -- which is exactly where it is most likely wrong
  * eight games kick off at once, so "10% each" is 80% at risk simultaneously
  * legs from the same game move together

    python3 examples/nfl_slate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgemodel import (  # noqa: E402
    Bet,
    EdgeModel,
    ModelConfig,
    bets_needed,
    devig,
    drawdown_probability,
    parse_odds,
    validation_comparison,
)
from edgemodel.report import format_bet, format_slate  # noqa: E402

BANKROLL = 25_000.0

# name, model probability, price available, sharp reference market (your side first)
SLATE = [
    ("Chiefs -3.5",        0.585, -105, [-108, -112], None),
    ("Bills ML",           0.640, +105, [-118, -102], None),
    ("Ravens -7",          0.545, -110, [-110, -110], None),
    ("Lions/Bears over 48", 0.560, -108, [-105, -115], "DET-CHI"),
    ("Lions -2.5",         0.575, -115, [-112, -108], "DET-CHI"),
    ("Jets +6.5",          0.520, -110, [-104, -116], None),
    ("Rams ML",            0.480, +140, [+150, -175], None),
    ("Texans -1",          0.610, -120, [-135, +115], None),
]


def main() -> None:
    config = ModelConfig(
        bankroll=BANKROLL,
        model_weight=0.35,      # you are not sharper than the market; you are a voice in it
        prob_sd=0.025,          # honest error bar on a single game probability
        kelly_multiple=0.5,     # half Kelly
        min_edge=0.01,
        min_confidence=0.55,
        max_bet_fraction=0.02,
        max_total_exposure=0.15,
        devig_method="shin",
    )
    model = EdgeModel(config)
    bets = [
        Bet(name, model_prob=p, price=price, reference_prices=ref, group=group)
        for name, p, price, ref, group in SLATE
    ]

    print(__doc__.split("\n")[0])
    print("=" * 78)
    print(f"bankroll {BANKROLL:,.0f}   half Kelly   model weight "
          f"{config.model_weight:.0%}   cap {config.max_bet_fraction:.0%}/bet, "
          f"{config.max_total_exposure:.0%} total")
    print("=" * 78)
    print()

    print("STEP 1 - what the market actually believes, once the margin is removed")
    print(f"  {'bet':<24}{'your price':>12}{'raw implied':>13}{'sharp fair':>12}"
          f"{'vs fair':>9}")
    for name, p, price, ref, _ in SLATE:
        d = parse_odds(price)
        fair = devig([parse_odds(x) for x in ref], "shin")[0]
        raw = 1.0 / d
        print(f"  {name:<24}{price:>12}{raw:>12.1%}{fair:>12.1%}"
              f"{raw - fair:>9.1%}")
    print()
    print("  'raw implied' is 1/price and is always inflated by the margin you")
    print("  are paying. 'sharp fair' is what the reference market believes once")
    print("  its own margin is stripped. Score your model against the raw column")
    print("  and you will find edges that are pure vig. A negative 'vs fair' is")
    print("  the good case: your book is offering a better price than the sharp")
    print("  consensus, which is line shopping doing more work than any model.")
    print()

    result = model.price_slate(bets, correlation=0.55)

    print("STEP 2 - the slate, priced and sized")
    print(format_slate(result, BANKROLL))
    print()

    print("STEP 3 - one rejection and one bet, in full")
    rejected = next((a for a in result.analyses if not a.bet), None)
    accepted = next((a for a in result.analyses if a.bet), None)
    for a in (rejected, accepted):
        if a is not None:
            print(format_bet(a))
            print()

    print("STEP 4 - what the sizing bought you")
    if result.bets:
        worst = 1.0 - result.total_exposure
        print(f"  total at risk        {result.total_stake:,.0f} "
              f"({result.total_exposure:.2%} of bankroll)")
        print(f"  if every bet loses   bankroll = {BANKROLL * worst:,.0f} "
              f"({worst - 1:.1%})")
        print(f"  expected profit      {result.expected_profit:+,.2f} per slate")
        print(f"  P(ever -50% from here at half Kelly) = "
              f"{drawdown_probability(0.5, 0.5):.1%}")
        print(f"  at full Kelly that would be           "
              f"{drawdown_probability(0.5, 1.0):.1%}")
    else:
        print("  Nothing cleared the bar. That is a normal Sunday, and passing")
        print("  costs you nothing -- formula 1 is an entrance ticket, not an")
        print("  obligation to walk through the door.")
    print()

    print("STEP 5 - how long before you know any of this was real")
    edge_per_bet = (
        result.expected_profit / result.total_stake if result.total_stake else 0.02
    )
    comp = validation_comparison(
        edge=max(edge_per_bet, 0.005), sigma_pnl=0.95,
        mean_clv=0.01, sd_clv=0.025,
    )
    per_week = max(len(result.bets), 1)
    print(f"  edge per dollar staked   {edge_per_bet:+.2%}  (this slate, after shrinkage)")
    print(f"  bets to prove it by P&L  {comp['bets_via_pnl']:>10,.0f}"
          f"   ~{comp['bets_via_pnl'] / per_week / 52:>5.1f} seasons")
    print(f"  bets to prove it by CLV  {comp['bets_via_clv']:>10,.0f}"
          f"   ~{comp['bets_via_clv'] / per_week / 52:>5.1f} seasons"
          f"   ({comp['speedup']:,.0f}x faster)")
    print()
    modest = validation_comparison(edge=0.02, sigma_pnl=0.95,
                                   mean_clv=0.01, sd_clv=0.025)
    print(f"  That first number flatters itself, because this slate's edge is")
    print(f"  carried by one 15% outlier. At a realistic 2% edge the P&L answer")
    print(f"  needs {modest['bets_via_pnl']:,.0f} bets "
          f"(~{modest['bets_via_pnl'] / per_week / 52:.0f} seasons) while CLV still "
          f"needs {modest['bets_via_clv']:,.0f}.")
    print("  This is why the closing line is the scoreboard and the bankroll is")
    print("  only the consequence.")


if __name__ == "__main__":
    main()
