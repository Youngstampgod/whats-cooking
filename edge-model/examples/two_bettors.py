"""Same coin. Same edge. Same direction, every time.

Two people sit down with the identical bet and the identical information, and
we deal them the *identical sequence of flips* -- not two random runs, the same
run, flip for flip. Neither of them is ever wrong about the coin. Neither of
them ever bets tails.

The only line of code that differs between them is the bet size.

    python3 examples/two_bettors.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgemodel import (  # noqa: E402
    drawdown_probability,
    expectancy,
    implied_kelly_multiple,
    kelly_fraction,
    log_growth_rate,
    risk_of_ruin,
)

P, B = 0.55, 1.0
FLIPS = 1000
START = 1_000.0
BUST = 1.0  # Below one dollar you cannot make a meaningful bet again.


def run(fraction: float, flips: list[bool]) -> tuple[list[float], bool]:
    """Walk one bankroll through a fixed sequence of flips."""
    wealth = START
    path = [wealth]
    for heads in flips:
        stake = wealth * fraction
        wealth += stake * B if heads else -stake
        path.append(wealth)
        if wealth < BUST:
            return path, True
    return path, False


def describe(label: str, fraction: float, path: list[float], busted: bool) -> None:
    k = implied_kelly_multiple(fraction, P, B)
    g = log_growth_rate(fraction, P, B)
    peak = max(path)
    worst = max(1.0 - w / max(path[: i + 1]) for i, w in enumerate(path))
    print(f"  {label}")
    print(f"    stakes            {fraction:.0%} of bankroll  ({k:.1f}x Kelly)")
    print(f"    growth rate       {g:+.4%} per flip  "
          f"({'compounding up' if g > 0 else 'compounding down'})")
    print(f"    peak bankroll     {peak:>16,.2f}")
    print(f"    final bankroll    {path[-1]:>16,.2f}"
          f"{'   [BUST]' if busted else ''}")
    print(f"    worst drawdown    {worst:.1%}")
    print()


def main() -> None:
    rng = random.Random(2024)
    flips = [rng.random() < P for _ in range(FLIPS)]
    heads = sum(flips)

    print(__doc__.split("\n")[0])
    print("=" * 66)
    print(f"A coin that lands heads {P:.0%} of the time, paid at even money.")
    print(f"Expectancy: {expectancy(P, B):+.2f} per dollar. A real edge; a casino "
          f"would kill for it.")
    print(f"Kelly says: stake {kelly_fraction(P, B):.0%}.")
    print()
    print(f"One shared sequence of {FLIPS:,} flips: {heads} heads, "
          f"{FLIPS - heads} tails ({heads / FLIPS:.1%}).")
    print("Both bettors see the same flips in the same order and both bet heads "
          "every time.")
    print("=" * 66)
    print()

    for label, fraction in (
        ("THE QUANT   - bets the Kelly fraction", kelly_fraction(P, B)),
        ("THE GAMBLER - bets half his stack, to feel something", 0.50),
    ):
        path, busted = run(fraction, flips)
        describe(label, fraction, path, busted)

    quant, _ = run(kelly_fraction(P, B), flips)
    gambler, _ = run(0.50, flips)

    print("-" * 66)
    print(f"{'after':>8}{'quant':>18}{'gambler':>18}")
    for n in (0, 10, 50, 100, 250, 500, 1000):
        # Index each path on its own length: the gambler's ends when he busts,
        # and freezing the quant's column to match would hide the whole point.
        q = quant[min(n, len(quant) - 1)]
        g_val = gambler[min(n, len(gambler) - 1)]
        g_txt = f"{g_val:,.2f}" if n < len(gambler) else "bust"
        print(f"{n:>8,}{q:>18,.2f}{g_txt:>18}")
    print("-" * 66)
    print()
    print(f"This sequence ran {heads / FLIPS:.1%} heads rather than {P:.0%} -- "
          f"ordinary noise, and")
    print("it cost the quant an order of magnitude against his expected finish. "
          "He still")
    print("compounded. The coin never turned on the gambler either; he was right "
          "about it")
    print("for a thousand flips.")
    print(f"He staked {implied_kelly_multiple(0.50, P, B):.0f}x Kelly, which puts "
          f"his growth rate at {log_growth_rate(0.50, P, B):+.2%} per flip --")
    print("a positive edge and a negative destiny. The distance between the two "
          "columns")
    print("is one number, and it is not the coin.")
    print()
    print("What each of them could have known before the first flip:")
    print(f"  quant   : ruin over 1,000 flips ~ "
          f"{risk_of_ruin(P, B, 1 / kelly_fraction(P, B)):.2%}, "
          f"P(ever -50%) = {drawdown_probability(0.5, 1.0):.0%}")
    print(f"  gambler : growth rate {log_growth_rate(0.50, P, B):+.2%}/flip, "
          f"P(ever -50%) = {drawdown_probability(0.5, 5.0):.0%}")


if __name__ == "__main__":
    main()
