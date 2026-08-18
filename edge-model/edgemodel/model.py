"""The edge model: the five formulas wired into one decision.

Read the pipeline as the five formulas in order.

1. Is there an edge?      de-vig the market, blend it with the model, compare
                          the result to the price you can actually get.
2. How loud is the noise?  attach a standard deviation and a signal-to-noise
                          ratio to the bet, and a sample size to the claim.
3. Can it kill me first?   convert the proposed stake into bankroll units and
                          read the ruin and drawdown numbers off it.
4. How much do I bet?      Kelly on a conservative quantile of the probability,
                          scaled by a fractional multiple, then capped.
5. What do I actually keep? the geometric growth this bet contributes, which is
                          the only number that compounds.

The one thing the parable does not have to worry about, and the thing that
decides whether a real sports model makes money, is that ``p`` is *estimated*.
The coin came stamped 0.55.  Your model produces a number with error bars, and
you place bets precisely where that number disagrees most with the market —
which is precisely where it is most likely to be wrong.  That selection effect
is the winner's curse, and shrinking toward the market price is the cheapest
defence against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log

from .devig import devig
from .expectancy import expectancy
from .growth import log_growth_rate
from .kelly import kelly_fraction, kelly_quantile, probability_edge_is_real
from .odds import decimal_to_american, decimal_to_prob, net_odds, parse_odds
from .portfolio import Leg, simultaneous_kelly
from .ruin import drawdown_probability, risk_of_ruin
from .volatility import bet_std, bets_to_significance, norm_ppf, signal_to_noise

__all__ = ["Bet", "ModelConfig", "BetAnalysis", "SlateResult", "EdgeModel", "logit", "sigmoid"]


def logit(p: float) -> float:
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    e = exp(x)
    return e / (1.0 + e)


@dataclass
class Bet:
    """A candidate bet.

    ``price`` is what you can actually get filled at — not the consensus.
    ``reference_prices`` is the full market (every outcome) from the sharpest
    book you can see; it is what gets de-vigged into a market probability.
    Leaving it out means the model has no consensus to shrink toward, which is
    allowed and clearly flagged, but it is the weakest way to run this.
    """

    name: str
    model_prob: float
    price: float
    reference_prices: list[float] | None = None
    group: str | None = None
    prob_sd: float | None = None

    def __post_init__(self) -> None:
        self.price = parse_odds(self.price)
        if self.reference_prices is not None:
            self.reference_prices = [parse_odds(x) for x in self.reference_prices]
        if not 0.0 < self.model_prob < 1.0:
            raise ValueError(
                f"{self.name}: model_prob must be in (0, 1), got {self.model_prob!r}"
            )


@dataclass
class ModelConfig:
    """Every knob that separates the quant from the gambler, in one place."""

    bankroll: float = 10_000.0
    #: Weight on your own model when blending with the de-vigged market.
    #: 1.0 means you believe you are sharper than the market on every bet;
    #: for most models the honest number is well under 0.5.
    model_weight: float = 0.35
    #: Standard error of a single probability estimate, in probability points.
    prob_sd: float = 0.02
    #: Size off this quantile of the probability estimate rather than the mean.
    sizing_quantile: float = 0.25
    #: Multiple of full Kelly. Half or less is the professional default.
    kelly_multiple: float = 0.5
    #: Reject bets whose expectancy per dollar is below this.
    min_edge: float = 0.01
    #: Reject bets less likely than this to have a genuinely positive edge.
    min_confidence: float = 0.55
    #: Hard cap on a single stake, as a fraction of bankroll.
    max_bet_fraction: float = 0.02
    #: Hard cap on total simultaneous exposure, as a fraction of bankroll.
    max_total_exposure: float = 0.15
    #: Acceptable lifetime probability of ruin, used to sanity-check sizing.
    ruin_target: float = 0.01
    devig_method: str = "shin"
    round_to: float | None = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.model_weight <= 1.0:
            raise ValueError(f"model_weight must be in [0, 1], got {self.model_weight}")
        if self.kelly_multiple < 0.0:
            raise ValueError("kelly_multiple must be >= 0")
        if self.kelly_multiple > 2.0:
            raise ValueError(
                "kelly_multiple above 2.0 has non-positive growth by construction "
                "(formula 5); refusing to size it"
            )


@dataclass
class BetAnalysis:
    """Everything the five formulas have to say about one bet."""

    name: str
    price_decimal: float
    price_american: float
    model_prob: float
    market_prob: float | None
    blended_prob: float
    breakeven_prob: float
    # 1 - expectancy
    edge: float
    edge_points: float
    # 2 - volatility
    sigma: float
    signal_to_noise: float
    bets_to_significance: float
    # 3 - ruin
    units_held: float
    ruin_flat: float
    drawdown_50: float
    # 4 - kelly
    kelly_full: float
    kelly_sized: float
    stake_fraction: float
    stake: float
    confidence: float
    # 5 - growth
    growth_rate: float
    growth_pct_of_max: float
    verdict: str
    reasons: list[str] = field(default_factory=list)

    @property
    def bet(self) -> bool:
        return self.verdict == "BET"

    @property
    def expected_profit(self) -> float:
        return self.stake * self.edge


@dataclass
class SlateResult:
    analyses: list[BetAnalysis]
    total_stake: float
    total_exposure: float
    portfolio_growth: float
    expected_profit: float

    @property
    def bets(self) -> list[BetAnalysis]:
        return [a for a in self.analyses if a.bet]


class EdgeModel:
    """Turns a candidate bet into a stake, or into a documented refusal."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()

    # -- formula 1 inputs -------------------------------------------------
    def market_probability(self, bet: Bet) -> float | None:
        """De-vig the reference market into a fair probability for this side."""
        if not bet.reference_prices:
            return None
        probs = devig(bet.reference_prices, method=self.config.devig_method)
        return probs[0]

    def blend(self, model_prob: float, market_prob: float | None) -> float:
        """Shrink the model toward the market in log-odds space.

        Log-odds rather than raw probability because it behaves correctly in
        the tails: blending 2% with 4% linearly gives 3%, which is a wildly
        different price, while the log-odds blend respects the geometry the
        odds are actually quoted on.
        """
        if market_prob is None:
            return model_prob
        w = self.config.model_weight
        return sigmoid(w * logit(model_prob) + (1.0 - w) * logit(market_prob))

    # -- the pipeline -----------------------------------------------------
    def price_bet(self, bet: Bet, bankroll: float | None = None) -> BetAnalysis:
        cfg = self.config
        bankroll = cfg.bankroll if bankroll is None else bankroll
        b = net_odds(bet.price)

        market_prob = self.market_probability(bet)
        p = self.blend(bet.model_prob, market_prob)
        breakeven = decimal_to_prob(bet.price)
        sd = bet.prob_sd if bet.prob_sd is not None else cfg.prob_sd

        # 1 - expectancy
        edge = expectancy(p, b)

        # 2 - volatility
        sigma = bet_std(p, b)
        snr = signal_to_noise(p, b)
        n_sig = bets_to_significance(p, b, z=2.0)

        # 4 - kelly (computed before 3, because ruin is a function of the stake)
        kelly_full = kelly_fraction(p, b)
        confidence = probability_edge_is_real(p, sd, bet.price)
        sized = kelly_quantile(p, sd, bet.price, cfg.sizing_quantile)
        sized *= cfg.kelly_multiple
        stake_fraction = min(sized, cfg.max_bet_fraction)

        reasons: list[str] = []
        verdict = "BET"
        if market_prob is None:
            reasons.append("no reference market: nothing to shrink toward")
        if edge <= 0.0:
            verdict, _ = "PASS", reasons.append(
                f"no edge after de-vig and shrinkage ({edge:+.3%} per dollar)"
            )
        elif edge < cfg.min_edge:
            verdict, _ = "PASS", reasons.append(
                f"edge {edge:+.3%} below {cfg.min_edge:.2%} threshold"
            )
        elif confidence < cfg.min_confidence:
            verdict, _ = "PASS", reasons.append(
                f"only {confidence:.1%} chance the edge is real "
                f"(needs {cfg.min_confidence:.0%}); estimate sd {sd:.3f}"
            )
        elif stake_fraction <= 0.0:
            p_low = p + norm_ppf(cfg.sizing_quantile) * sd
            verdict, _ = "PASS", reasons.append(
                f"sized to zero: the {cfg.sizing_quantile:.0%} quantile of your "
                f"estimate ({p_low:.2%}) sits below the breakeven price "
                f"({breakeven:.2%}), so the edge does not survive its own error bar"
            )

        if verdict != "BET":
            stake_fraction = 0.0
        elif sized > cfg.max_bet_fraction:
            reasons.append(
                f"stake capped at {cfg.max_bet_fraction:.2%} of bankroll "
                f"(Kelly wanted {sized:.2%})"
            )

        stake = bankroll * stake_fraction
        if cfg.round_to and stake > 0:
            stake = round(stake / cfg.round_to) * cfg.round_to
            stake_fraction = stake / bankroll if bankroll else 0.0

        # 3 - ruin, measured on the stake we actually landed on
        units = (1.0 / stake_fraction) if stake_fraction > 0 else float("inf")
        ruin_flat = risk_of_ruin(p, b, units) if edge > 0 else 1.0
        k_multiple = (stake_fraction / kelly_full) if kelly_full > 0 else 0.0
        dd50 = drawdown_probability(0.5, k_multiple) if k_multiple > 0 else 0.0

        # 5 - growth
        g = log_growth_rate(stake_fraction, p, b) if stake_fraction > 0 else 0.0
        g_max = log_growth_rate(kelly_full, p, b) if kelly_full > 0 else 0.0
        pct_max = (g / g_max) if g_max > 0 else 0.0

        return BetAnalysis(
            name=bet.name,
            price_decimal=bet.price,
            price_american=decimal_to_american(bet.price),
            model_prob=bet.model_prob,
            market_prob=market_prob,
            blended_prob=p,
            breakeven_prob=breakeven,
            edge=edge,
            edge_points=p - breakeven,
            sigma=sigma,
            signal_to_noise=snr,
            bets_to_significance=n_sig,
            units_held=units,
            ruin_flat=ruin_flat,
            drawdown_50=dd50,
            kelly_full=kelly_full,
            kelly_sized=sized,
            stake_fraction=stake_fraction,
            stake=stake,
            confidence=confidence,
            growth_rate=g,
            growth_pct_of_max=pct_max,
            verdict=verdict,
            reasons=reasons,
        )

    def price_slate(
        self,
        bets: list[Bet],
        bankroll: float | None = None,
        correlation: float = 0.0,
    ) -> SlateResult:
        """Price a whole card, then re-size the survivors as a portfolio.

        Bets are screened one at a time (a bad bet does not become good in
        company), but the ones that pass are sized jointly, because they settle
        together.  Correlated legs — same game, same team, same weather — get
        cut hardest, which is the correct answer and the one that flat-staking
        a slate never gives you.
        """
        cfg = self.config
        bankroll = cfg.bankroll if bankroll is None else bankroll
        analyses = [self.price_bet(b, bankroll) for b in bets]

        keep = [(a, b) for a, b in zip(analyses, bets) if a.bet]
        if keep:
            legs = [
                Leg(name=a.name, p=a.blended_prob, decimal_odds=a.price_decimal, group=b.group)
                for a, b in keep
            ]
            result = simultaneous_kelly(
                legs,
                correlation=correlation,
                kelly_multiple=cfg.kelly_multiple,
                max_per_bet=cfg.max_bet_fraction,
                total_cap=cfg.max_total_exposure,
            )
            for (a, _), f_portfolio in zip(keep, result.fractions):
                # Never size a bet up because of the portfolio solve; the
                # standalone screen already applied the uncertainty haircut.
                f = min(f_portfolio, a.stake_fraction)
                stake = bankroll * f
                if cfg.round_to and stake > 0:
                    stake = round(stake / cfg.round_to) * cfg.round_to
                    f = stake / bankroll if bankroll else 0.0
                if f < a.stake_fraction - 1e-12:
                    a.reasons.append(
                        f"cut from {a.stake_fraction:.2%} to {f:.2%} by simultaneous "
                        "Kelly across the slate"
                    )
                a.stake_fraction, a.stake = f, stake
                a.units_held = (1.0 / f) if f > 0 else float("inf")
                a.growth_rate = (
                    log_growth_rate(f, a.blended_prob, net_odds(a.price_decimal))
                    if f > 0 else 0.0
                )
            portfolio_growth = result.growth_rate
        else:
            portfolio_growth = 0.0

        total_stake = sum(a.stake for a in analyses if a.bet)
        return SlateResult(
            analyses=analyses,
            total_stake=total_stake,
            total_exposure=total_stake / bankroll if bankroll else 0.0,
            portfolio_growth=portfolio_growth,
            expected_profit=sum(a.expected_profit for a in analyses if a.bet),
        )
