"""A sports betting edge model built on five formulas.

    1. Expectancy       is there an edge at all?          E = (W*A) - (L*B)
    2. Volatility       how loud is the noise?            s = sqrt(E[X^2] - u^2)
    3. Risk of ruin     can the noise kill me first?      R = (q/p)^N
    4. Kelly            how much do I bet?                f* = (b*p - q)/b
    5. Geometric growth what do I actually keep?          g ~= u - s^2/2

There is one enemy in the whole story and it is variance.  Volatility measures
it, risk of ruin turns it into a probability of hitting zero, Kelly sizes
against it, and geometric growth shows it being subtracted from every dollar
you compound.  Expectancy is only there to tell you the fight is worth having.

Quick start::

    from edgemodel import EdgeModel, ModelConfig, Bet

    model = EdgeModel(ModelConfig(bankroll=10_000, kelly_multiple=0.5))
    bet = Bet("Chiefs -3", model_prob=0.58, price=-110,
              reference_prices=[-108, -112])
    print(model.price_bet(bet).verdict)
"""

from .devig import devig, no_vig_prices
from .expectancy import breakeven_win_rate, edge, expectancy, expected_value
from .growth import (
    doubling_time,
    ensemble_vs_time,
    growth_approx,
    growth_curve,
    log_growth_rate,
    zero_growth_fraction,
)
from .kelly import (
    fractional_kelly,
    growth_fraction_of_max,
    implied_kelly_multiple,
    kelly_fraction,
    kelly_from_odds,
    kelly_quantile,
    kelly_stake,
    probability_edge_is_real,
)
from .model import Bet, BetAnalysis, EdgeModel, ModelConfig, SlateResult
from .odds import (
    american_to_decimal,
    american_to_prob,
    decimal_to_american,
    decimal_to_prob,
    hold,
    net_odds,
    overround,
    parse_odds,
    prob_to_american,
    prob_to_decimal,
)
from .portfolio import Leg, simultaneous_kelly
from .ruin import (
    drawdown_for_probability,
    drawdown_probability,
    max_bet_fraction_for_ruin,
    risk_of_ruin,
    risk_of_ruin_even_money,
    units_for_ruin_target,
)
from .simulate import compare_stakes, simulate_flat, simulate_kelly
from .validate import (
    bets_needed,
    brier_score,
    calibration_table,
    clv_percent,
    clv_probability,
    log_loss,
    summarize_clv,
    validation_comparison,
)
from .volatility import (
    bet_std,
    bet_variance,
    bets_to_significance,
    prob_behind_after,
    sharpe_after,
    signal_to_noise,
)

__version__ = "1.0.0"

__all__ = [
    "EdgeModel", "ModelConfig", "Bet", "BetAnalysis", "SlateResult",
    "expectancy", "edge", "expected_value", "breakeven_win_rate",
    "bet_variance", "bet_std", "signal_to_noise", "sharpe_after",
    "bets_to_significance", "prob_behind_after",
    "risk_of_ruin", "risk_of_ruin_even_money", "units_for_ruin_target",
    "max_bet_fraction_for_ruin", "drawdown_probability", "drawdown_for_probability",
    "kelly_fraction", "kelly_from_odds", "fractional_kelly", "kelly_stake",
    "kelly_quantile", "probability_edge_is_real", "growth_fraction_of_max",
    "implied_kelly_multiple",
    "log_growth_rate", "growth_approx", "growth_curve", "zero_growth_fraction",
    "doubling_time", "ensemble_vs_time",
    "american_to_decimal", "decimal_to_american", "decimal_to_prob",
    "prob_to_decimal", "american_to_prob", "prob_to_american", "net_odds",
    "parse_odds", "overround", "hold",
    "devig", "no_vig_prices",
    "Leg", "simultaneous_kelly",
    "simulate_flat", "simulate_kelly", "compare_stakes",
    "brier_score", "log_loss", "calibration_table", "clv_percent",
    "clv_probability", "summarize_clv", "bets_needed", "validation_comparison",
]
