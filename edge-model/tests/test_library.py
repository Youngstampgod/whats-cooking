"""Behaviour of the sports-betting layer: odds, de-vig, sizing, portfolio."""

import unittest

from edgemodel import (
    Bet,
    EdgeModel,
    Leg,
    ModelConfig,
    american_to_decimal,
    decimal_to_american,
    devig,
    hold,
    kelly_fraction,
    no_vig_prices,
    overround,
    parse_odds,
    simultaneous_kelly,
)


class TestOdds(unittest.TestCase):
    def test_american_to_decimal(self):
        self.assertAlmostEqual(american_to_decimal(-110), 1.9090909, places=6)
        self.assertAlmostEqual(american_to_decimal(+150), 2.5, places=10)
        self.assertAlmostEqual(american_to_decimal(-200), 1.5, places=10)

    def test_round_trip(self):
        for a in (-1000, -300, -110, -101, 100, 137, 425, 2000):
            self.assertAlmostEqual(
                decimal_to_american(american_to_decimal(a)), a, places=6
            )

    def test_rejects_impossible_american_prices(self):
        for bad in (0, 50, -99, 99):
            with self.assertRaises(ValueError):
                american_to_decimal(bad)

    def test_parse_odds_detects_format(self):
        self.assertAlmostEqual(parse_odds("-110"), 1.9090909, places=6)
        self.assertAlmostEqual(parse_odds(-110), 1.9090909, places=6)
        self.assertAlmostEqual(parse_odds(1.91), 1.91, places=10)
        self.assertAlmostEqual(parse_odds("10/11"), 1.9090909, places=6)
        self.assertAlmostEqual(parse_odds(150), 2.5, places=10)

    def test_parse_odds_refuses_the_ambiguous_middle(self):
        with self.assertRaises(ValueError):
            parse_odds(0.9)

    def test_overround_and_hold_differ(self):
        market = [american_to_decimal(-110)] * 2
        self.assertAlmostEqual(overround(market), 0.047619, places=6)
        self.assertAlmostEqual(hold(market), 0.045455, places=6)
        self.assertLess(hold(market), overround(market))


class TestDevig(unittest.TestCase):
    def setUp(self):
        self.even = [american_to_decimal(-110)] * 2
        self.lopsided = [american_to_decimal(-600), american_to_decimal(+425)]

    def test_all_methods_sum_to_one(self):
        for method in ("multiplicative", "additive", "power", "shin"):
            for market in (self.even, self.lopsided):
                self.assertAlmostEqual(sum(devig(market, method)), 1.0, places=9)

    def test_symmetric_market_is_a_coin_under_every_method(self):
        for method in ("multiplicative", "additive", "power", "shin"):
            self.assertAlmostEqual(devig(self.even, method)[0], 0.5, places=9)

    def test_methods_disagree_on_lopsided_markets(self):
        favourites = [devig(self.lopsided, m)[0] for m in
                      ("multiplicative", "additive", "power", "shin")]
        # This spread is the reason a 1% "edge" on a heavy favourite is noise.
        self.assertGreater(max(favourites) - min(favourites), 0.02)

    def test_zero_margin_market_is_returned_unchanged(self):
        fair = [2.0, 2.0]
        for method in ("multiplicative", "additive", "power", "shin"):
            self.assertAlmostEqual(devig(fair, method)[0], 0.5, places=8)

    def test_no_vig_prices_are_longer_than_quoted(self):
        for quoted, fair in zip(self.even, no_vig_prices(self.even)):
            self.assertGreater(fair, quoted)

    def test_rejects_a_single_price(self):
        with self.assertRaises(ValueError):
            devig([1.91])

    def test_rejects_an_arbitrage(self):
        with self.assertRaises(ValueError):
            devig([2.5, 2.5])

    def test_three_way_market(self):
        soccer = [2.40, 3.40, 3.10]
        probs = devig(soccer, "shin")
        self.assertAlmostEqual(sum(probs), 1.0, places=9)
        self.assertEqual(len(probs), 3)


class TestEdgeModel(unittest.TestCase):
    def setUp(self):
        self.cfg = ModelConfig(bankroll=10_000, model_weight=1.0, prob_sd=0.0,
                               min_edge=0.0, min_confidence=0.0, max_bet_fraction=1.0,
                               kelly_multiple=1.0, round_to=None)

    def test_with_full_confidence_and_no_shrinkage_it_is_plain_kelly(self):
        model = EdgeModel(self.cfg)
        a = model.price_bet(Bet("coin", model_prob=0.55, price=2.0))
        self.assertAlmostEqual(a.stake_fraction, kelly_fraction(0.55, 1.0), places=9)
        self.assertAlmostEqual(a.stake, 1000.0, places=6)
        self.assertEqual(a.verdict, "BET")

    def test_shrinkage_pulls_the_model_toward_the_market(self):
        model = EdgeModel(ModelConfig(model_weight=0.35))
        a = model.price_bet(
            Bet("x", model_prob=0.58, price=-110, reference_prices=[-108, -112])
        )
        self.assertLess(a.blended_prob, 0.58)
        self.assertGreater(a.blended_prob, a.market_prob)

    def test_a_confident_looking_edge_can_vanish_after_devig(self):
        # 58% against a market that says 49.5% is a 5.6-point "edge" that is
        # worth +0.28% per dollar once shrunk -- below any sane threshold.
        model = EdgeModel(ModelConfig())
        a = model.price_bet(
            Bet("x", model_prob=0.58, price=-110, reference_prices=[-108, -112])
        )
        self.assertEqual(a.verdict, "PASS")
        self.assertEqual(a.stake, 0.0)
        self.assertTrue(any("below" in r for r in a.reasons))

    def test_negative_edge_is_refused(self):
        model = EdgeModel(self.cfg)
        a = model.price_bet(Bet("bad", model_prob=0.40, price=-110))
        self.assertEqual(a.verdict, "PASS")
        self.assertEqual(a.stake, 0.0)

    def test_uncertainty_shrinks_the_stake(self):
        certain = EdgeModel(ModelConfig(prob_sd=0.0, model_weight=1.0, min_edge=0.0,
                                        min_confidence=0.0, max_bet_fraction=1.0,
                                        kelly_multiple=1.0, round_to=None))
        unsure = EdgeModel(ModelConfig(prob_sd=0.03, model_weight=1.0, min_edge=0.0,
                                       min_confidence=0.0, max_bet_fraction=1.0,
                                       kelly_multiple=1.0, round_to=None))
        bet = Bet("x", model_prob=0.60, price=2.0)
        self.assertGreater(
            certain.price_bet(bet).stake_fraction,
            unsure.price_bet(bet).stake_fraction,
        )

    def test_caps_are_enforced(self):
        model = EdgeModel(ModelConfig(max_bet_fraction=0.01, model_weight=1.0,
                                      prob_sd=0.0, min_edge=0.0, min_confidence=0.0,
                                      round_to=None))
        a = model.price_bet(Bet("x", model_prob=0.70, price=2.0))
        self.assertAlmostEqual(a.stake_fraction, 0.01, places=9)
        self.assertTrue(any("capped" in r for r in a.reasons))

    def test_kelly_multiple_above_two_is_refused_outright(self):
        with self.assertRaises(ValueError):
            ModelConfig(kelly_multiple=2.5)

    def test_invalid_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            Bet("x", model_prob=1.4, price=2.0)

    def test_slate_respects_the_total_exposure_cap(self):
        model = EdgeModel(ModelConfig(bankroll=10_000, model_weight=1.0, prob_sd=0.0,
                                      min_edge=0.0, min_confidence=0.0,
                                      max_bet_fraction=0.10, max_total_exposure=0.20,
                                      kelly_multiple=1.0, round_to=None))
        bets = [Bet(f"g{i}", model_prob=0.60, price=2.0) for i in range(8)]
        result = model.price_slate(bets)
        self.assertLessEqual(result.total_exposure, 0.20 + 1e-9)
        self.assertEqual(len(result.bets), 8)


class TestPortfolio(unittest.TestCase):
    def test_one_leg_reproduces_plain_kelly(self):
        r = simultaneous_kelly([Leg("coin", 0.55, 2.0)])
        self.assertAlmostEqual(r.fractions[0], kelly_fraction(0.55, 1.0), places=5)

    def test_simultaneous_bets_are_sized_below_naive_kelly(self):
        legs = [Leg(f"g{i}", 0.55, 2.0) for i in range(6)]
        r = simultaneous_kelly(legs, total_cap=0.95)
        self.assertLess(r.fractions[0], kelly_fraction(0.55, 1.0))

    def test_correlation_shrinks_stakes_hard(self):
        legs = [Leg(f"leg{i}", 0.55, 2.0, group="same-game") for i in range(4)]
        independent = simultaneous_kelly(legs, correlation=0.0)
        correlated = simultaneous_kelly(legs, correlation=0.8)
        self.assertLess(correlated.total_exposure, independent.total_exposure * 0.6)

    def test_negative_edge_legs_get_nothing(self):
        legs = [Leg("good", 0.60, 2.0), Leg("bad", 0.40, 2.0)]
        r = simultaneous_kelly(legs, total_cap=0.95)
        self.assertGreater(r.fractions[0], 0.0)
        self.assertAlmostEqual(r.fractions[1], 0.0, places=6)

    def test_zero_correlation_with_groups_uses_exact_enumeration(self):
        legs = [Leg(f"leg{i}", 0.55, 2.0, group="g") for i in range(3)]
        r = simultaneous_kelly(legs, correlation=0.0)
        self.assertEqual(r.scenarios_used, 8)  # 2^3 exactly, not sampled

    def test_total_cap_is_respected(self):
        legs = [Leg(f"g{i}", 0.70, 2.0) for i in range(10)]
        r = simultaneous_kelly(legs, total_cap=0.30)
        self.assertLessEqual(r.total_exposure, 0.30 + 1e-9)


class TestValidation(unittest.TestCase):
    def test_brier_rewards_calibration(self):
        from edgemodel import brier_score

        good = brier_score([0.9, 0.9, 0.9, 0.1], [1, 1, 1, 0])
        bad = brier_score([0.5, 0.5, 0.5, 0.5], [1, 1, 1, 0])
        self.assertLess(good, bad)

    def test_clv_percent(self):
        from edgemodel import clv_percent

        # Bet +100, closed -110: you beat the close.
        self.assertGreater(clv_percent(100, -110), 0)
        self.assertLess(clv_percent(-110, 100), 0)

    def test_clv_proves_an_edge_far_faster_than_pnl(self):
        from edgemodel import validation_comparison

        r = validation_comparison(edge=0.02, sigma_pnl=0.95,
                                  mean_clv=0.01, sd_clv=0.025)
        self.assertGreater(r["bets_via_pnl"], 8000)
        self.assertLess(r["bets_via_clv"], 40)
        self.assertGreater(r["speedup"], 100)

    def test_clv_summary_flags_negative_clv(self):
        from edgemodel import summarize_clv

        s = summarize_clv([-0.01] * 100)
        self.assertLess(s.t_stat, 0)
        self.assertIn("not an edge", s.verdict)


class TestSimulationMatchesTheory(unittest.TestCase):
    """The closed forms are only worth having if the paths agree with them."""

    def test_flat_bet_ruin_matches_the_closed_form(self):
        import random

        from edgemodel import risk_of_ruin_even_money

        p, units, paths = 0.55, 4, 4000
        rng = random.Random(3)
        ruined = 0
        for _ in range(paths):
            w = units
            for _ in range(3000):
                w += 1 if rng.random() < p else -1
                if w <= 0:
                    ruined += 1
                    break
        self.assertAlmostEqual(ruined / paths, risk_of_ruin_even_money(p, units),
                               delta=0.02)

    def test_median_path_grows_at_the_geometric_rate(self):
        from math import exp

        from edgemodel import log_growth_rate, simulate_kelly

        r = simulate_kelly(0.55, 1.0, fraction=0.10, bankroll=1000,
                           bets=500, paths=1500, seed=9)
        expected = 1000 * exp(log_growth_rate(0.10, 0.55, 1.0) * 500)
        self.assertAlmostEqual(r.median_terminal / expected, 1.0, delta=0.15)

    def test_overbetting_kills_the_path_you_actually_live(self):
        from edgemodel import simulate_kelly

        r = simulate_kelly(0.55, 1.0, fraction=0.40, bankroll=1000,
                           bets=500, paths=1500, seed=9)
        self.assertLess(r.median_terminal, 1000)
        self.assertLess(r.p05_terminal, 1.0)
        # ...on a bet with positive expectancy on every single flip.
        from edgemodel import expectancy
        self.assertGreater(expectancy(0.55, 1.0), 0)

    def test_the_ensemble_average_is_not_merely_unlived_but_unobservable(self):
        """A sharper version of formula 5 than the parable states.

        The ensemble mean at 4x Kelly is ~2.5e11, and it is carried entirely by
        paths roughly nine standard deviations into the right tail. Simulate
        fifteen hundred bettors and not one of them produces such a path, so
        the sampled mean comes in astronomically *below* the true mean and
        below the starting bankroll besides. The number in the pitch deck is
        not just a number nobody lives in -- at any realistic sample size it is
        a number nobody even measures.
        """
        from edgemodel import ensemble_vs_time, simulate_kelly

        theory = ensemble_vs_time(0.40, 0.55, 1.0, n=500)
        r = simulate_kelly(0.55, 1.0, fraction=0.40, bankroll=1000,
                           bets=500, paths=1500, seed=9)
        self.assertGreater(theory["ensemble_mean"], 1e8)
        self.assertLess(r.mean_terminal, 1000)
        self.assertGreater(theory["ensemble_mean"] * 1000 / r.mean_terminal, 1e8)


if __name__ == "__main__":
    unittest.main()
