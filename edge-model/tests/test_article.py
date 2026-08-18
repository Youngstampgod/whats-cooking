"""Every number quoted in the five-formulas argument, checked against the code.

If this file passes, the library reproduces the parable exactly. It is the
regression test that matters most: the formulas are the product.
"""

import unittest

from edgemodel import (
    bet_std,
    bets_to_significance,
    ensemble_vs_time,
    expectancy,
    growth_approx,
    kelly_fraction,
    log_growth_rate,
    risk_of_ruin,
    risk_of_ruin_even_money,
    signal_to_noise,
    zero_growth_fraction,
)

P, B = 0.55, 1.0


class TestFormula1Expectancy(unittest.TestCase):
    def test_coin_pays_ten_cents_on_the_dollar(self):
        # E = (0.55 x 1) - (0.45 x 1) = +0.10
        self.assertAlmostEqual(expectancy(P, B), 0.10, places=10)

    def test_expectancy_is_negative_on_the_wrong_side(self):
        self.assertAlmostEqual(expectancy(1 - P, B), -0.10, places=10)

    def test_at_minus_110_the_bar_is_higher_than_a_coin_flip(self):
        from edgemodel import breakeven_win_rate

        self.assertAlmostEqual(breakeven_win_rate(1.909090909), 0.5238, places=4)


class TestFormula2Volatility(unittest.TestCase):
    def test_sigma_is_about_one_dollar(self):
        # "edge = 0.10, sigma ~= 0.99"
        self.assertAlmostEqual(bet_std(P, B), 0.99499, places=5)
        self.assertAlmostEqual(bet_std(P, B) ** 2, 0.99, places=10)

    def test_signal_to_noise_is_about_one_tenth(self):
        self.assertAlmostEqual(signal_to_noise(P, B), 0.1005, places=4)

    def test_the_edge_takes_hundreds_of_bets_to_appear(self):
        n = bets_to_significance(P, B, z=2.0)
        self.assertAlmostEqual(n, 396.0, delta=1.0)


class TestFormula3RiskOfRuin(unittest.TestCase):
    def test_q_over_p_is_point_eight_two(self):
        self.assertAlmostEqual((1 - P) / P, 0.8182, places=4)

    def test_four_units_is_a_forty_five_percent_chance_of_zero(self):
        # "N = 4 : ruin 45%"
        self.assertAlmostEqual(risk_of_ruin_even_money(P, 4), 0.448, places=3)

    def test_twenty_units_is_under_two_percent(self):
        # "N = 20 : ruin under 2%"
        self.assertLess(risk_of_ruin_even_money(P, 20), 0.02)
        self.assertAlmostEqual(risk_of_ruin_even_money(P, 20), 0.0181, places=4)

    def test_general_form_reduces_to_the_closed_form_at_even_money(self):
        for units in (1, 4, 7, 20, 50):
            self.assertAlmostEqual(
                risk_of_ruin(P, B, units),
                risk_of_ruin_even_money(P, units),
                places=10,
            )


class TestFormula4Kelly(unittest.TestCase):
    def test_kelly_says_ten_percent(self):
        # f* = (1 x 0.55 - 0.45) / 1 = 0.10
        self.assertAlmostEqual(kelly_fraction(P, B), 0.10, places=10)

    def test_kelly_declines_a_negative_edge_entirely(self):
        self.assertEqual(kelly_fraction(0.45, B), 0.0)

    def test_kelly_is_the_argmax_of_the_growth_rate(self):
        f_star = kelly_fraction(P, B)
        best = max(
            (log_growth_rate(i / 10000, P, B), i / 10000) for i in range(1, 9000)
        )[1]
        self.assertAlmostEqual(best, f_star, places=3)


class TestFormula5GeometricGrowth(unittest.TestCase):
    def test_every_quoted_growth_rate(self):
        # "5% -> +0.38%, 10% -> +0.50%, 20% -> zero,
        #  35% -> -2.9%, 50% -> -8.9%"
        cases = {
            0.05: 0.0038,
            0.10: 0.0050,
            0.20: 0.0000,
            0.35: -0.0290,
            0.50: -0.0889,
        }
        for f, expected in cases.items():
            self.assertAlmostEqual(
                log_growth_rate(f, P, B), expected, places=3,
                msg=f"growth rate at f={f}",
            )

    def test_growth_peaks_at_kelly(self):
        peak = log_growth_rate(kelly_fraction(P, B), P, B)
        self.assertAlmostEqual(peak, 0.005008, places=6)
        for f in (0.05, 0.08, 0.12, 0.15, 0.20, 0.35):
            self.assertLess(log_growth_rate(f, P, B), peak)

    def test_growth_returns_to_zero_at_about_twice_kelly(self):
        f_zero = zero_growth_fraction(P, B)
        self.assertAlmostEqual(f_zero, 0.1987, places=4)
        self.assertAlmostEqual(f_zero / kelly_fraction(P, B), 2.0, delta=0.02)
        self.assertAlmostEqual(log_growth_rate(f_zero, P, B), 0.0, places=9)

    def test_mu_minus_half_variance_approximates_the_exact_rate(self):
        for f in (0.02, 0.05, 0.10, 0.15):
            self.assertAlmostEqual(
                growth_approx(f, P, B), log_growth_rate(f, P, B), places=3
            )

    def test_the_ensemble_average_grows_while_the_path_dies(self):
        # The whole point of formula 5: at half the bankroll per bet the mean
        # ending wealth is astronomical and the median is zero.
        r = ensemble_vs_time(0.50, P, B, n=1000)
        self.assertGreater(r["ensemble_mean"], 1e20)
        self.assertLess(r["median_path"], 1e-30)
        # And it is still a positive-expectancy bet on every single flip.
        self.assertGreater(expectancy(P, B), 0)


class TestTheFormulasAgree(unittest.TestCase):
    """The five are four views of one enemy, so they must be consistent."""

    def test_kelly_equals_edge_over_variance(self):
        from edgemodel.kelly import kelly_variance_form

        self.assertAlmostEqual(kelly_variance_form(P, B), kelly_fraction(P, B), places=2)

    def test_lundberg_exponent_is_log_p_over_q(self):
        from math import log

        from edgemodel.ruin import lundberg_exponent

        self.assertAlmostEqual(lundberg_exponent(P, B), log(P / (1 - P)), places=10)

    def test_zero_growth_and_certain_drawdown_are_the_same_wall(self):
        from edgemodel.ruin import drawdown_probability

        # Formula 5 puts zero growth at 2x Kelly; formula 3 puts certain
        # drawdown at the same place. Same variance, two descriptions.
        self.assertEqual(drawdown_probability(0.5, 2.0), 1.0)
        self.assertAlmostEqual(zero_growth_fraction(P, B) / kelly_fraction(P, B), 2.0, delta=0.02)

    def test_growth_kept_at_fractional_kelly(self):
        from edgemodel.kelly import growth_fraction_of_max

        self.assertAlmostEqual(growth_fraction_of_max(0.5), 0.75, places=10)
        self.assertAlmostEqual(growth_fraction_of_max(2.0), 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
