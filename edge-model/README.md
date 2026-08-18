# edgemodel

A sports betting edge model built on five formulas.

Same coin, same edge, same direction every time — and one bettor compounds
while the other ends at zero. The distance between them is not knowledge. It
is arithmetic, and this is that arithmetic, written as code you can run.

```
python3 -m edgemodel coin
```

Zero dependencies. Python 3.10+. Pure standard library.

---

## The five

| # | Formula | Question | Module |
|---|---------|----------|--------|
| 1 | `E = (W×A) − (L×B)` | Is there an edge at all? | `expectancy.py` |
| 2 | `σ = √(E[X²] − μ²)` | How loud is the noise around it? | `volatility.py` |
| 3 | `R = (q/p)^N` | Can the noise kill me before the edge pays? | `ruin.py` |
| 4 | `f* = (b·p − q)/b` | Exactly how much do I bet? | `kelly.py` |
| 5 | `g ≈ μ − σ²/2` | What do I actually keep? | `growth.py` |

There is one enemy in the whole story and it is variance. Formula 2 measures
it, formula 3 turns it into a probability of hitting zero, formula 4 sizes
against it, and formula 5 shows it being subtracted from every dollar you
compound. Formula 1 is only there to say the fight is worth having.

The library reproduces every number in that argument, and `tests/test_article.py`
asserts it:

```
edge          +0.1000        σ 0.9950        signal/noise 0.1005
ruin at 4 units    44.8%     at 20 units          1.8%
Kelly f*           0.1000    zero growth at       0.1987  (≈ 2×)
g(0.05) +0.375%   g(0.10) +0.501%   g(0.20) −0.014%   g(0.50) −8.891%
```

---

## What the coin does not tell you

The parable hands you a coin with `p = 0.55` stamped on it. Sports betting
never does, and four things stand between a model's output and a stake. Each
one is a module here.

**The price contains the margin.** A `-110/-110` market implies 104.8%, not
100%. Score your model against `1/price` and you will discover edges that are
pure vig. `devig.py` strips it four different ways — and on a lopsided market
those four disagree with each other by more than two probability points, which
is larger than most claimed edges:

```
$ python3 -m edgemodel devig -600 +425
multiplicative  0.8182   additive 0.8333   power 0.8423   shin 0.8333
methods disagree by 2.41 probability points on the first outcome.
If your claimed edge is smaller than that, you have found a de-vig artefact.
```

**Your probability is estimated, and you bet where it is most wrong.** You
place bets precisely where your model disagrees most with the market, which is
precisely where your model is most likely to be the one in error. That is the
winner's curse, and the defence is to shrink toward the de-vigged market price
in log-odds space (`model_weight`) and to size off a downside quantile of your
estimate rather than its mean (`sizing_quantile`).

A worked example of what that costs: a model claiming 58% on a `-110` line
looks like a 5.6-point edge over the 52.4% breakeven. De-vig the market to
49.5%, shrink with a 35% model weight, and you are left with 52.7% — a +0.28%
edge that fails a 1% threshold and is not worth the click.

> **A correction to the usual folklore.** Averaging expected growth over
> parameter uncertainty does *not* move the optimum — `g(f, p)` is linear in
> `p`, so the expectation is just `g(f, E[p])` and the posterior mean stays
> growth-optimal. Uncertainty matters for a different reason: it puts real mass
> on the region where your true edge is *negative*, where full Kelly is
> leverage on a losing bet. So the honest adjustment is not to average, it is
> to size off a quantile. See `kelly.kelly_quantile`.

**Bets settle simultaneously.** Eight games kick off at once, so "10% each" is
80% of your bankroll at risk against the chance they all lose together.
`portfolio.py` maximises log growth over the *joint* outcome distribution, and
correlated legs get cut hardest:

```
6 legs, same game, 55% each at -110
  ρ = 0.0   stake 5.42% each   total 32.5%
  ρ = 0.6   stake 1.79% each   total 11.0%     ← 0.33× the naive size
  ρ = 0.9   stake 1.33% each   total  7.3%
```

**You cannot tell whether it worked.** At a 2% edge, proving it from P&L takes
about 9,000 bets. Closing line value measures the same edge against a far
quieter yardstick and converges in tens. `validate.py` computes both, and the
ratio is usually 100× or more. Beat the closing line and you have an edge even
while your P&L is red; miss it and you do not have one while your P&L is green.

---

## Using it

```python
from edgemodel import EdgeModel, ModelConfig, Bet

model = EdgeModel(ModelConfig(
    bankroll=25_000,
    model_weight=0.35,     # you are a voice in the market, not sharper than it
    prob_sd=0.025,         # honest error bar on one game
    kelly_multiple=0.5,    # half Kelly
    min_edge=0.01,
    max_bet_fraction=0.02,
    max_total_exposure=0.15,
))

analysis = model.price_bet(Bet(
    "Bills ML",
    model_prob=0.64,
    price=+105,                     # the price you can actually get
    reference_prices=[-118, -102],  # the sharp market, your side first
))

print(analysis.verdict, analysis.stake, analysis.reasons)
```

A whole card, sized as a portfolio:

```python
result = model.price_slate(bets, correlation=0.55)
print(result.total_exposure, result.expected_profit)
```

Every rejection explains itself:

```
Chiefs -3.5   -105 (1.952)   [PASS]
  1 expectancy   model 58.50% -> market 49.55% -> blended 52.71% vs breakeven 51.22%
                 edge +2.904% per dollar (+1.49 pts), 72.4% chance it is real
  2 volatility   sigma 0.975, signal/noise 0.0298, 4,508 bets to prove it
  note           sized to zero: the 25% quantile of your estimate (51.02%) sits
                 below the breakeven price (51.22%), so the edge does not
                 survive its own error bar
```

### CLI

```
python3 -m edgemodel coin              # all five formulas on the weighted coin
python3 -m edgemodel devig -110 -110   # strip the margin from a market
python3 -m edgemodel price --p 0.58 --price -110 --market -108 -112
python3 -m edgemodel curve             # growth rate against stake size
python3 -m edgemodel ruin              # ruin and drawdown tables
python3 -m edgemodel simulate          # monte carlo the bankroll paths
```

### Examples

```
python3 examples/two_bettors.py   # the quant and the gambler, identical flips
python3 examples/nfl_slate.py     # a Sunday card through the whole pipeline
```

`two_bettors.py` deals both players the *same* thousand flips. The gambler
peaks at $21,669 and busts before flip 250. The quant finishes at $13,468. One
line of code differs between them.

### Tests

```
python3 -m unittest discover -s tests -t .
```

59 tests. `test_article.py` pins every number in the five-formula argument;
`test_library.py` covers odds, de-vig, sizing and the portfolio solver, and
checks the closed forms against Monte Carlo — flat-bet ruin lands within 2
points of `(q/p)^N`, and the median simulated path grows at exactly `g`.

---

## Two things worth knowing before you trust it

**The model is the hard part, and it is not in here.** This library sizes an
edge and tells you whether one survives contact with a real price. It does not
find one. Feeding it an optimistic `model_prob` produces confident, precisely
sized, losing bets — and the sizing machinery will make them *look* rigorous
while doing it. The defaults are deliberately sceptical for that reason:
`model_weight` starts at 0.35, not 1.0.

**Drawdown means decline from where you stand now.** `drawdown_probability`
measures the chance of falling to a fraction of *today's* bankroll, not of a
future high-water mark. Those differ enormously: at quarter Kelly your chance
of ever sitting 50% below today is under 1%, while your chance of eventually
sitting 50% below some future peak approaches certainty, because the peak keeps
climbing and every peak gets retraced. Use the closed form for the first
question and `simulate.py` for the second.

---

## Layout

```
edgemodel/
  odds.py         American / decimal / fractional, overround, hold
  devig.py        multiplicative, additive, power, Shin
  expectancy.py   formula 1
  volatility.py   formula 2, plus the normal CDF and its inverse
  ruin.py         formula 3, Lundberg exponent, Kelly drawdown
  kelly.py        formula 4, fractional and quantile-based sizing
  growth.py       formula 5, growth curve, ensemble vs time average
  portfolio.py    simultaneous Kelly, Gaussian copula for correlated legs
  model.py        the pipeline: EdgeModel, Bet, ModelConfig
  validate.py     Brier, calibration, CLV, how long until you know
  simulate.py     bankroll paths
  report.py       terminal rendering
  cli.py          python3 -m edgemodel
```

Learn the five and gambling quietly turns into a job.
