# Volatility Drag: What Position Sizing Cannot Fix

A Monte Carlo experiment on a **fair** coin-flip game, showing that betting a fixed
fraction of your capital loses money even when the game has no edge against you.

Start with 100 units. Bet 10% of your current capital on a 50/50 coin at even money, two
hundred times. Nothing about this game favors the house — expected *wealth* is flat.

The median outcome is **36.60**.

That is not variance or bad luck. It is arithmetic, and it is the single most important
thing to understand before thinking about position sizing at all.

---

## Why it happens

Win then lose, in either order, and your capital is multiplied by

```
(1 + f)(1 - f) = 1 - f²
```

which is less than 1 for any nonzero `f`. A 10% gain followed by a 10% loss does not
return you to where you started — it leaves you 1% down. Repeat that and the shortfall
compounds. This is **volatility drag** (also called variance drain).

Formally, the expected log-growth per round is

```
g(f) = p·ln(1 + f) + (1 - p)·ln(1 - f)
```

At `p = 0.5, f = 0.10`:

```
g = 0.5·ln(1.10) + 0.5·ln(0.90) = -0.005025 per round
```

Over 200 rounds, `100 · exp(200 · g) = 36.60`.

**The simulation returns 36.60.** Theory and 2,000 simulated paths agree to the cent —
which is the point of running the simulation at all: not to discover the result, but to
confirm the model is doing what the algebra says it should.

## The mean is not the story

| Strategy | Mean final wealth | Median final wealth |
|---|---:|---:|
| Fixed 10% | 97.17 | **36.60** |

Expected *wealth* is roughly preserved. Expected *log wealth* is not. The mean is held up
by a small number of paths that ran away upward, while the typical path is down by nearly
two thirds — and you only get to live one path.

Any time capital compounds, the mean is the wrong summary statistic. This is the same
reason Kelly sizing optimizes expected log wealth rather than expected wealth.

## How much is too much?

Since `g(f) < 0` for every `f > 0` when `p = 0.5`, the growth-optimal bet on a fair game
is to **not bet**. The cost of betting anyway rises roughly as `f²`:

| f | log-growth / round | median after 200 rounds |
|---|---:|---:|
| 0.00 | +0.00000 | 100.00 |
| 0.01 | −0.00005 | 99.00 |
| 0.02 | −0.00020 | 96.08 |
| 0.05 | −0.00125 | 77.86 |
| 0.10 | −0.00503 | 36.60 |
| 0.20 | −0.02041 | 1.69 |
| 0.30 | −0.04716 | 0.01 |
| 0.50 | −0.14384 | 0.00 |

Doubling the bet size roughly quadruples the drag. At 20% of capital per flip, the median
player is effectively wiped out inside 200 rounds of a *fair* game.

## Secondary comparison: uncontrolled sizing

For contrast the simulator also runs a strategy that bets a uniformly random fraction
between 0% and 50% each round:

| Strategy | Mean final | Median final | P(near ruin) | Avg max drawdown |
|---|---:|---:|---:|---:|
| Fixed 10% | 97.17 | 36.60 | 0.008 | 0.841 |
| Random 0–50% | 154.13 | **0.01** | 0.835 | 0.998 |

The median finishes at one cent. The mean is the *highest* of the two, which is the same
mean-versus-median trap in a more extreme form.

This comparison is deliberately secondary. Nobody defends random position sizing, so
beating it proves little — the fixed-fraction result above is the one that matters,
because fixed-fraction sizing is what people actually do.

Note that "P(near ruin)" is the fraction of paths finishing at or below 1 unit from a
start of 100. Because betting is a fixed fraction of *current* capital and the maximum
fraction is 50%, wealth is multiplicative and never reaches exactly zero.

## Setup

- 2,000 paths, 200 rounds, starting capital 100, `seed = 42`
- `p_win = 0.5`, even money: win `+f · capital`, lose `−f · capital`
- Fixed-fraction `f = 0.10`; random strategy draws `f ~ U(0, 0.5)` each round

## Run

```bash
pip install numpy matplotlib
python coinflip_risk_sim.py
```

Prints summary statistics and plots sample equity curves plus the final-wealth
distribution for both strategies.

## Companion project

This repo covers the case where you have **no** edge. The natural sequel is what to do
when you **do**: [kelly-betting-simulator](https://github.com/bpatel1121/kelly-betting-simulator)
takes `p = 0.55` and shows that the growth-optimal fraction is `f* = 2p − 1 = 0.10`, that
Full Kelly maximizes long-run log-growth, and that betting twice Kelly turns growth
negative *despite* the positive edge.

Read together: drag is what you fight, and an edge is what buys you the right to bet at
all. Sizing determines whether you keep it.

## License

MIT. See [LICENSE](LICENSE).
