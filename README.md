# Coinflip Risk Simulator

A small Monte Carlo experiment exploring how different betting strategies behave in a simple coin-flip game.

The project compares:

- **Fixed-fraction betting** – always bet a constant percentage of current capital  
- **Random (YOLO) betting** – bet a random percentage of capital each round  

For each strategy, the simulator tracks:

- Equity curves (capital over time)
- Final wealth distribution
- Probability of (near) ruin
- Average max drawdown

This is a toy model, but it captures real ideas from risk management and position sizing in quantitative finance.

---

## Problem Setup

We consider a repeated coin-flip game:

- Each round, you bet some **fraction _f_ of your current capital**
- The coin has win probability `p_win` (0.5 by default)
- If you win, you gain `+f * capital`
- If you lose, you lose `-f * capital`

Capital is updated multiplicatively each round, so over time the process looks like geometric growth/decay.

---

## Strategies

### Fixed-Fraction Strategy

Always bet the same percentage of your current bankroll:

```python
def fixed_fraction_strategy(capital, t):
    return fixed_fraction  # e.g. 0.1 for 10%
```
This is similar to “bet 10% of your account on every trade” in a trading context.

YOLO Random Strategy
Bet a random percentage between a minimum and maximum each round:

```python
def yolo_random_strategy(capital, t):
    return np.random.uniform(yolo_min_frac, yolo_max_frac)
```

This is intentionally unstable: sometimes you bet very small, sometimes very large.
It shows how random, uncontrolled sizing can lead to large drawdowns and frequent near-ruin.

## How to Run
Requirements:

numpy

matplotlib

Install:

pip install numpy matplotlib

From the project root:

python coinflip_risk_sim.py
This will run the Monte Carlo experiment, print summary statistics, and show:

Sample equity curves for each strategy

A histogram of final wealth for fixed vs YOLO

## License
This project is licensed under the MIT License. See the LICENSE file for details.
