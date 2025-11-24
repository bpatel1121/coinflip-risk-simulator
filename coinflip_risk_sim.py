import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Parameters
# --------------------------------------------------
np.random.seed(42)          # for reproducibility

p_win = 0.5                 # probability of winning each bet
n_rounds = 200              # number of bets per path
n_paths = 2000              # number of simulated paths
initial_capital = 100.0     # starting amount

fixed_fraction = 0.1        # 10% of current capital each round
yolo_min_frac = 0.0         # YOLO strategy: random fraction between 0% and 50% of capital each round
yolo_max_frac = 0.5        

ruin_threshold = 1.0        # consider "ruin" when capital less than or equal to original amouont


# --------------------------------------------------
# Core simulation functions
# --------------------------------------------------
def simulate_path(strategy_func, n_rounds, initial_capital, p_win):
    """
    Simulate a single equity path for a given betting strategy.
    
    strategy_func: function(capital, t) -> fraction_of_capital_to_bet
    """
    capital = initial_capital
    equity_curve = [capital]

    for t in range(n_rounds):
        if capital <= 0:
            equity_curve.append(0.0)  # Once broke cannot gain money
            continue

        frac = strategy_func(capital, t)
        frac = max(0.0, min(frac, 1.0))  # clamp 0–1

        bet_size = frac * capital

        # coin flip: +bet_size on win, -bet_size on loss
        outcome = np.random.rand() < p_win
        if outcome:
            capital += bet_size
        else:
            capital -= bet_size

        equity_curve.append(capital)

    return np.array(equity_curve)


def simulate_many(strategy_func, n_paths, n_rounds, initial_capital, p_win):
    """
    Simulate many paths and return matrix of shape (n_paths, n_rounds+1).
    """
    paths = np.zeros((n_paths, n_rounds + 1))
    for i in range(n_paths):
        paths[i] = simulate_path(strategy_func, n_rounds, initial_capital, p_win)
    return paths


def max_drawdown(equity_curve):
    """
    Compute max drawdown of a single equity curve.
    """
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = 1.0 - equity_curve / running_max
    return np.max(drawdowns)


# --------------------------------------------------
# Strategy definitions
# --------------------------------------------------
def fixed_fraction_strategy(capital, t):
    """
    Bet a fixed fraction of current capital each round.
    """
    return fixed_fraction


def yolo_random_strategy(capital, t):
    """
    Bet a random fraction between yolo_min_frac and yolo_max_frac.
    Sometimes tiny, sometimes huge = very swingy.
    """
    return np.random.uniform(yolo_min_frac, yolo_max_frac)


# --------------------------------------------------
# Run simulations
# --------------------------------------------------
paths_fixed = simulate_many(
    fixed_fraction_strategy, n_paths, n_rounds, initial_capital, p_win
)
paths_yolo = simulate_many(
    yolo_random_strategy, n_paths, n_rounds, initial_capital, p_win
)

final_fixed = paths_fixed[:, -1]
final_yolo = paths_yolo[:, -1]

# Metrics
def summarize_results(name, final_values, paths):
    avg_final = np.mean(final_values)
    med_final = np.median(final_values)
    prob_ruin = np.mean(final_values <= ruin_threshold)
    avg_max_dd = np.mean([max_drawdown(p) for p in paths])

    print(f"=== {name} ===")
    print(f"Average final wealth: {avg_final:8.2f}")
    print(f"Median final wealth : {med_final:8.2f}")
    print(f"Probability of ruin : {prob_ruin:8.3f}")
    print(f"Avg max drawdown    : {avg_max_dd:8.3f}")
    print()

summarize_results("Fixed Fraction Strategy", final_fixed, paths_fixed)
summarize_results("YOLO Random Strategy", final_yolo, paths_yolo)


# --------------------------------------------------
# Plot a few sample equity curves
# --------------------------------------------------
n_show = 10  # how many paths to plot for illustration

plt.figure(figsize=(10, 5))
for i in range(n_show):
    plt.plot(paths_fixed[i], alpha=0.7)
plt.title("Fixed Fraction Strategy - Sample Equity Curves")
plt.xlabel("Round")
plt.ylabel("Capital")
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(10, 5))
for i in range(n_show):
    plt.plot(paths_yolo[i], alpha=0.7)
plt.title("YOLO Random Strategy - Sample Equity Curves")
plt.xlabel("Round")
plt.ylabel("Capital")
plt.grid(True)
plt.tight_layout()

# --------------------------------------------------
# Plot distribution of final wealth
# --------------------------------------------------
plt.figure(figsize=(10, 5))
plt.hist(final_fixed, bins=50, histtype="step", linewidth=2, label="Fixed Fraction")
plt.hist(final_yolo, bins=50, alpha=0.5, label="YOLO Random")
plt.axvline(initial_capital, linestyle="--", label="Initial Capital")
plt.title("Distribution of Final Wealth")
plt.xlabel("Final Capital")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
