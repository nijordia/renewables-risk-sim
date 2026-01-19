"""
Renewables Risk Simulator - Analysis Module

Part 2: Exploratory Analysis
Part 3: Linear Regression
Part 4: Monte Carlo Simulation
"""

import json
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from utils import print_header


# ============================================================================
# Part 2: Exploratory Analysis
# ============================================================================

def exploratory_analysis(df: pd.DataFrame, output_dir: str = "/outputs") -> dict:
    """
    Perform exploratory data analysis on renewable share vs price.

    - Scatter plot: renewable share vs price
    - Pearson correlation coefficient
    - Price volatility comparison: high renewable days (>60%) vs low

    Args:
        df: DataFrame with renewable_share and price_daily_avg columns
        output_dir: Directory for output plots

    Returns:
        Dictionary with analysis results
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {}

    print_header("Part 2: Exploratory Analysis")

    # 1. Pearson correlation
    corr, p_value = stats.pearsonr(df["renewable_share"], df["price_daily_avg"])
    results["pearson_correlation"] = corr
    results["pearson_p_value"] = p_value

    print(f"Pearson Correlation: {corr:.4f}")
    print(f"P-value: {p_value:.2e}")
    if p_value < 0.05:
        print("  -> Statistically significant (p < 0.05)")
    if corr < 0:
        print("  -> Negative correlation: higher renewable share = lower prices")
    else:
        print("  -> Positive correlation: higher renewable share = higher prices")

    # 2. Volatility comparison: high vs low renewable days
    high_renewable_threshold = 60
    high_renewable = df[df["renewable_share"] >= high_renewable_threshold]
    low_renewable = df[df["renewable_share"] < high_renewable_threshold]

    high_vol = high_renewable["price_daily_avg"].std()
    low_vol = low_renewable["price_daily_avg"].std()
    high_mean = high_renewable["price_daily_avg"].mean()
    low_mean = low_renewable["price_daily_avg"].mean()

    results["high_renewable_threshold"] = high_renewable_threshold
    results["high_renewable_days"] = len(high_renewable)
    results["low_renewable_days"] = len(low_renewable)
    results["high_renewable_price_mean"] = high_mean
    results["low_renewable_price_mean"] = low_mean
    results["high_renewable_price_std"] = high_vol
    results["low_renewable_price_std"] = low_vol
    results["volatility_ratio"] = high_vol / low_vol if low_vol > 0 else None

    print()
    print(f"Price Volatility Comparison (threshold: {high_renewable_threshold}% renewable):")
    print(f"  High renewable days (>={high_renewable_threshold}%): n={len(high_renewable)}")
    print(f"    Mean price: {high_mean:.2f} EUR/MWh")
    print(f"    Std dev:    {high_vol:.2f} EUR/MWh")
    print(f"  Low renewable days (<{high_renewable_threshold}%): n={len(low_renewable)}")
    print(f"    Mean price: {low_mean:.2f} EUR/MWh")
    print(f"    Std dev:    {low_vol:.2f} EUR/MWh")
    if results["volatility_ratio"]:
        print(f"  Volatility ratio (high/low): {results['volatility_ratio']:.2f}")

    # 3. Create scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot: renewable share vs price
    ax1 = axes[0]
    ax1.scatter(df["renewable_share"], df["price_daily_avg"], alpha=0.5, s=10)
    ax1.set_xlabel("Renewable Share (%)")
    ax1.set_ylabel("Daily Avg Price (EUR/MWh)")
    ax1.set_title(f"Renewable Share vs Electricity Price\n(Pearson r = {corr:.3f}, p = {p_value:.2e})")

    # Add trend line
    z = np.polyfit(df["renewable_share"], df["price_daily_avg"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["renewable_share"].min(), df["renewable_share"].max(), 100)
    ax1.plot(x_line, p(x_line), "r-", linewidth=2, label=f"Trend: y = {z[0]:.2f}x + {z[1]:.1f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot: price distribution by renewable category
    ax2 = axes[1]
    df_plot = df.copy()
    df_plot["renewable_category"] = df_plot["renewable_share"].apply(
        lambda x: f"High (>={high_renewable_threshold}%)" if x >= high_renewable_threshold else f"Low (<{high_renewable_threshold}%)"
    )
    categories = [f"Low (<{high_renewable_threshold}%)", f"High (>={high_renewable_threshold}%)"]
    data_to_plot = [
        df_plot[df_plot["renewable_category"] == cat]["price_daily_avg"].values
        for cat in categories
    ]
    bp = ax2.boxplot(data_to_plot, labels=categories, patch_artist=True)
    bp["boxes"][0].set_facecolor("lightcoral")
    bp["boxes"][1].set_facecolor("lightgreen")
    ax2.set_ylabel("Daily Avg Price (EUR/MWh)")
    ax2.set_title("Price Distribution by Renewable Share")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "exploratory_analysis.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print()
    print(f"Saved plot: {plot_path}")

    results["plot_path"] = plot_path
    return results


# ============================================================================
# Part 3: Linear Regression
# ============================================================================

def linear_regression(df: pd.DataFrame, output_dir: str = "/outputs") -> dict:
    """
    Fit linear regression: price ~ renewable_share

    Extracts coefficient to estimate price impact per percentage point change
    in renewable share.

    Args:
        df: DataFrame with renewable_share and price_daily_avg columns
        output_dir: Directory for output plots

    Returns:
        Dictionary with regression results
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {}

    print_header("Part 3: Linear Regression")

    X = df["renewable_share"].values
    y = df["price_daily_avg"].values

    # Fit linear regression using scipy
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

    results["slope"] = slope
    results["intercept"] = intercept
    results["r_squared"] = r_value ** 2
    results["r_value"] = r_value
    results["p_value"] = p_value
    results["std_err"] = std_err

    print(f"Model: price = {slope:.3f} * renewable_share + {intercept:.2f}")
    print()
    print(f"Coefficient (slope): {slope:.4f} EUR/MWh per %")
    print(f"  -> Each 1% increase in renewable share changes price by {slope:.2f} EUR/MWh")
    print(f"  -> Each 10% drop in renewable share increases price by {-slope * 10:.2f} EUR/MWh")
    print()
    print(f"Intercept: {intercept:.2f} EUR/MWh")
    print(f"R-squared: {r_value**2:.4f} ({r_value**2*100:.1f}% of variance explained)")
    print(f"Standard error: {std_err:.4f}")
    print(f"P-value: {p_value:.2e}")

    # Calculate residuals for diagnostics
    y_pred = slope * X + intercept
    residuals = y - y_pred
    results["residuals_mean"] = residuals.mean()
    results["residuals_std"] = residuals.std()

    print()
    print(f"Residuals: mean = {residuals.mean():.4f}, std = {residuals.std():.2f} EUR/MWh")

    # Create diagnostic plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Regression line with data
    ax1 = axes[0]
    ax1.scatter(X, y, alpha=0.4, s=10, label="Data")
    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = slope * x_line + intercept
    ax1.plot(x_line, y_line, "r-", linewidth=2, label=f"Fit: y = {slope:.2f}x + {intercept:.1f}")
    ax1.set_xlabel("Renewable Share (%)")
    ax1.set_ylabel("Price (EUR/MWh)")
    ax1.set_title(f"Linear Regression (R² = {r_value**2:.3f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Residuals vs fitted
    ax2 = axes[1]
    ax2.scatter(y_pred, residuals, alpha=0.4, s=10)
    ax2.axhline(y=0, color="r", linestyle="--")
    ax2.set_xlabel("Fitted Values (EUR/MWh)")
    ax2.set_ylabel("Residuals (EUR/MWh)")
    ax2.set_title("Residuals vs Fitted")
    ax2.grid(True, alpha=0.3)

    # 3. Residuals histogram
    ax3 = axes[2]
    ax3.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    ax3.axvline(x=0, color="r", linestyle="--")
    ax3.set_xlabel("Residuals (EUR/MWh)")
    ax3.set_ylabel("Frequency")
    ax3.set_title(f"Residuals Distribution (std = {residuals.std():.1f})")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "regression_diagnostics.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print()
    print(f"Saved plot: {plot_path}")

    results["plot_path"] = plot_path
    return results


# ============================================================================
# Part 4: Monte Carlo Simulation
# ============================================================================

def monte_carlo_simulation(
    df: pd.DataFrame,
    regression_slope: float,
    output_dir: str = "/outputs",
    n_simulations: int = 5000,
    high_renewable_threshold: float = 60,
    drop_mean: float = 20,
    drop_std: float = 10,
    seed: Optional[int] = 42
) -> dict:
    """
    Monte Carlo simulation of price spikes from sudden renewable drops.

    Methodology:
    1. Filter high renewable days (>threshold% share)
    2. For each simulation run:
       - Randomly sample a high renewable day
       - Simulate a random drop: Normal(mean=drop_mean, std=drop_std)
       - Calculate new price: base_price + (drop_pct * -slope)
    3. Aggregate results: mean spike, 95th percentile, probability >100 EUR/MWh

    Args:
        df: DataFrame with renewable_share and price_daily_avg columns
        regression_slope: Slope from linear regression (EUR/MWh per %)
        output_dir: Directory for output plots
        n_simulations: Number of simulation runs
        high_renewable_threshold: Minimum renewable share to consider (%)
        drop_mean: Mean of random drop distribution (%)
        drop_std: Std dev of random drop distribution (%)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with simulation results
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if seed is not None:
        np.random.seed(seed)

    results = {}

    print_header("Part 4: Monte Carlo Simulation")

    # Filter high renewable days
    high_renewable = df[df["renewable_share"] >= high_renewable_threshold].copy()

    if len(high_renewable) == 0:
        print(f"Warning: No days with renewable share >= {high_renewable_threshold}%")
        print("Cannot run simulation.")
        return {"error": "No high renewable days found"}

    print(f"Simulation Parameters:")
    print(f"  - High renewable days (>={high_renewable_threshold}%): {len(high_renewable)}")
    print(f"  - Renewable drop distribution: Normal(mean={drop_mean}%, std={drop_std}%)")
    print(f"  - Price coefficient (from regression): {regression_slope:.3f} EUR/MWh per %")
    print(f"  - Number of simulations: {n_simulations:,}")
    print()

    # Run simulations
    base_prices = high_renewable["price_daily_avg"].values
    base_renewable = high_renewable["renewable_share"].values

    # Sample random base days
    sample_indices = np.random.randint(0, len(high_renewable), size=n_simulations)
    sampled_base_prices = base_prices[sample_indices]
    sampled_base_renewable = base_renewable[sample_indices]

    # Generate random drops (clipped to avoid negative renewable share)
    random_drops = np.random.normal(drop_mean, drop_std, n_simulations)
    random_drops = np.clip(random_drops, 0, sampled_base_renewable)  # Can't drop more than current share

    # Calculate price changes
    # Negative slope means higher renewable = lower price
    # So a drop in renewable (positive drop) causes price increase
    price_changes = random_drops * (-regression_slope)
    new_prices = sampled_base_prices + price_changes

    # Calculate spikes (price increase from base)
    price_spikes = new_prices - sampled_base_prices

    # Aggregate statistics
    results["n_simulations"] = n_simulations
    results["high_renewable_threshold"] = high_renewable_threshold
    results["high_renewable_days"] = len(high_renewable)
    results["drop_mean"] = drop_mean
    results["drop_std"] = drop_std
    results["regression_slope"] = regression_slope

    results["base_price_mean"] = sampled_base_prices.mean()
    results["actual_drop_mean"] = random_drops.mean()
    results["actual_drop_std"] = random_drops.std()

    results["spike_mean"] = price_spikes.mean()
    results["spike_median"] = np.median(price_spikes)
    results["spike_std"] = price_spikes.std()
    results["spike_95th"] = np.percentile(price_spikes, 95)
    results["spike_99th"] = np.percentile(price_spikes, 99)
    results["spike_max"] = price_spikes.max()

    results["new_price_mean"] = new_prices.mean()
    results["new_price_95th"] = np.percentile(new_prices, 95)
    results["new_price_max"] = new_prices.max()

    # Probability of extreme events
    threshold_100 = 100
    threshold_150 = 150
    prob_above_100 = (new_prices > threshold_100).sum() / n_simulations * 100
    prob_above_150 = (new_prices > threshold_150).sum() / n_simulations * 100

    results["prob_price_above_100"] = prob_above_100
    results["prob_price_above_150"] = prob_above_150

    print("Simulation Results:")
    print()
    print(f"  Base price (sampled high-renewable days):")
    print(f"    Mean: {sampled_base_prices.mean():.2f} EUR/MWh")
    print()
    print(f"  Renewable drop (simulated):")
    print(f"    Mean: {random_drops.mean():.1f}%")
    print(f"    Std:  {random_drops.std():.1f}%")
    print()
    print(f"  Price spike (increase from base):")
    print(f"    Mean:   {price_spikes.mean():.2f} EUR/MWh")
    print(f"    Median: {np.median(price_spikes):.2f} EUR/MWh")
    print(f"    95th:   {np.percentile(price_spikes, 95):.2f} EUR/MWh")
    print(f"    99th:   {np.percentile(price_spikes, 99):.2f} EUR/MWh")
    print(f"    Max:    {price_spikes.max():.2f} EUR/MWh")
    print()
    print(f"  New price (after spike):")
    print(f"    Mean:   {new_prices.mean():.2f} EUR/MWh")
    print(f"    95th:   {np.percentile(new_prices, 95):.2f} EUR/MWh")
    print(f"    Max:    {new_prices.max():.2f} EUR/MWh")
    print()
    print(f"  Probability of extreme prices:")
    print(f"    P(price > {threshold_100} EUR/MWh): {prob_above_100:.2f}%")
    print(f"    P(price > {threshold_150} EUR/MWh): {prob_above_150:.2f}%")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Distribution of price spikes
    ax1 = axes[0, 0]
    ax1.hist(price_spikes, bins=50, edgecolor="black", alpha=0.7, color="coral")
    ax1.axvline(x=price_spikes.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {price_spikes.mean():.1f}")
    ax1.axvline(x=np.percentile(price_spikes, 95), color="darkred", linestyle=":", linewidth=2, label=f"95th: {np.percentile(price_spikes, 95):.1f}")
    ax1.set_xlabel("Price Spike (EUR/MWh)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Simulated Price Spikes")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Distribution of new prices
    ax2 = axes[0, 1]
    ax2.hist(new_prices, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax2.axvline(x=threshold_100, color="orange", linestyle="--", linewidth=2, label=f"Threshold: {threshold_100} EUR/MWh")
    ax2.axvline(x=new_prices.mean(), color="blue", linestyle="--", linewidth=2, label=f"Mean: {new_prices.mean():.1f}")
    ax2.set_xlabel("New Price (EUR/MWh)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Prices After Renewable Drop")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Renewable drop vs price spike scatter
    ax3 = axes[1, 0]
    ax3.scatter(random_drops, price_spikes, alpha=0.1, s=5)
    ax3.set_xlabel("Renewable Drop (%)")
    ax3.set_ylabel("Price Spike (EUR/MWh)")
    ax3.set_title("Renewable Drop vs Price Spike")
    ax3.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(random_drops, price_spikes, 1)
    x_line = np.linspace(random_drops.min(), random_drops.max(), 100)
    ax3.plot(x_line, np.poly1d(z)(x_line), "r-", linewidth=2, label=f"Slope: {z[0]:.2f}")
    ax3.legend()

    # 4. Summary text box
    ax4 = axes[1, 1]
    ax4.axis("off")

    summary_text = f"""
    Monte Carlo Simulation Summary
    ==============================

    Input Parameters:
    - Simulations: {n_simulations:,}
    - High renewable threshold: {high_renewable_threshold}%
    - Drop distribution: N({drop_mean}, {drop_std}²)
    - Regression slope: {regression_slope:.3f} EUR/MWh per %

    Key Results:
    - Mean price spike: {price_spikes.mean():.2f} EUR/MWh
    - 95th percentile spike: {np.percentile(price_spikes, 95):.2f} EUR/MWh
    - P(price > 100 EUR/MWh): {prob_above_100:.2f}%
    - P(price > 150 EUR/MWh): {prob_above_150:.2f}%

    Interpretation:
    When renewable generation drops by an average of
    {drop_mean}% on high-renewable days, prices increase
    by {price_spikes.mean():.1f} EUR/MWh on average, with
    a {prob_above_100:.1f}% chance of exceeding 100 EUR/MWh.
    """

    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "monte_carlo_simulation.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print()
    print(f"Saved plot: {plot_path}")

    results["plot_path"] = plot_path
    return results


# ============================================================================
# Full Analysis Pipeline
# ============================================================================

def run_full_analysis(
    df: pd.DataFrame,
    output_dir: str = "/outputs",
    mc_config: Optional[dict] = None
) -> dict:
    """
    Run the complete analysis pipeline (Parts 2-4).

    Args:
        df: DataFrame with renewable_share and price_daily_avg columns
        output_dir: Directory for output plots and report
        mc_config: Optional Monte Carlo configuration with keys:
                   high_renewable_threshold, drop_mean, drop_std

    Returns:
        Dictionary with all analysis results
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Default Monte Carlo config
    if mc_config is None:
        mc_config = {}
    threshold = mc_config.get("high_renewable_threshold", 60)
    drop_mean = mc_config.get("drop_mean", 20)
    drop_std = mc_config.get("drop_std", 10)

    all_results = {
        "data_summary": {
            "n_records": len(df),
            "date_start": str(df["date"].min().date()),
            "date_end": str(df["date"].max().date()),
            "renewable_share_mean": df["renewable_share"].mean(),
            "renewable_share_min": df["renewable_share"].min(),
            "renewable_share_max": df["renewable_share"].max(),
            "price_mean": df["price_daily_avg"].mean(),
            "price_min": df["price_daily_avg"].min(),
            "price_max": df["price_daily_avg"].max(),
        }
    }

    # Part 2: Exploratory Analysis
    exploratory_results = exploratory_analysis(df, output_dir)
    all_results["exploratory"] = exploratory_results

    print()

    # Part 3: Linear Regression
    regression_results = linear_regression(df, output_dir)
    all_results["regression"] = regression_results

    print()

    # Part 4: Monte Carlo Simulation
    slope = regression_results["slope"]
    mc_results = monte_carlo_simulation(
        df, slope, output_dir,
        high_renewable_threshold=threshold,
        drop_mean=drop_mean,
        drop_std=drop_std
    )
    all_results["monte_carlo"] = mc_results

    # Save JSON report
    report_path = os.path.join(output_dir, "report.json")

    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    all_results_clean = convert_types(all_results)

    with open(report_path, "w") as f:
        json.dump(all_results_clean, f, indent=2)

    print()
    print_header("Analysis Complete")
    print(f"Report saved: {report_path}")
    print()
    print("Output files:")
    print(f"  - {exploratory_results.get('plot_path', 'N/A')}")
    print(f"  - {regression_results.get('plot_path', 'N/A')}")
    print(f"  - {mc_results.get('plot_path', 'N/A')}")
    print(f"  - {report_path}")

    # Print key insight summary
    print()
    print_key_insight(regression_results, mc_results)

    return all_results


def print_key_insight(regression: dict, monte_carlo: dict) -> None:
    """Print a concise summary of key findings."""
    slope = regression.get("slope", 0)
    spike_mean = monte_carlo.get("spike_mean", 0)
    spike_95 = monte_carlo.get("spike_95th", 0)
    drop_mean = monte_carlo.get("drop_mean", 20)

    print("=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print(f"A {drop_mean:.0f}% renewable drop -> ~{spike_mean:.0f} EUR/MWh price spike")
    print(f"95th percentile spike: {spike_95:.0f} EUR/MWh")
    print(f"Coefficient: {-slope:.2f} EUR/MWh per 1% renewable drop")
    print("=" * 60)
