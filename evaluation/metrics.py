"""
Trading Metrics for XAU-SNIPER
===============================

Functions to calculate trading performance metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union


def calculate_sharpe(
    returns: Union[np.ndarray, pd.Series],
    risk_free: float = 0.0,
    periods_per_year: int = 252 * 24
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Array of periodic returns
        risk_free: Risk-free rate (annualized)
        periods_per_year: Trading periods per year (252*24 for hourly)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    returns = np.array(returns)
    excess_returns = returns - risk_free / periods_per_year

    if np.std(returns) == 0:
        return 0.0

    sharpe = np.mean(excess_returns) / np.std(returns)
    return sharpe * np.sqrt(periods_per_year)


def calculate_sortino(
    returns: Union[np.ndarray, pd.Series],
    risk_free: float = 0.0,
    periods_per_year: int = 252 * 24
) -> float:
    """
    Calculate annualized Sortino ratio.

    Uses only downside deviation (negative returns).

    Args:
        returns: Array of periodic returns
        risk_free: Risk-free rate (annualized)
        periods_per_year: Trading periods per year

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    returns = np.array(returns)
    excess_returns = returns - risk_free / periods_per_year

    # Downside deviation (only negative returns)
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0:
        return float('inf') if np.mean(excess_returns) > 0 else 0.0

    downside_std = np.std(downside_returns)
    if downside_std == 0:
        return float('inf') if np.mean(excess_returns) > 0 else 0.0

    sortino = np.mean(excess_returns) / downside_std
    return sortino * np.sqrt(periods_per_year)


def calculate_max_drawdown(equity: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate maximum drawdown.

    Args:
        equity: Equity curve

    Returns:
        Maximum drawdown as a fraction (0 to 1)
    """
    if len(equity) < 2:
        return 0.0

    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    return float(np.max(drawdown))


def calculate_calmar(
    returns: Union[np.ndarray, pd.Series],
    equity: Union[np.ndarray, pd.Series],
    periods_per_year: int = 252 * 24
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Array of periodic returns
        equity: Equity curve
        periods_per_year: Trading periods per year

    Returns:
        Calmar ratio
    """
    max_dd = calculate_max_drawdown(equity)

    if max_dd == 0:
        return float('inf') if np.mean(returns) > 0 else 0.0

    annualized_return = np.mean(returns) * periods_per_year
    return annualized_return / max_dd


def calculate_profit_factor(trade_pnls: List[float]) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        trade_pnls: List of trade PnLs

    Returns:
        Profit factor (> 1 is profitable)
    """
    if not trade_pnls:
        return 0.0

    profits = sum(p for p in trade_pnls if p > 0)
    losses = abs(sum(p for p in trade_pnls if p < 0))

    if losses == 0:
        return float('inf') if profits > 0 else 0.0

    return profits / losses


def calculate_win_rate(trade_pnls: List[float]) -> float:
    """
    Calculate win rate.

    Args:
        trade_pnls: List of trade PnLs

    Returns:
        Win rate (0 to 1)
    """
    if not trade_pnls:
        return 0.0

    wins = sum(1 for p in trade_pnls if p > 0)
    return wins / len(trade_pnls)


def calculate_expectancy(trade_pnls: List[float]) -> float:
    """
    Calculate trade expectancy (average PnL per trade).

    Args:
        trade_pnls: List of trade PnLs

    Returns:
        Expectancy
    """
    if not trade_pnls:
        return 0.0
    return np.mean(trade_pnls)


def calculate_avg_win_loss_ratio(trade_pnls: List[float]) -> float:
    """
    Calculate average win / average loss ratio.

    Args:
        trade_pnls: List of trade PnLs

    Returns:
        Win/loss ratio
    """
    wins = [p for p in trade_pnls if p > 0]
    losses = [abs(p) for p in trade_pnls if p < 0]

    if not wins or not losses:
        return 0.0

    return np.mean(wins) / np.mean(losses)


def calculate_recovery_factor(
    total_return: float,
    max_drawdown: float
) -> float:
    """
    Calculate recovery factor (total return / max drawdown).

    Args:
        total_return: Total return as fraction
        max_drawdown: Maximum drawdown as fraction

    Returns:
        Recovery factor
    """
    if max_drawdown == 0:
        return float('inf') if total_return > 0 else 0.0
    return total_return / max_drawdown


def calculate_all_metrics(
    equity: np.ndarray,
    trade_pnls: List[float],
    periods_per_year: int = 252 * 24
) -> Dict[str, float]:
    """
    Calculate all trading metrics.

    Args:
        equity: Equity curve
        trade_pnls: List of trade PnLs
        periods_per_year: Trading periods per year

    Returns:
        Dictionary of all metrics
    """
    equity = np.array(equity)
    returns = np.diff(equity) / equity[:-1]

    total_return = (equity[-1] / equity[0]) - 1 if len(equity) > 0 else 0
    max_dd = calculate_max_drawdown(equity)

    metrics = {
        # Return metrics
        'total_return': total_return,
        'annualized_return': np.mean(returns) * periods_per_year if len(returns) > 0 else 0,

        # Risk metrics
        'max_drawdown': max_dd,
        'volatility': np.std(returns) * np.sqrt(periods_per_year) if len(returns) > 0 else 0,

        # Risk-adjusted metrics
        'sharpe_ratio': calculate_sharpe(returns, periods_per_year=periods_per_year),
        'sortino_ratio': calculate_sortino(returns, periods_per_year=periods_per_year),
        'calmar_ratio': calculate_calmar(returns, equity, periods_per_year),
        'recovery_factor': calculate_recovery_factor(total_return, max_dd),

        # Trade metrics
        'total_trades': len(trade_pnls),
        'win_rate': calculate_win_rate(trade_pnls),
        'profit_factor': calculate_profit_factor(trade_pnls),
        'expectancy': calculate_expectancy(trade_pnls),
        'avg_win_loss_ratio': calculate_avg_win_loss_ratio(trade_pnls),

        # Trade statistics
        'avg_trade_pnl': np.mean(trade_pnls) if trade_pnls else 0,
        'max_trade_pnl': max(trade_pnls) if trade_pnls else 0,
        'min_trade_pnl': min(trade_pnls) if trade_pnls else 0,
    }

    return metrics


def print_metrics(metrics: Dict[str, float]):
    """Pretty print trading metrics."""
    print("\n" + "=" * 50)
    print("TRADING PERFORMANCE METRICS")
    print("=" * 50)

    print("\n--- Return Metrics ---")
    print(f"Total Return:      {metrics['total_return']*100:>8.2f}%")
    print(f"Annualized Return: {metrics['annualized_return']*100:>8.2f}%")

    print("\n--- Risk Metrics ---")
    print(f"Max Drawdown:      {metrics['max_drawdown']*100:>8.2f}%")
    print(f"Volatility:        {metrics['volatility']*100:>8.2f}%")

    print("\n--- Risk-Adjusted Metrics ---")
    print(f"Sharpe Ratio:      {metrics['sharpe_ratio']:>8.3f}")
    print(f"Sortino Ratio:     {metrics['sortino_ratio']:>8.3f}")
    print(f"Calmar Ratio:      {metrics['calmar_ratio']:>8.3f}")
    print(f"Recovery Factor:   {metrics['recovery_factor']:>8.3f}")

    print("\n--- Trade Metrics ---")
    print(f"Total Trades:      {metrics['total_trades']:>8d}")
    print(f"Win Rate:          {metrics['win_rate']*100:>8.1f}%")
    print(f"Profit Factor:     {metrics['profit_factor']:>8.2f}")
    print(f"Expectancy:        {metrics['expectancy']:>8.2f}")
    print(f"Avg Win/Loss:      {metrics['avg_win_loss_ratio']:>8.2f}")

    print("\n--- Trade Statistics ---")
    print(f"Avg Trade PnL:     {metrics['avg_trade_pnl']:>8.2f}")
    print(f"Max Trade PnL:     {metrics['max_trade_pnl']:>8.2f}")
    print(f"Min Trade PnL:     {metrics['min_trade_pnl']:>8.2f}")

    print("=" * 50)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing metrics calculations...")

    # Generate sample data
    np.random.seed(42)
    n_periods = 1000

    # Simulated equity curve with positive drift
    returns = np.random.normal(0.0001, 0.001, n_periods)
    equity = 10000 * np.cumprod(1 + returns)

    # Simulated trades
    trade_pnls = list(np.random.normal(5, 20, 50))  # 50 trades

    # Calculate all metrics
    metrics = calculate_all_metrics(equity, trade_pnls)

    # Print results
    print_metrics(metrics)
