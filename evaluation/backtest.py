"""
Backtesting Module for XAU-SNIPER
==================================

Run backtests on historical data with trained models.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.gold_env import GoldEnv
from env.risk_manager import RiskConfig
from evaluation.metrics import calculate_all_metrics, print_metrics

logger = logging.getLogger(__name__)


class BacktestResult:
    """Container for backtest results."""

    def __init__(self):
        self.equity_curve: List[float] = []
        self.trades: List[Dict] = []
        self.actions: List[int] = []
        self.timestamps: List[datetime] = []
        self.metrics: Dict[str, float] = {}

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        df = pd.DataFrame({
            'timestamp': self.timestamps[:len(self.equity_curve)],
            'equity': self.equity_curve,
            'action': self.actions[:len(self.equity_curve)]
        })
        return df


def run_backtest(
    model,
    df_h1: pd.DataFrame,
    df_m1: Optional[pd.DataFrame] = None,
    feature_columns: Optional[List[str]] = None,
    risk_config: Optional[RiskConfig] = None,
    initial_balance: float = 10000.0,
    start_idx: int = 200,
    end_idx: Optional[int] = None,
    deterministic: bool = True,
    verbose: bool = True
) -> BacktestResult:
    """
    Run backtest with trained model.

    Args:
        model: Trained PPO-LSTM model
        df_h1: Hourly data with features
        df_m1: Minute data for execution (optional)
        feature_columns: Feature columns for observation
        risk_config: Risk management configuration
        initial_balance: Starting capital
        start_idx: Start index in data
        end_idx: End index in data (None = end of data)
        deterministic: Use deterministic predictions
        verbose: Print progress

    Returns:
        BacktestResult with equity curve, trades, and metrics
    """
    if verbose:
        logger.info("Starting backtest...")

    # Create environment
    env = GoldEnv(
        df_h1=df_h1,
        df_m1=df_m1,
        feature_columns=feature_columns,
        initial_balance=initial_balance,
        risk_config=risk_config or RiskConfig(),
        episode_length=None,  # Full data
        randomize_start=False
    )

    # Initialize result
    result = BacktestResult()

    # Reset environment
    obs, info = env.reset()

    # Override start position
    env.current_step = start_idx
    env.start_step = start_idx
    obs = env._get_observation()

    # Determine end index
    if end_idx is None:
        end_idx = len(env.df_h1) - 1

    # LSTM states
    lstm_states = None
    episode_start = True

    # Run backtest
    total_steps = end_idx - start_idx
    step = 0

    while env.current_step < end_idx:
        # Get action from model
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=deterministic
        )

        # Execute step
        obs, reward, terminated, truncated, info = env.step(action)
        episode_start = False

        # Record results
        result.equity_curve.append(info['equity'])
        result.actions.append(int(action))

        # Get timestamp
        if env.current_step < len(env.df_h1):
            timestamp = env.df_h1.index[env.current_step]
            result.timestamps.append(timestamp)

        step += 1

        # Progress update
        if verbose and step % 500 == 0:
            progress = step / total_steps * 100
            logger.info(f"Progress: {progress:.1f}% | "
                       f"Equity: ${info['equity']:.2f} | "
                       f"Trades: {info['trade_count']}")

        if terminated or truncated:
            break

    # Get final trades
    result.trades = env.risk_manager.trade_history.copy()

    # Calculate metrics
    trade_pnls = [t['pnl'] for t in result.trades]
    result.metrics = calculate_all_metrics(
        np.array(result.equity_curve),
        trade_pnls,
        periods_per_year=252 * 24
    )

    if verbose:
        logger.info("Backtest complete!")
        print_metrics(result.metrics)

    return result


def run_walk_forward_backtest(
    model,
    df_h1: pd.DataFrame,
    df_m1: Optional[pd.DataFrame] = None,
    feature_columns: Optional[List[str]] = None,
    risk_config: Optional[RiskConfig] = None,
    n_splits: int = 5,
    train_ratio: float = 0.7,
    verbose: bool = True
) -> List[BacktestResult]:
    """
    Run walk-forward backtest with multiple periods.

    Args:
        model: Trained model (or model factory)
        df_h1: Full historical data
        df_m1: Minute data
        feature_columns: Feature columns
        risk_config: Risk config
        n_splits: Number of train/test splits
        train_ratio: Ratio for training in each split
        verbose: Print progress

    Returns:
        List of BacktestResult for each split
    """
    results = []
    n_samples = len(df_h1)
    split_size = n_samples // n_splits

    for i in range(n_splits):
        if verbose:
            logger.info(f"\n{'='*50}")
            logger.info(f"Walk-Forward Split {i+1}/{n_splits}")
            logger.info(f"{'='*50}")

        # Calculate split indices
        test_start = i * split_size
        test_end = (i + 1) * split_size if i < n_splits - 1 else n_samples

        # Skip warmup period
        if test_start < 200:
            test_start = 200

        # Run backtest on this split
        result = run_backtest(
            model=model,
            df_h1=df_h1,
            df_m1=df_m1,
            feature_columns=feature_columns,
            risk_config=risk_config,
            start_idx=test_start,
            end_idx=test_end,
            verbose=verbose
        )

        results.append(result)

    # Summary statistics
    if verbose:
        logger.info("\n" + "=" * 50)
        logger.info("WALK-FORWARD SUMMARY")
        logger.info("=" * 50)

        sharpes = [r.metrics['sharpe_ratio'] for r in results]
        returns = [r.metrics['total_return'] for r in results]
        drawdowns = [r.metrics['max_drawdown'] for r in results]

        logger.info(f"Avg Sharpe:   {np.mean(sharpes):.3f} +/- {np.std(sharpes):.3f}")
        logger.info(f"Avg Return:   {np.mean(returns)*100:.2f}% +/- {np.std(returns)*100:.2f}%")
        logger.info(f"Avg MaxDD:    {np.mean(drawdowns)*100:.2f}% +/- {np.std(drawdowns)*100:.2f}%")

    return results


def compare_to_buy_and_hold(
    result: BacktestResult,
    df_h1: pd.DataFrame
) -> Dict[str, float]:
    """
    Compare strategy performance to buy-and-hold.

    Args:
        result: Backtest result
        df_h1: Price data

    Returns:
        Dictionary with comparison metrics
    """
    # Get price series for the backtest period
    if 'close' in df_h1.columns:
        prices = df_h1['close'].values
    else:
        prices = df_h1.iloc[:, 3].values  # Assume OHLC order

    # Align with backtest period
    n_steps = len(result.equity_curve)
    start_price = prices[0]
    end_price = prices[min(n_steps, len(prices)-1)]

    # Buy and hold return
    bh_return = (end_price / start_price) - 1

    # Strategy return
    strategy_return = result.metrics['total_return']

    # Alpha (excess return)
    alpha = strategy_return - bh_return

    return {
        'strategy_return': strategy_return,
        'buy_hold_return': bh_return,
        'alpha': alpha,
        'strategy_sharpe': result.metrics['sharpe_ratio'],
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Backtest module ready for use")
    print("Use run_backtest() with a trained model")
