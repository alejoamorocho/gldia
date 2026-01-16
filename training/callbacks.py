"""
Training Callbacks for XAU-SNIPER
==================================

Custom callbacks for monitoring and controlling training.
"""

import os
import numpy as np
from typing import Dict, Any, Optional
import logging

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

logger = logging.getLogger(__name__)


class SniperCallback(BaseCallback):
    """
    Custom callback for Sniper trading strategy.

    Monitors:
    - Trade frequency (should be ~1 per week)
    - Win rate and profit factor
    - Sharpe/Sortino ratios
    - Stops training if no improvement
    """

    def __init__(
        self,
        check_freq: int = 2048,
        min_trades_per_check: int = 5,
        patience: int = 50,
        min_sharpe: float = 0.5,
        verbose: int = 1
    ):
        """
        Initialize callback.

        Args:
            check_freq: Steps between checks
            min_trades_per_check: Minimum trades expected per check
            patience: Steps without improvement before early stop
            min_sharpe: Minimum Sharpe ratio to consider improvement
            verbose: Verbosity level
        """
        super().__init__(verbose)

        self.check_freq = check_freq
        self.min_trades_per_check = min_trades_per_check
        self.patience = patience
        self.min_sharpe = min_sharpe

        self.best_sharpe = -np.inf
        self.no_improvement_count = 0
        self.trade_counts = []
        self.sharpe_history = []

    def _on_step(self) -> bool:
        """Called at each step."""
        if self.n_calls % self.check_freq == 0:
            return self._check_progress()
        return True

    def _check_progress(self) -> bool:
        """Check training progress and metrics."""
        # Get environment info
        env = self.training_env.envs[0]

        # Try to get episode stats
        try:
            stats = env.get_episode_stats()
            trades = stats.get('total_trades', 0)
            sharpe = stats.get('sharpe_ratio', 0)
            win_rate = stats.get('win_rate', 0)
            profit_factor = stats.get('profit_factor', 0)
        except Exception:
            # Environment doesn't have stats method
            trades = 0
            sharpe = 0
            win_rate = 0
            profit_factor = 0

        self.trade_counts.append(trades)
        self.sharpe_history.append(sharpe)

        if self.verbose > 0:
            logger.info(
                f"Step {self.n_calls}: "
                f"Trades={trades}, Sharpe={sharpe:.3f}, "
                f"WinRate={win_rate:.2%}, PF={profit_factor:.2f}"
            )

        # Check for improvement
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            self.no_improvement_count = 0

            if self.verbose > 0:
                logger.info(f"New best Sharpe: {sharpe:.3f}")
        else:
            self.no_improvement_count += 1

        # Early stopping
        if self.no_improvement_count >= self.patience:
            if self.verbose > 0:
                logger.warning(
                    f"Early stopping: No improvement for {self.patience} checks"
                )
            return False

        # Check trade frequency warning
        if trades < self.min_trades_per_check:
            if self.verbose > 0:
                logger.warning(
                    f"Low trade frequency: {trades} trades "
                    f"(expected >= {self.min_trades_per_check})"
                )

        return True

    def _on_training_end(self) -> None:
        """Called at end of training."""
        if self.verbose > 0:
            logger.info(f"Training finished. Best Sharpe: {self.best_sharpe:.3f}")
            logger.info(f"Total checks: {len(self.sharpe_history)}")


class MetricsCallback(BaseCallback):
    """
    Callback for logging detailed trading metrics to TensorBoard.
    """

    def __init__(self, log_freq: int = 1024, verbose: int = 0):
        """
        Initialize callback.

        Args:
            log_freq: Steps between metric logging
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_freq = log_freq

        # Metric accumulators
        self.episode_rewards = []
        self.episode_lengths = []
        self.trade_pnls = []

    def _on_step(self) -> bool:
        """Called at each step."""
        # Collect episode info from monitor wrapper
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])

            # Collect trade PnL if available
            if 'trade_stats' in info:
                stats = info['trade_stats']
                if stats.get('total_trades', 0) > 0:
                    self.trade_pnls.append(stats.get('avg_pnl', 0))

        # Log metrics
        if self.n_calls % self.log_freq == 0:
            self._log_metrics()

        return True

    def _log_metrics(self):
        """Log accumulated metrics to TensorBoard."""
        if len(self.episode_rewards) > 0:
            self.logger.record('rollout/ep_rew_mean', np.mean(self.episode_rewards[-100:]))
            self.logger.record('rollout/ep_len_mean', np.mean(self.episode_lengths[-100:]))

        if len(self.trade_pnls) > 0:
            self.logger.record('trading/avg_trade_pnl', np.mean(self.trade_pnls[-100:]))
            self.logger.record('trading/total_trades', len(self.trade_pnls))

            # Win rate
            wins = sum(1 for p in self.trade_pnls[-100:] if p > 0)
            self.logger.record('trading/win_rate', wins / len(self.trade_pnls[-100:]))


class SaveBestCallback(BaseCallback):
    """
    Callback to save model when new best is found.
    """

    def __init__(
        self,
        save_path: str,
        check_freq: int = 2048,
        metric: str = 'sharpe',
        verbose: int = 1
    ):
        """
        Initialize callback.

        Args:
            save_path: Path to save best model
            check_freq: Steps between checks
            metric: Metric to optimize ('sharpe', 'return', 'profit_factor')
            verbose: Verbosity level
        """
        super().__init__(verbose)

        self.save_path = save_path
        self.check_freq = check_freq
        self.metric = metric
        self.best_value = -np.inf

    def _on_step(self) -> bool:
        """Called at each step."""
        if self.n_calls % self.check_freq == 0:
            self._check_and_save()
        return True

    def _check_and_save(self):
        """Check metric and save if improved."""
        # Fix: Unwrap to find GoldEnv, whether it is DummyVecEnv, Monitor, etc
        env = self.training_env.envs[0]
        while hasattr(env, 'env'):
            env = env.env
        # Also check for 'unwrapped' attribute if it's a specific wrapper
        if hasattr(env, 'unwrapped'):
            env = env.unwrapped

        try:
            if hasattr(env, 'get_episode_stats'):
                stats = env.get_episode_stats()

                if self.metric == 'sharpe':
                    current_value = stats.get('sharpe_ratio', -np.inf)
                elif self.metric == 'return':
                    current_value = stats.get('total_return', -np.inf)
                elif self.metric == 'profit_factor':
                    current_value = stats.get('profit_factor', -np.inf)
                else:
                    current_value = -np.inf

                if current_value > self.best_value:
                    self.best_value = current_value
                    self.model.save(self.save_path)
                    if self.verbose > 0:
                        logger.info(f"New best {self.metric}: {current_value:.4f}. Model saved.")
            else:
                 pass # Cannot get stats

        except Exception as e:
            if self.verbose > 0:
                logger.warning(f"Could not check metric: {e}")


def create_eval_callback(
    eval_env,
    log_path: str,
    eval_freq: int = 10000,
    n_eval_episodes: int = 5,
    best_model_save_path: Optional[str] = None
) -> EvalCallback:
    """
    Create evaluation callback.

    Args:
        eval_env: Evaluation environment
        log_path: Path for logs
        eval_freq: Evaluation frequency
        n_eval_episodes: Episodes per evaluation
        best_model_save_path: Path to save best model

    Returns:
        EvalCallback instance
    """
    return EvalCallback(
        eval_env=eval_env,
        log_path=log_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        best_model_save_path=best_model_save_path,
        deterministic=True,
        render=False,
        verbose=1
    )
