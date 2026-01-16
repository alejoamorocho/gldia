"""
Training Script for XAU-SNIPER PPO-LSTM
========================================

Main training script with all configurations.
"""

import os
import sys
import argparse
from datetime import datetime
import logging
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.gold_env import GoldEnv
from env.risk_manager import RiskConfig
from models.ppo_lstm import create_ppo_lstm_model, PPOLSTMConfig
from training.callbacks import SniperCallback, MetricsCallback, SaveBestCallback
from features.h1_features import add_features_h1, get_feature_columns, H1FeatureConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(data_dir: str, train_ratio: float = 0.8):
    """
    Load and prepare data for training.

    Args:
        data_dir: Directory containing data files
        train_ratio: Ratio of data for training

    Returns:
        Tuple of (train_h1, train_m1, eval_h1, eval_m1)
    """
    logger.info("Loading data...")

    # Load H1 data
    h1_path = os.path.join(data_dir, 'xauusd_h1.csv')
    df_h1 = pd.read_csv(h1_path)
    logger.info(f"Loaded H1 data: {len(df_h1)} rows")

    # Load M1 data (optional but recommended)
    m1_path = os.path.join(data_dir, 'xauusd_m1.csv')
    if os.path.exists(m1_path):
        df_m1 = pd.read_csv(m1_path)
        logger.info(f"Loaded M1 data: {len(df_m1)} rows")
    else:
        df_m1 = None
        logger.warning("M1 data not found - using H1 only for execution")

    # Load DXY for macro features (optional)
    dxy_path = os.path.join(data_dir, 'dxy_daily.csv')
    if os.path.exists(dxy_path):
        dxy_df = pd.read_csv(dxy_path)
        logger.info(f"Loaded DXY data: {len(dxy_df)} rows")
    else:
        dxy_df = None

    # Load US10Y
    us10y_path = os.path.join(data_dir, 'us10y_daily.csv')
    if os.path.exists(us10y_path):
        us10y_df = pd.read_csv(us10y_path)
        logger.info(f"Loaded US10Y data: {len(us10y_df)} rows")
    else:
        us10y_df = None

    # Load VIX
    vix_path = os.path.join(data_dir, 'vix_daily.csv')
    if os.path.exists(vix_path):
        vix_df = pd.read_csv(vix_path)
        logger.info(f"Loaded VIX data: {len(vix_df)} rows")
    else:
        vix_df = None

    # Add features to H1
    logger.info("Computing features...")
    feature_config = H1FeatureConfig()
    # Add features to H1
    logger.info("Computing features...")
    feature_config = H1FeatureConfig()
    df_h1 = add_features_h1(
        df_h1, 
        feature_config, 
        dxy_df=dxy_df,
        us10y_df=us10y_df,
        vix_df=vix_df
    )

    # Split into train/eval
    split_idx = int(len(df_h1) * train_ratio)

    train_h1 = df_h1.iloc[:split_idx].copy()
    eval_h1 = df_h1.iloc[split_idx:].copy()

    logger.info(f"Train H1: {len(train_h1)} rows")
    logger.info(f"Eval H1: {len(eval_h1)} rows")

    # Split M1 data
    if df_m1 is not None:
        # Force timezone naive for consistency
        df_m1['time'] = pd.to_datetime(df_m1['time'], utc=True).dt.tz_localize(None)
        train_h1_time = pd.to_datetime(train_h1.iloc[0]['time'] if 'time' in train_h1.columns else train_h1.index[0])
        eval_h1_time = pd.to_datetime(eval_h1.iloc[0]['time'] if 'time' in eval_h1.columns else eval_h1.index[0])

        # Get split time from H1
        h1_times = pd.to_datetime(df_h1['time'] if 'time' in df_h1.columns else df_h1.index)
        split_time = h1_times[split_idx]

        train_m1 = df_m1[df_m1['time'] < split_time].copy()
        eval_m1 = df_m1[df_m1['time'] >= split_time].copy()

        logger.info(f"Train M1: {len(train_m1)} rows")
        logger.info(f"Eval M1: {len(eval_m1)} rows")
    else:
        train_m1 = None
        eval_m1 = None

    return train_h1, train_m1, eval_h1, eval_m1


def create_environments(
    train_h1: pd.DataFrame,
    train_m1: pd.DataFrame,
    eval_h1: pd.DataFrame,
    eval_m1: pd.DataFrame,
    feature_columns: list,
    risk_config: RiskConfig
):
    """Create training and evaluation environments."""
    logger.info("Creating environments...")

    train_env = GoldEnv(
        df_h1=train_h1,
        df_m1=train_m1,
        feature_columns=feature_columns,
        risk_config=risk_config,
        episode_length=2048,
        randomize_start=True
    )

    eval_env = GoldEnv(
        df_h1=eval_h1,
        df_m1=eval_m1,
        feature_columns=feature_columns,
        risk_config=risk_config,
        episode_length=2048,
        randomize_start=True  # Randomize to test robustness (was False)
    )

    logger.info(f"Observation space: {train_env.observation_space}")
    logger.info(f"Action space: {train_env.action_space}")

    return train_env, eval_env


def train_agent(
    data_dir: str = "data",
    output_dir: str = "outputs",
    total_timesteps: int = 2_000_000,
    eval_freq: int = 10_000,
    save_freq: int = 50_000,
    seed: int = 42,
    resume_path: str = None
):
    """
    Main training function.

    Args:
        data_dir: Directory containing data files
        output_dir: Directory for outputs (models, logs)
        total_timesteps: Total training steps
        eval_freq: Evaluation frequency
        save_freq: Model save frequency
        seed: Random seed
        resume_path: Path to resume training from
    """
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    logger.info(f"Output directory: {run_dir}")

    # Load data
    train_h1, train_m1, eval_h1, eval_m1 = load_and_prepare_data(data_dir)

    # Get feature columns
    feature_columns = get_feature_columns()

    # Risk configuration - 1:1 R:R with trailing for extended gains
    risk_config = RiskConfig(
        tp_target=0.003,      # 0.3% target (triggers trailing)
        sl_initial=0.003,     # 0.3% stop (1:1 ratio)
        trailing_distance=0.0015,  # 0.15% trailing distance
        breakeven_level=0.0015,    # 0.15% breakeven buffer after TP
        commission=0.00005,   # 0.5 pips
        slippage=0.0002,      # 2 pips
        use_atr_stops=False   # Use fixed % stops (more consistent)
    )

    # Create environments
    train_env, eval_env = create_environments(
        train_h1, train_m1, eval_h1, eval_m1,
        feature_columns, risk_config
    )

    # Model configuration
    model_config = PPOLSTMConfig(
        learning_rate=3e-5,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.95,
        ent_coef=0.02,  # Increased entropy to force exploration (was 0.005)
        lstm_hidden_size=256,
        seed=seed,
        tensorboard_log=None # os.path.join(run_dir, "logs")
    )

    # Create or load model
    if resume_path and os.path.exists(resume_path):
        model = create_ppo_lstm_model(train_env, model_config, load_path=resume_path)
        logger.info(f"Resumed training from {resume_path}")
    else:
        model = create_ppo_lstm_model(train_env, model_config)
        logger.info("Created new model")

    # Create callbacks
    callbacks = [
        MetricsCallback(log_freq=1024),
        SaveBestCallback(
            save_path=os.path.join(run_dir, "models", "best_model"),
            check_freq=eval_freq,
            metric='sharpe',
            verbose=1
        )
    ]

    # Train
    logger.info(f"Starting training for {total_timesteps:,} timesteps...")
    logger.info(f"Config: {model_config}")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=resume_path is None
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise

    # Save final model
    final_path = os.path.join(run_dir, "models", "final_model")
    model.save(final_path)
    logger.info(f"Saved final model to {final_path}")

    # Final evaluation (Test Set - 2023-2025)
    # Use 1 episode covering the FULL eval period to avoid duplicates
    logger.info("Running TEST SET evaluation (2023-2025)...")

    # Create test env that covers full eval period (no randomization, no overlap)
    test_env = GoldEnv(
        df_h1=eval_h1,
        df_m1=eval_m1,
        feature_columns=feature_columns,
        risk_config=risk_config,
        episode_length=len(eval_h1),  # Cover FULL eval period
        randomize_start=False  # Start from beginning
    )

    _, trades_test = evaluate_model(model, test_env, n_episodes=1, run_dir=run_dir)
    analyze_and_report(trades_test, "TEST_SET", run_dir)

    # Full Backtest (2015-2025)
    logger.info("Running FULL BACKTEST (2015-2025)...")
    
    # Reconstruct full dataset
    full_h1 = pd.concat([train_h1, eval_h1])
    full_m1 = pd.concat([train_m1, eval_m1]) if train_m1 is not None else None
    
    # Create Full Env
    full_env = GoldEnv(
        df_h1=full_h1,
        df_m1=full_m1,
        feature_columns=feature_columns,
        risk_config=risk_config,
        episode_length=len(full_h1), # Play the WHOLE history in 1 episode
        randomize_start=False # Start from beginning 
    )
    
    # Run 1 massive episode
    _, trades_full = evaluate_model(model, full_env, n_episodes=1, run_dir=run_dir)
    
    # Rename output file to avoid overwrite (evaluate_model saves 'evaluation_trades.csv')
    if os.path.exists(os.path.join(run_dir, "evaluation_trades.csv")):
        # The evaluate_model function writes to 'evaluation_trades.csv' each time.
        # We just ran it for full, so 'evaluation_trades.csv' is now FULL.
        # We should rename it.
        os.rename(
            os.path.join(run_dir, "evaluation_trades.csv"),
            os.path.join(run_dir, "evaluation_trades_FULL.csv")
        )
        # Restore the TEST set file if we want, but 'trades_test' dataframe holds it.
        # Let's save standard test trades back to file just in case
        trades_test.to_csv(os.path.join(run_dir, "evaluation_trades.csv"), index=False)
        
    analyze_and_report(trades_full, "FULL_DATASET", run_dir)
    
    return model, run_dir


def analyze_and_report(trades_df: pd.DataFrame, report_name: str, run_dir: str):
    """
    Generate detailed Annual report from trades DataFrame.
    EXHAUSTIVE VERSION - All metrics calculated from single source of truth.
    """
    if trades_df is None or trades_df.empty:
        logger.warning(f"No trades to analyze for {report_name}")
        return

    logger.info(f"\n{'='*80}")
    logger.info(f"COMPREHENSIVE REPORT: {report_name}")
    logger.info(f"{'='*80}")

    # =========================================================================
    # 1. PARSE DATES - Handle all formats
    # =========================================================================
    trades_df = trades_df.copy()

    # Try multiple parsing strategies
    if 'exit_dt' not in trades_df.columns:
        for date_col in ['exit_date', 'entry_date']:
            if date_col in trades_df.columns:
                try:
                    trades_df['exit_dt'] = pd.to_datetime(trades_df[date_col], utc=True, errors='coerce')
                    if trades_df['exit_dt'].notna().sum() > 0:
                        break
                except:
                    try:
                        trades_df['exit_dt'] = pd.to_datetime(trades_df[date_col], errors='coerce')
                        if trades_df['exit_dt'].notna().sum() > 0:
                            break
                    except:
                        continue

    # Filter valid trades
    if 'exit_dt' not in trades_df.columns or trades_df['exit_dt'].isna().all():
        logger.error("Could not parse any dates from trades")
        return

    valid_trades = trades_df[trades_df['exit_dt'].notna()].copy()
    invalid_count = len(trades_df) - len(valid_trades)

    if invalid_count > 0:
        logger.warning(f"  {invalid_count} trades had invalid dates and were excluded")

    if len(valid_trades) == 0:
        logger.warning("No valid trades found")
        return

    # Add year/month columns
    valid_trades['year'] = valid_trades['exit_dt'].dt.year.astype(int)
    valid_trades['month'] = valid_trades['exit_dt'].dt.month.astype(int)

    # =========================================================================
    # 2. CALCULATE GLOBAL METRICS (from trades_df - single source of truth)
    # =========================================================================
    total_trades = len(valid_trades)
    wins = valid_trades[valid_trades['pnl'] > 0]
    losses = valid_trades[valid_trades['pnl'] <= 0]

    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
    total_pnl = valid_trades['pnl'].sum()
    avg_pnl = valid_trades['pnl'].mean() if total_trades > 0 else 0.0

    gross_wins = wins['pnl'].sum() if len(wins) > 0 else 0.0
    gross_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else (999.99 if gross_wins > 0 else 0.0)

    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0.0
    avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0.0
    max_win = valid_trades['pnl'].max() if total_trades > 0 else 0.0
    max_loss = valid_trades['pnl'].min() if total_trades > 0 else 0.0

    # Calculate date range and trades per year
    min_date = valid_trades['exit_dt'].min()
    max_date = valid_trades['exit_dt'].max()
    date_range_years = (max_date - min_date).days / 365.25
    trades_per_year = total_trades / date_range_years if date_range_years > 0 else total_trades

    # Exit reasons breakdown
    exit_reasons = valid_trades['exit_reason'].value_counts() if 'exit_reason' in valid_trades.columns else pd.Series()

    # =========================================================================
    # 3. PRINT GLOBAL SUMMARY
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"SUMMARY: {report_name}")
    print(f"{'='*80}")
    print(f"Date Range:       {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({date_range_years:.1f} years)")
    print(f"Total Trades:     {total_trades}")
    print(f"Trades/Year:      {trades_per_year:.1f}")
    print(f"Win Rate:         {win_rate:.1f}%")
    print(f"Profit Factor:    {profit_factor:.2f}")
    print(f"Total PnL:        ${total_pnl:.2f}")
    print(f"Avg PnL/Trade:    ${avg_pnl:.2f}")
    print(f"Avg Win:          ${avg_win:.2f}")
    print(f"Avg Loss:         ${avg_loss:.2f}")
    print(f"Largest Win:      ${max_win:.2f}")
    print(f"Largest Loss:     ${max_loss:.2f}")

    if len(exit_reasons) > 0:
        print(f"\nExit Reasons:")
        for reason, count in exit_reasons.items():
            pct = count / total_trades * 100
            print(f"  {reason:<12}: {count:>5} ({pct:.1f}%)")

    # =========================================================================
    # 4. ANNUAL PERFORMANCE TABLE
    # =========================================================================
    years = sorted(valid_trades['year'].unique())

    print(f"\n{'='*80}")
    print("ANNUAL PERFORMANCE:")
    print(f"{'='*80}")
    print(f"{'Year':<6} | {'Trades':<7} | {'Win%':<6} | {'PF':<8} | {'Total PnL':<12} | {'Avg PnL':<10} | {'MaxWin':<10} | {'MaxLoss':<10}")
    print("-" * 90)

    annual_data = []
    for y in years:
        y_trades = valid_trades[valid_trades['year'] == y]
        y_wins = y_trades[y_trades['pnl'] > 0]
        y_losses = y_trades[y_trades['pnl'] <= 0]

        y_count = len(y_trades)
        y_win_rate = (len(y_wins) / y_count * 100) if y_count > 0 else 0.0
        y_total_pnl = y_trades['pnl'].sum()
        y_avg_pnl = y_trades['pnl'].mean() if y_count > 0 else 0.0

        y_gross_win = y_wins['pnl'].sum() if len(y_wins) > 0 else 0.0
        y_gross_loss = abs(y_losses['pnl'].sum()) if len(y_losses) > 0 else 0.0
        y_pf = y_gross_win / y_gross_loss if y_gross_loss > 0 else (999.99 if y_gross_win > 0 else 0.0)

        y_max_win = y_trades['pnl'].max() if y_count > 0 else 0.0
        y_max_loss = y_trades['pnl'].min() if y_count > 0 else 0.0

        # Handle nan
        y_avg_pnl = 0.0 if pd.isna(y_avg_pnl) else y_avg_pnl
        y_total_pnl = 0.0 if pd.isna(y_total_pnl) else y_total_pnl

        print(f"{y:<6} | {y_count:<7} | {y_win_rate:<6.1f} | {y_pf:<8.2f} | ${y_total_pnl:<11.2f} | ${y_avg_pnl:<9.2f} | ${y_max_win:<9.2f} | ${y_max_loss:<9.2f}")

        annual_data.append({
            'year': y, 'trades': y_count, 'win_rate': y_win_rate,
            'profit_factor': y_pf, 'total_pnl': y_total_pnl, 'avg_pnl': y_avg_pnl,
            'max_win': y_max_win, 'max_loss': y_max_loss
        })

    # TOTAL ROW
    print("-" * 90)
    print(f"{'TOTAL':<6} | {total_trades:<7} | {win_rate:<6.1f} | {profit_factor:<8.2f} | ${total_pnl:<11.2f} | ${avg_pnl:<9.2f} | ${max_win:<9.2f} | ${max_loss:<9.2f}")
    print(f"{'='*80}")

    # =========================================================================
    # 5. SAVE TO FILE
    # =========================================================================
    report_path = os.path.join(run_dir, f"report_{report_name}.txt")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(f"COMPREHENSIVE REPORT: {report_name}\n")
        f.write(f"{'='*80}\n\n")

        f.write(f"SUMMARY\n")
        f.write(f"{'-'*40}\n")
        f.write(f"Date Range:       {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({date_range_years:.1f} years)\n")
        f.write(f"Total Trades:     {total_trades}\n")
        f.write(f"Trades/Year:      {trades_per_year:.1f}\n")
        f.write(f"Win Rate:         {win_rate:.1f}%\n")
        f.write(f"Profit Factor:    {profit_factor:.2f}\n")
        f.write(f"Total PnL:        ${total_pnl:.2f}\n")
        f.write(f"Avg PnL/Trade:    ${avg_pnl:.2f}\n")
        f.write(f"Avg Win:          ${avg_win:.2f}\n")
        f.write(f"Avg Loss:         ${avg_loss:.2f}\n")
        f.write(f"Largest Win:      ${max_win:.2f}\n")
        f.write(f"Largest Loss:     ${max_loss:.2f}\n\n")

        if len(exit_reasons) > 0:
            f.write(f"Exit Reasons:\n")
            for reason, count in exit_reasons.items():
                pct = count / total_trades * 100
                f.write(f"  {reason:<12}: {count:>5} ({pct:.1f}%)\n")
            f.write("\n")

        f.write(f"ANNUAL PERFORMANCE\n")
        f.write(f"{'-'*90}\n")
        f.write(f"{'Year':<6} | {'Trades':<7} | {'Win%':<6} | {'PF':<8} | {'Total PnL':<12} | {'Avg PnL':<10} | {'MaxWin':<10} | {'MaxLoss':<10}\n")
        f.write(f"{'-'*90}\n")

        for row in annual_data:
            f.write(f"{row['year']:<6} | {row['trades']:<7} | {row['win_rate']:<6.1f} | {row['profit_factor']:<8.2f} | ${row['total_pnl']:<11.2f} | ${row['avg_pnl']:<9.2f} | ${row['max_win']:<9.2f} | ${row['max_loss']:<9.2f}\n")

        f.write(f"{'-'*90}\n")
        f.write(f"{'TOTAL':<6} | {total_trades:<7} | {win_rate:<6.1f} | {profit_factor:<8.2f} | ${total_pnl:<11.2f} | ${avg_pnl:<9.2f} | ${max_win:<9.2f} | ${max_loss:<9.2f}\n")
        f.write(f"{'='*80}\n")

    # Save annual CSV
    annual_df = pd.DataFrame(annual_data)
    annual_path = os.path.join(run_dir, f"annual_{report_name}.csv")
    annual_df.to_csv(annual_path, index=False)

    logger.info(f"Report saved to {report_path}")
    logger.info(f"Annual CSV saved to {annual_path}")


def evaluate_model(model, env, n_episodes: int = 10, run_dir: str = None):
    """
    Evaluate trained model and collect all trades.

    Returns trades_df as single source of truth for all metrics.
    """
    all_stats = []
    all_trades = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        lstm_states = None
        episode_start = True
        done = False
        step_count = 0

        while not done:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_start = False
            step_count += 1

            # Progress logging for long episodes
            if step_count % 10000 == 0:
                logger.info(f"  Episode {ep+1} progress: {step_count} steps...")

        # Get episode stats (for Sharpe, MaxDD which need equity curve)
        stats = env.get_episode_stats()
        all_stats.append(stats)

        # EXTRACT TRADES - unwrap env to get trade history
        inner_env = env
        if hasattr(env, 'envs'):
            inner_env = env.envs[0]
        while hasattr(inner_env, 'env') and not hasattr(inner_env, 'risk_manager'):
            inner_env = inner_env.env

        if hasattr(inner_env, 'risk_manager') and inner_env.risk_manager.trade_history:
            for t in inner_env.risk_manager.trade_history:
                trade_copy = t.copy()
                trade_copy['episode'] = ep + 1
                all_trades.append(trade_copy)

        logger.info(f"Episode {ep+1}: Sharpe={stats['sharpe_ratio']:.3f}, "
                   f"Return={stats['total_return']*100:.2f}%, "
                   f"MaxDD={stats['max_drawdown']*100:.2f}%, "
                   f"Trades={stats['total_trades']}")

    # =========================================================================
    # CREATE TRADES DATAFRAME (single source of truth)
    # =========================================================================
    if not all_trades:
        logger.warning("No trades collected!")
        return all_stats, pd.DataFrame()

    # Flatten nested dictionaries
    flat_trades = []
    for t in all_trades:
        flat_t = t.copy()
        if 'entry_features' in t and isinstance(t['entry_features'], dict):
            for k, v in t['entry_features'].items():
                flat_t[f'entry_{k}'] = v
            del flat_t['entry_features']
        if 'exit_features' in t and isinstance(t['exit_features'], dict):
            for k, v in t['exit_features'].items():
                flat_t[f'exit_{k}'] = v
            del flat_t['exit_features']
        flat_trades.append(flat_t)

    trades_df = pd.DataFrame(flat_trades)

    # Save to CSV
    if run_dir:
        trades_path = os.path.join(run_dir, "evaluation_trades.csv")
        trades_df.to_csv(trades_path, index=False)
        logger.info(f"Trades saved to: {trades_path}")

    # =========================================================================
    # CALCULATE ALL METRICS FROM trades_df (single source of truth)
    # =========================================================================
    total_trades = len(trades_df)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]

    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0.0
    total_pnl = trades_df['pnl'].sum()

    gross_wins = wins['pnl'].sum() if len(wins) > 0 else 0.0
    gross_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else (999.99 if gross_wins > 0 else 0.0)

    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0.0
    avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0.0
    max_win = trades_df['pnl'].max() if total_trades > 0 else 0.0
    max_loss = trades_df['pnl'].min() if total_trades > 0 else 0.0

    # Sharpe and MaxDD from episode stats (need equity curve)
    def safe_mean(values):
        clean = [v for v in values if v is not None and not np.isnan(v) and not np.isinf(v)]
        return np.mean(clean) if clean else 0.0

    avg_sharpe = safe_mean([s['sharpe_ratio'] for s in all_stats])
    avg_return = safe_mean([s['total_return'] for s in all_stats])
    avg_maxdd = safe_mean([s['max_drawdown'] for s in all_stats])

    # =========================================================================
    # PRINT EVALUATION SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Trades:     {total_trades}")
    logger.info(f"Win Rate:         {win_rate:.1f}%")
    logger.info(f"Profit Factor:    {profit_factor:.2f}")
    logger.info(f"Total PnL:        ${total_pnl:.2f}")
    logger.info(f"Avg Win:          ${avg_win:.2f}")
    logger.info(f"Avg Loss:         ${avg_loss:.2f}")
    logger.info(f"Largest Win:      ${max_win:.2f}")
    logger.info(f"Largest Loss:     ${max_loss:.2f}")
    logger.info(f"-" * 40)
    logger.info(f"Avg Sharpe:       {avg_sharpe:.3f}")
    logger.info(f"Avg Return:       {avg_return*100:.2f}%")
    logger.info(f"Avg Max DD:       {avg_maxdd*100:.2f}%")

    # Exit reasons
    if 'exit_reason' in trades_df.columns:
        logger.info(f"-" * 40)
        logger.info("Exit Reasons:")
        for reason, count in trades_df['exit_reason'].value_counts().items():
            pct = count / total_trades * 100
            logger.info(f"  {reason:<12}: {count:>5} ({pct:.1f}%)")

    logger.info("=" * 60)

    return all_stats, trades_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XAU-SNIPER PPO-LSTM")

    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory containing data files")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Directory for outputs")
    parser.add_argument("--timesteps", type=int, default=2_000_000,
                       help="Total training timesteps")
    parser.add_argument("--eval_freq", type=int, default=10_000,
                       help="Evaluation frequency")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to resume training from")
    parser.add_argument("--save_freq", type=int, default=50_000,
                       help="Model checkpoint frequency")

    args = parser.parse_args()

    train_agent(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        seed=args.seed,
        resume_path=args.resume
    )
