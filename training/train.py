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

    # Risk configuration
    risk_config = RiskConfig(
        tp_target=0.005,      # 0.5% target (was 0.3%)
        sl_initial=0.0025,    # 0.25% stop (was 0.15%)
        trailing_distance=0.0015,
        commission=0.00005,   # 0.5 pips (was 0.5!)
        slippage=0.0002,      # 2 pips (was 0.3!)
        use_atr_stops=True    # Enable dynamic stops
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

    # Final evaluation (Test Set - 2024-2025)
    logger.info("Running standard evaluation (Test Set)...")
    _, trades_test = evaluate_model(model, eval_env, n_episodes=10, run_dir=run_dir)
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
    Generate detailed Monthly and Annual report from trades DataFrame.
    """
    if trades_df.empty:
        return

    logger.info(f"\n Generating Advanced Report: {report_name} ")
    logger.info("=" * 60)

    # Ensure exit_date is datetime
    # Try different parsing strategies since 'exit_date' comes from str(current_time)
    try:
        trades_df['exit_dt'] = pd.to_datetime(trades_df['exit_date'], utc=True)
    except Exception as e:
        logger.error(f"Could not parse dates: {e}")
        return

    trades_df['year'] = trades_df['exit_dt'].dt.year
    trades_df['month'] = trades_df['exit_dt'].dt.month

    # Annual Stats
    annual_stats = []
    years = sorted(trades_df['year'].unique())
    
    print("\nANNUAL PERFORMANCE:")
    print("-" * 80)
    print(f"{'Year':<6} | {'Trades':<6} | {'Win%':<6} | {'Profit Factor':<13} | {'Total PnL':<10} | {'Avg PnL':<8}")
    print("-" * 80)

    for y in years:
        mask = trades_df['year'] == y
        y_trades = trades_df[mask]
        
        count = len(y_trades)
        wins = y_trades[y_trades['pnl'] > 0]
        losses = y_trades[y_trades['pnl'] <= 0]
        
        if count > 0:
            win_rate = (len(wins) / count) * 100
        else:
            win_rate = 0.0
        
        total_pnl = y_trades['pnl'].sum()
        avg_pnl = y_trades['pnl'].mean()
        
        gross_win = wins['pnl'].sum()
        gross_loss = abs(losses['pnl'].sum())
        pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
        
        print(f"{y:<6} | {count:<6} | {win_rate:<6.1f} | {pf:<13.2f} | {total_pnl:<10.2f} | {avg_pnl:<8.2f}")
        
    print("-" * 80)
    
    # Save report to file
    report_path = os.path.join(run_dir, f"report_{report_name}.txt")
    with open(report_path, "w") as f:
        f.write(f"ADVANCED REPORT: {report_name}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("ANNUAL PERFORMANCE:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Year':<6} | {'Trades':<6} | {'Win%':<6} | {'Profit Factor':<13} | {'Total PnL':<10} | {'Avg PnL':<8}\n")
        f.write("-" * 80 + "\n")
        
        for y in years:
            mask = trades_df['year'] == y
            y_trades = trades_df[mask]
            
            count = len(y_trades)
            wins = y_trades[y_trades['pnl'] > 0]
            losses = y_trades[y_trades['pnl'] <= 0]
            
            if count > 0:
                win_rate = (len(wins) / count) * 100
            else:
                win_rate = 0.0
                
            total_pnl = y_trades['pnl'].sum()
            avg_pnl = y_trades['pnl'].mean()
            
            gross_win = wins['pnl'].sum()
            gross_loss = abs(losses['pnl'].sum())
            pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
            
            line = f"{y:<6} | {count:<6} | {win_rate:<6.1f} | {pf:<13.2f} | {total_pnl:<10.2f} | {avg_pnl:<8.2f}\n"
            f.write(line)
            
        f.write("-" * 80 + "\n")

    logger.info(f"Report saved to {report_path}")


def evaluate_model(model, env, n_episodes: int = 10, run_dir: str = None):
    """
    Evaluate trained model.

    Args:
        model: Trained model
        env: Evaluation environment
        n_episodes: Number of evaluation episodes
    """
    all_stats = []
    super_complete_trades = [] # List to store all trades from all episodes

    for ep in range(n_episodes):
        obs, info = env.reset()
        lstm_states = None
        episode_start = True
        done = False

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

        stats = env.get_episode_stats()
        all_stats.append(stats)

        # EXTRACT TRADES BEFORE RESET
        # Need to unwrap env to get trade history
        inner_env = env
        if hasattr(env, 'envs'):
            inner_env = env.envs[0]
            
        while hasattr(inner_env, 'env') and not isinstance(inner_env, GoldEnv):
            inner_env = inner_env.env
            
        # Fallback for deep wrapping if GoldEnv still not found directly
        # But allow accessing if it has risk_manager
        while hasattr(inner_env, 'env') and not hasattr(inner_env, 'risk_manager'):
             inner_env = inner_env.env
        
        if hasattr(inner_env, 'risk_manager') and inner_env.risk_manager.trade_history:
            current_ep_trades = inner_env.risk_manager.trade_history
            # Append episode number to each trade for tracking
            for t in current_ep_trades:
                t['episode'] = ep + 1
                super_complete_trades.append(t)

        logger.info(f"Episode {ep+1}: Sharpe={stats['sharpe_ratio']:.3f}, "
                   f"Return={stats['total_return']*100:.2f}%, "
                   f"MaxDD={stats['max_drawdown']*100:.2f}%, "
                   f"Trades={stats['total_trades']}")

    # Average stats
    avg_sharpe = np.mean([s['sharpe_ratio'] for s in all_stats])
    avg_return = np.mean([s['total_return'] for s in all_stats])
    avg_maxdd = np.mean([s['max_drawdown'] for s in all_stats])
    avg_trades = np.mean([s['total_trades'] for s in all_stats])
    avg_winrate = np.mean([s['win_rate'] for s in all_stats])
    
    # Calculate additional metrics
    total_gains = sum(s.get('total_gains', 0) for s in all_stats)
    total_losses = abs(sum(s.get('total_losses', 0) for s in all_stats))
    profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')

    logger.info("\n" + "=" * 50)
    logger.info("FINAL EVALUATION REPORT (SUPER COMPLETE)")
    logger.info("=" * 50)
    logger.info(f"Avg Sharpe Ratio: {avg_sharpe:.3f}")
    logger.info(f"Avg Return: {avg_return*100:.2f}%")
    logger.info(f"Avg Max Drawdown: {avg_maxdd*100:.2f}%")
    # Scale trades to annual (assuming 2048 hours per episode)
    # Year = 8760 hours (approx) -> Scale factor = 8760 / 2048 = 4.28
    avg_trades_annual = avg_trades * (8760 / 2048)
    logger.info(f"Avg Trades/Year: {avg_trades_annual:.1f} (Avg/Ep: {avg_trades:.1f})")
    logger.info(f"Avg Win Rate: {avg_winrate*100:.1f}%")
    logger.info(f"Profit Factor: {profit_factor:.2f}")
    logger.info("-" * 50)
    
    # Save detailed trade history
    if super_complete_trades:
        # Flatten nested dictionaries (entry_features, exit_features)
        flat_trades = []
        for t in super_complete_trades:
            flat_t = t.copy()
            
            # Flatten entry features
            if 'entry_features' in t and isinstance(t['entry_features'], dict):
                for k, v in t['entry_features'].items():
                    flat_t[f'entry_{k}'] = v
                del flat_t['entry_features']
                
            # Flatten exit features
            if 'exit_features' in t and isinstance(t['exit_features'], dict):
                for k, v in t['exit_features'].items():
                    flat_t[f'exit_{k}'] = v
                del flat_t['exit_features']
                
            flat_trades.append(flat_t)

        trades_df = pd.DataFrame(flat_trades)
        trades_path = os.path.join(run_dir, "evaluation_trades.csv")
        trades_df.to_csv(trades_path, index=False)
        logger.info(f"Detailed trade history saved to: {trades_path}")
        
        # Analyze trades df
        if not trades_df.empty:
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if not trades_df[trades_df['pnl'] > 0].empty else 0
            avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if not trades_df[trades_df['pnl'] <= 0].empty else 0
            logger.info(f"Avg Win: ${avg_win:.2f}")
            logger.info(f"Avg Loss: ${avg_loss:.2f}")
            logger.info(f"Largest Win: ${trades_df['pnl'].max():.2f}")
            logger.info(f"Largest Loss: ${trades_df['pnl'].min():.2f}")
        
            # Count closure reasons
            logger.info("Exit Reasons:")
            logger.info(trades_df['exit_reason'].value_counts().to_string())

    logger.info("=" * 50)

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
