"""
XAU-SNIPER: PPO-LSTM Gold Trading Agent
========================================

Main entry point for training, evaluation, and backtesting.

Usage:
    python main.py train --timesteps 2000000
    python main.py backtest --model outputs/best_model.zip
    python main.py evaluate --model outputs/best_model.zip
"""

import argparse
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="XAU-SNIPER: PPO-LSTM Gold Trading Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train a new model:
    python main.py train --timesteps 2000000

  Resume training:
    python main.py train --resume outputs/run_xxx/models/best_model.zip

  Run backtest:
    python main.py backtest --model outputs/run_xxx/models/best_model.zip

  Evaluate model:
    python main.py evaluate --model outputs/run_xxx/models/best_model.zip --episodes 20
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--data_dir', type=str, default='data',
                             help='Directory containing data files')
    train_parser.add_argument('--output_dir', type=str, default='outputs',
                             help='Directory for outputs')
    train_parser.add_argument('--timesteps', type=int, default=2_000_000,
                             help='Total training timesteps')
    train_parser.add_argument('--seed', type=int, default=42,
                             help='Random seed')
    train_parser.add_argument('--resume', type=str, default=None,
                             help='Path to resume training from')

    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--model', type=str, required=True,
                                help='Path to trained model')
    backtest_parser.add_argument('--data_dir', type=str, default='data',
                                help='Directory containing data files')
    backtest_parser.add_argument('--output', type=str, default=None,
                                help='Output file for results')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--model', type=str, required=True,
                            help='Path to trained model')
    eval_parser.add_argument('--data_dir', type=str, default='data',
                            help='Directory containing data files')
    eval_parser.add_argument('--episodes', type=int, default=10,
                            help='Number of evaluation episodes')

    # Test command (quick sanity check)
    test_parser = subparsers.add_parser('test', help='Run quick test')

    args = parser.parse_args()

    if args.command == 'train':
        run_training(args)
    elif args.command == 'backtest':
        run_backtest_cmd(args)
    elif args.command == 'evaluate':
        run_evaluation(args)
    elif args.command == 'test':
        run_test()
    else:
        parser.print_help()


def run_training(args):
    """Run training."""
    from training.train import train_agent

    logger.info("Starting training...")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Timesteps: {args.timesteps:,}")

    train_agent(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        total_timesteps=args.timesteps,
        seed=args.seed,
        resume_path=args.resume
    )


def run_backtest_cmd(args):
    """Run backtest."""
    import pandas as pd
    from sb3_contrib import RecurrentPPO

    from env.gold_env import GoldEnv
    from env.risk_manager import RiskConfig
    from features.h1_features import add_features_h1, get_feature_columns
    from evaluation.backtest import run_backtest

    logger.info("Running backtest...")
    logger.info(f"Model: {args.model}")

    # Load data
    df_h1 = pd.read_csv(os.path.join(args.data_dir, 'xauusd_h1.csv'))

    m1_path = os.path.join(args.data_dir, 'xauusd_m1.csv')
    df_m1 = pd.read_csv(m1_path) if os.path.exists(m1_path) else None

    # Add features
    df_h1 = add_features_h1(df_h1)
    feature_columns = get_feature_columns()

    # Create dummy env for model loading
    env = GoldEnv(
        df_h1=df_h1,
        df_m1=df_m1,
        feature_columns=feature_columns,
        episode_length=1000
    )

    # Load model
    model = RecurrentPPO.load(args.model, env=env)

    # Run backtest
    result = run_backtest(
        model=model,
        df_h1=df_h1,
        df_m1=df_m1,
        feature_columns=feature_columns,
        verbose=True
    )

    # Save results if output specified
    if args.output:
        result.to_dataframe().to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")


def run_evaluation(args):
    """Run evaluation."""
    import pandas as pd
    from sb3_contrib import RecurrentPPO

    from env.gold_env import GoldEnv
    from env.risk_manager import RiskConfig
    from features.h1_features import add_features_h1, get_feature_columns
    from training.train import evaluate_model

    logger.info("Running evaluation...")
    logger.info(f"Model: {args.model}")
    logger.info(f"Episodes: {args.episodes}")

    # Load data
    df_h1 = pd.read_csv(os.path.join(args.data_dir, 'xauusd_h1.csv'))

    m1_path = os.path.join(args.data_dir, 'xauusd_m1.csv')
    df_m1 = pd.read_csv(m1_path) if os.path.exists(m1_path) else None

    # Add features
    df_h1 = add_features_h1(df_h1)
    feature_columns = get_feature_columns()

    # Create env
    env = GoldEnv(
        df_h1=df_h1,
        df_m1=df_m1,
        feature_columns=feature_columns,
        episode_length=2048,
        randomize_start=True
    )

    # Load model
    model = RecurrentPPO.load(args.model, env=env)

    # Run evaluation
    evaluate_model(model, env, n_episodes=args.episodes)


def run_test():
    """Run quick sanity test."""
    logger.info("Running quick test...")

    try:
        # Test imports
        from env.gold_env import GoldEnv
        from env.risk_manager import RiskManager, RiskConfig
        from features.h1_features import add_features_h1, get_feature_columns
        from models.ppo_lstm import PPOLSTMConfig
        from evaluation.metrics import calculate_all_metrics

        logger.info("All imports successful")

        # Test data loading
        import pandas as pd
        import numpy as np

        data_path = 'data/xauusd_h1.csv'
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            logger.info(f"Loaded data: {len(df)} rows")

            # Test feature computation
            df = add_features_h1(df)
            logger.info(f"Features computed: {len(get_feature_columns())} features")

            # Test environment creation
            env = GoldEnv(
                df_h1=df,
                episode_length=100,
                randomize_start=False
            )

            # Test reset and step
            obs, info = env.reset()
            logger.info(f"Observation shape: {obs.shape}")

            for i in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated:
                    break

            logger.info(f"Episode stats: {env.get_episode_stats()}")
            logger.info("Environment test passed!")
        else:
            logger.warning(f"Data file not found: {data_path}")

        logger.info("\nAll tests passed!")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
