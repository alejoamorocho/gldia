"""
Evaluate trained model and generate detailed annual report
"""
import pandas as pd
import numpy as np
from sb3_contrib import RecurrentPPO
from env.gold_env import GoldEnv
from env.risk_manager import RiskConfig
from features.h1_features import add_features_h1, get_feature_columns
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, default='outputs/run_20260114_190751/models/final_model',
                        help='Path to trained model')
    parser.add_argument('--output', type=str, default='evaluation_report.txt',
                        help='Output report file')
    args = parser.parse_args()

    print('='*70)
    print(f'FULL MODEL EVALUATION')
    print(f'Model: {args.model}')
    print('='*70)

    # Load data
    print('\n[1/5] Loading data...')
    df_h1 = pd.read_csv('data/xauusd_h1.csv')
    df_m1 = pd.read_csv('data/xauusd_m1.csv')
    dxy = pd.read_csv('data/dxy_daily.csv')
    us10y = pd.read_csv('data/us10y_daily.csv')
    vix = pd.read_csv('data/vix_daily.csv')
    print(f'    H1 data: {len(df_h1)} rows')
    print(f'    M1 data: {len(df_m1)} rows')

    # Add features
    print('\n[2/5] Computing features...')
    df_h1 = add_features_h1(df_h1, dxy_df=dxy, us10y_df=us10y, vix_df=vix)
    feature_columns = get_feature_columns()
    print(f'    Features: {len(feature_columns)}')

    # Prepare M1 data
    df_m1['time'] = pd.to_datetime(df_m1['time'])
    df_m1 = df_m1.set_index('time')

    # Risk config
    risk_config = RiskConfig(
        tp_target=0.003,
        sl_initial=0.003,
        trailing_distance=0.0015,
        breakeven_level=0.0015,
    )

    # Create environment
    print('\n[3/5] Creating environment...')
    env = GoldEnv(
        df_h1=df_h1,
        df_m1=df_m1,
        feature_columns=feature_columns,
        risk_config=risk_config,
        episode_length=len(df_h1),
        randomize_start=False
    )

    # Load model
    print('\n[4/5] Loading model...')
    model = RecurrentPPO.load(args.model)
    print('    Model loaded successfully')

    # Run evaluation
    print('\n[5/5] Running full backtest (2015-2025)...')
    obs, info = env.reset()
    lstm_states = None
    episode_start = True
    done = False
    step = 0
    total_steps = len(df_h1)

    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_start = False
        step += 1
        if step % 5000 == 0:
            pct = 100 * step / total_steps
            print(f'    Progress: {step:,}/{total_steps:,} steps ({pct:.1f}%)')

    print(f'\n    Backtest complete: {step:,} steps')

    # Get results
    stats = env.get_episode_stats()
    trades = env.risk_manager.trade_history

    # Create trades dataframe
    trades_df = pd.DataFrame(trades)

    if len(trades_df) == 0:
        print('\nNo trades executed!')
        return

    # Parse dates
    trades_df['exit_dt'] = pd.to_datetime(trades_df['exit_date'], errors='coerce')
    trades_df = trades_df[trades_df['exit_dt'].notna()]
    trades_df['year'] = trades_df['exit_dt'].dt.year.astype(int)

    # Generate report
    print('\n' + '='*70)
    print('EVALUATION RESULTS')
    print('='*70)

    # Overall stats
    total_trades = len(trades_df)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]

    total_pnl = trades_df['pnl'].sum()
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    gross_wins = wins['pnl'].sum() if len(wins) > 0 else 0
    gross_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else 999.99

    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0

    # Calculate max drawdown from equity curve
    equity = np.array(env.equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / np.where(peak == 0, 1, peak)
    max_dd = np.max(drawdown) * 100

    print(f'\nOVERALL PERFORMANCE:')
    print('-'*50)
    print(f'Total Trades:     {total_trades}')
    print(f'Win Rate:         {win_rate:.1f}%')
    print(f'Profit Factor:    {profit_factor:.2f}')
    print(f'Total PnL:        ${total_pnl:.2f}')
    print(f'Max Drawdown:     {max_dd:.2f}%')
    print(f'Avg Win:          ${avg_win:.2f}')
    print(f'Avg Loss:         ${avg_loss:.2f}')
    print(f'Sharpe Ratio:     {stats["sharpe_ratio"]:.2f}')
    print(f'Total Return:     {stats["total_return"]*100:.2f}%')

    # Annual breakdown
    print(f'\nANNUAL PERFORMANCE:')
    print('-'*100)
    print(f'{"Year":<6} | {"Trades":<7} | {"Win%":<6} | {"PF":<8} | {"Total PnL":<12} | {"Avg PnL":<10} | {"Max DD":<8} | {"Sharpe":<8}')
    print('-'*100)

    years = sorted(trades_df['year'].unique())
    annual_data = []

    for y in years:
        y_trades = trades_df[trades_df['year'] == y]
        y_wins = y_trades[y_trades['pnl'] > 0]
        y_losses = y_trades[y_trades['pnl'] <= 0]

        count = len(y_trades)
        y_win_rate = len(y_wins) / count * 100 if count > 0 else 0
        y_total_pnl = y_trades['pnl'].sum()
        y_avg_pnl = y_trades['pnl'].mean() if count > 0 else 0

        y_gross_wins = y_wins['pnl'].sum() if len(y_wins) > 0 else 0
        y_gross_losses = abs(y_losses['pnl'].sum()) if len(y_losses) > 0 else 0
        y_pf = y_gross_wins / y_gross_losses if y_gross_losses > 0 else 999.99

        # Estimate max DD for year (simplified)
        y_cumsum = y_trades['pnl'].cumsum()
        y_peak = y_cumsum.cummax()
        y_dd = (y_peak - y_cumsum)
        y_max_dd = y_dd.max() if len(y_dd) > 0 else 0

        # Estimate sharpe for year
        y_returns = y_trades['pnl'] / 10000  # Normalize by account size
        y_sharpe = y_returns.mean() / y_returns.std() * np.sqrt(252) if y_returns.std() > 0 else 0

        print(f'{y:<6} | {count:<7} | {y_win_rate:<6.1f} | {y_pf:<8.2f} | ${y_total_pnl:<11.2f} | ${y_avg_pnl:<9.2f} | ${y_max_dd:<7.2f} | {y_sharpe:<8.2f}')

        annual_data.append({
            'year': y, 'trades': count, 'win_rate': y_win_rate,
            'pf': y_pf, 'total_pnl': y_total_pnl, 'avg_pnl': y_avg_pnl,
            'max_dd': y_max_dd, 'sharpe': y_sharpe
        })

    print('-'*100)

    # Exit reasons
    print(f'\nEXIT REASONS:')
    print('-'*30)
    for reason, count in trades_df['exit_reason'].value_counts().items():
        pct = count / total_trades * 100
        print(f'{reason:<15}: {count:>5} ({pct:.1f}%)')

    # Trade sides
    print(f'\nTRADE SIDES:')
    print('-'*30)
    for side, count in trades_df['side'].value_counts().items():
        pct = count / total_trades * 100
        print(f'{side:<15}: {count:>5} ({pct:.1f}%)')

    # Save report
    print(f'\n' + '='*70)

    # Save trades to CSV
    trades_path = args.output.replace('.txt', '_trades.csv')
    trades_df.to_csv(trades_path, index=False)
    print(f'Trades saved to: {trades_path}')

    # Save annual summary
    annual_df = pd.DataFrame(annual_data)
    annual_path = args.output.replace('.txt', '_annual.csv')
    annual_df.to_csv(annual_path, index=False)
    print(f'Annual summary saved to: {annual_path}')

    print('\nEvaluation complete!')


if __name__ == '__main__':
    main()
