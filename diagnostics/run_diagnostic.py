"""
XAU-SNIPER Diagnostic Test V2
==============================

Test completo con 500k steps y metricas detalladas.
Configurado para estrategia SNIPER: 1-20 trades/semana.

Features:
- RSI Gold-Optimized (21, 75/25)
- MACD Gold-Optimized (16/34/13)
- High entry penalty for sniper behavior
- Extended episodes (1 week = 120 hours H1)
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
import warnings
import functools

# Force unbuffered output
print = functools.partial(print, flush=True)

warnings.filterwarnings('ignore')

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.callbacks import BaseCallback


class DetailedDiagnosticCallback(BaseCallback):
    """
    Callback de diagnostico detallado para estrategia SNIPER.
    Muestra toda la informacion necesaria para mejorar el modelo.
    """

    def __init__(self, check_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq

        # Metricas de entrenamiento
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_trades = []
        self.episode_win_rates = []
        self.episode_sharpes = []
        self.episode_returns = []
        self.trades_per_week = []

        # Distribucion de acciones
        self.action_counts = defaultdict(int)
        self.action_history = []

        # Trades detallados
        self.all_trades = []
        self.trade_pnls = []
        self.trade_durations = []
        self.trade_exit_reasons = defaultdict(int)

        # Learning progress
        self.reward_history = []
        self.best_reward = -np.inf
        self.best_sharpe = -np.inf
        self.improvement_steps = []

        # Policy behavior
        self.hold_streaks = []
        self.current_hold_streak = 0

    def _on_step(self) -> bool:
        # Registrar accion
        action = self.locals.get('actions', [0])[0]
        self.action_counts[int(action)] += 1
        self.action_history.append(int(action))

        # Track hold streaks
        if action == 0:
            self.current_hold_streak += 1
        else:
            if self.current_hold_streak > 0:
                self.hold_streaks.append(self.current_hold_streak)
            self.current_hold_streak = 0

        # Obtener info del ambiente
        infos = self.locals.get('infos', [{}])
        if infos:
            info = infos[0]

            # Verificar fin de episodio
            if 'episode' in info:
                ep_reward = info['episode']['r']
                ep_length = info['episode']['l']
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                self.reward_history.append(ep_reward)

                # Track best
                if ep_reward > self.best_reward:
                    self.best_reward = ep_reward
                    self.improvement_steps.append(self.n_calls)

            # Stats de trades
            if 'trade_stats' in info:
                stats = info['trade_stats']
                if stats.get('total_trades', 0) > 0:
                    self.episode_trades.append(stats['total_trades'])
                    self.episode_win_rates.append(stats.get('win_rate', 0))

            # Trades per week estimate
            if 'trades_per_week_est' in info:
                self.trades_per_week.append(info['trades_per_week_est'])

        # Imprimir diagnostico periodicamente
        if self.n_calls % self.check_freq == 0:
            self._print_detailed_diagnostic()

        return True

    def _print_detailed_diagnostic(self):
        """Imprimir diagnostico detallado."""
        print("\n" + "="*80)
        print(f"[DIAGNOSTIC] Step {self.n_calls:,} / Training Progress")
        print("="*80)

        # 1. Distribucion de acciones
        total_actions = sum(self.action_counts.values())
        print("\n[1] DISTRIBUCION DE ACCIONES:")
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
        for action, count in sorted(self.action_counts.items()):
            pct = count / total_actions * 100 if total_actions > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"    {action_names.get(action, action):5s}: {count:>8,} ({pct:>5.1f}%) {bar}")

        # Alertas de acciones
        hold_pct = self.action_counts.get(0, 0) / total_actions * 100 if total_actions > 0 else 0
        buy_pct = self.action_counts.get(1, 0) / total_actions * 100 if total_actions > 0 else 0
        sell_pct = self.action_counts.get(2, 0) / total_actions * 100 if total_actions > 0 else 0

        if hold_pct > 95:
            print("    [!] PROBLEMA: Modelo hace HOLD >95% - no esta explorando")
        elif hold_pct < 50:
            print("    [!] PROBLEMA: Modelo hace HOLD <50% - demasiados trades")
        else:
            print(f"    [OK] Balance razonable de acciones")

        # Sesgo direccional
        if buy_pct > 0 and sell_pct > 0:
            bias_ratio = buy_pct / sell_pct
            if bias_ratio > 5:
                print(f"    [i] Sesgo LONG fuerte ({bias_ratio:.1f}x) - Normal para oro")
            elif bias_ratio < 0.2:
                print(f"    [!] Sesgo SHORT fuerte - Inusual para oro")

        # 2. Episodios y Rewards
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-50:]
            print(f"\n[2] EPISODIOS ({len(self.episode_rewards)} completados):")
            print(f"    Reward promedio (ultimos 50): {np.mean(recent_rewards):>8.4f}")
            print(f"    Reward min/max:               {min(recent_rewards):>8.4f} / {max(recent_rewards):>8.4f}")
            print(f"    Reward std:                   {np.std(recent_rewards):>8.4f}")
            print(f"    Mejor reward historico:       {self.best_reward:>8.4f}")

            # Tendencia de mejora
            if len(self.episode_rewards) >= 20:
                first_10 = np.mean(self.episode_rewards[:10])
                last_10 = np.mean(self.episode_rewards[-10:])
                improvement = ((last_10 - first_10) / abs(first_10) * 100) if first_10 != 0 else 0
                trend = "MEJORANDO" if improvement > 5 else "ESTANCADO" if abs(improvement) < 5 else "EMPEORANDO"
                print(f"    Tendencia: {trend} ({improvement:+.1f}% desde inicio)")

        # 3. Trades
        if self.episode_trades:
            avg_trades = np.mean(self.episode_trades)
            print(f"\n[3] TRADES:")
            print(f"    Trades por episodio:          {avg_trades:>8.1f}")
            print(f"    Win rate promedio:            {np.mean(self.episode_win_rates)*100:>8.1f}%")

            if self.trades_per_week:
                avg_tpw = np.mean(self.trades_per_week[-20:])
                print(f"    Trades/semana estimado:       {avg_tpw:>8.1f}")

                # Evaluar si esta en el rango objetivo
                if 1 <= avg_tpw <= 20:
                    print(f"    [OK] En rango objetivo (1-20 trades/semana)")
                elif avg_tpw < 1:
                    print(f"    [!] Muy pocos trades - considerar reducir entry_penalty")
                else:
                    print(f"    [!] Demasiados trades - considerar aumentar entry_penalty")
        else:
            print(f"\n[3] TRADES: Ninguno aun")

        # 4. Comportamiento del Modelo
        if len(self.hold_streaks) > 0:
            print(f"\n[4] COMPORTAMIENTO:")
            print(f"    Hold streak promedio:         {np.mean(self.hold_streaks):>8.1f} pasos")
            print(f"    Hold streak maximo:           {max(self.hold_streaks):>8d} pasos")
            print(f"    Variabilidad de acciones:     {len(set(self.action_history[-100:])):>8d}/3 tipos")

        # 5. Learning Progress
        if len(self.improvement_steps) > 0:
            print(f"\n[5] PROGRESO DE APRENDIZAJE:")
            print(f"    Mejoras registradas:          {len(self.improvement_steps):>8d}")
            if len(self.improvement_steps) > 1:
                avg_steps_between = np.mean(np.diff(self.improvement_steps))
                print(f"    Steps entre mejoras:          {avg_steps_between:>8.0f}")

        print("\n" + "-"*80)


def run_full_diagnostic(
    timesteps: int = 500000,
    episode_length: int = 120,  # 1 week = 5 trading days * 24 hours = 120 H1 bars
    check_freq: int = 25000,
    entry_penalty: float = 0.01
):
    """
    Ejecutar diagnostico completo con 500k steps.

    Episode length de 120 H1 = 1 semana de trading.
    Target: 1-20 trades por semana (sniper strategy).
    """
    print("="*80)
    print("[TEST] XAU-SNIPER FULL DIAGNOSTIC - 500K STEPS")
    print("="*80)
    print(f"Timesteps: {timesteps:,}")
    print(f"Episode length: {episode_length} H1 bars (~{episode_length/24/5:.1f} weeks)")
    print(f"Check frequency: {check_freq:,}")
    print(f"Entry penalty: {entry_penalty}")
    print("="*80)

    # Importar modulos del proyecto
    from env.gold_env import GoldEnv
    from env.risk_manager import RiskConfig
    from features.h1_features import add_features_h1, get_feature_columns, H1FeatureConfig

    # 1. Cargar datos
    print("\n[DATA] Cargando datos...")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    h1_path = os.path.join(data_dir, 'xauusd_h1.csv')
    df_h1 = pd.read_csv(h1_path)
    print(f"    H1 data: {len(df_h1):,} filas ({len(df_h1)/24/5/52:.1f} years)")

    # Cargar M1
    m1_path = os.path.join(data_dir, 'xauusd_m1.csv')
    if os.path.exists(m1_path):
        # Cargar mas M1 data para episodios mas largos
        df_m1 = pd.read_csv(m1_path, nrows=2000000)
        print(f"    M1 data: {len(df_m1):,} filas")
    else:
        df_m1 = None
        print("    M1 data: No disponible")

    # Cargar DXY (correlacion inversa con oro)
    dxy_path = os.path.join(data_dir, 'dxy_daily.csv')
    if os.path.exists(dxy_path):
        dxy_df = pd.read_csv(dxy_path)
        print(f"    DXY data: {len(dxy_df):,} filas")
    else:
        dxy_df = None
        print("    DXY data: No disponible")

    # 2. Calcular features con parametros optimizados para oro
    print("\n[FEATURES] Calculando features GOLD-OPTIMIZED...")
    config = H1FeatureConfig()
    print(f"    RSI: Period={config.rsi_period}, OB={config.rsi_overbought}, OS={config.rsi_oversold}")
    print(f"    MACD: Fast={config.macd_fast}, Slow={config.macd_slow}, Signal={config.macd_signal}")

    # Agregar features incluyendo DXY si disponible
    df_h1 = add_features_h1(df_h1, dxy_df=dxy_df)
    feature_cols = get_feature_columns()
    print(f"    Total features: {len(feature_cols)}")
    if dxy_df is not None:
        print(f"    DXY features: Incluidos")

    # Verificar features importantes
    print("\n[VERIFY] Features clave:")
    key_features = ['rsi_norm', 'macd_norm', 'macd_hist_norm', 'atr_ratio', 'trend_dir']
    for col in key_features:
        if col in df_h1.columns:
            vals = df_h1[col].dropna()
            print(f"    {col:20s}: min={vals.min():>7.3f}, max={vals.max():>7.3f}, mean={vals.mean():>7.3f}")

    # Verificar sesgo en tendencia
    if 'trend_dir' in df_h1.columns:
        trend = df_h1['trend_dir'].dropna()
        bullish_pct = (trend > 0.3).sum() / len(trend) * 100
        bearish_pct = (trend < -0.3).sum() / len(trend) * 100
        neutral_pct = 100 - bullish_pct - bearish_pct
        print(f"\n[TREND] Distribucion de tendencia en datos:")
        print(f"    Alcista (>0.3):  {bullish_pct:5.1f}%")
        print(f"    Bajista (<-0.3): {bearish_pct:5.1f}%")
        print(f"    Neutral:         {neutral_pct:5.1f}%")
        if bullish_pct > bearish_pct * 1.5:
            print(f"    [!] SESGO ALCISTA detectado - oro tiende a subir")
        elif bearish_pct > bullish_pct * 1.5:
            print(f"    [!] SESGO BAJISTA detectado - oro tiende a bajar")
        else:
            print(f"    [OK] Tendencia balanceada")

    # 3. Crear ambiente SNIPER
    print("\n[ENV] Creando ambiente SNIPER...")
    risk_config = RiskConfig(
        tp_target=0.003,      # 0.3% target (trigger for trailing)
        sl_initial=0.003,     # 0.3% stop (R:R 1:1, trailing puede dar 1:5+)
        trailing_distance=0.0015,  # Trailing ajustado
        # Costos realistas del broker
        spread=0.0003,        # 3 pips
        slippage=0.0002,      # 2 pips
        commission=0.00005,   # 0.5 pips
    )
    print(f"    R:R = 1:1 (SL={risk_config.sl_initial*100:.1f}%, TP trigger={risk_config.tp_target*100:.1f}%)")
    print(f"    Trailing distance: {risk_config.trailing_distance*100:.2f}%")
    print(f"    Costos realistas:")
    print(f"      - Spread: {risk_config.spread*10000:.1f} pips")
    print(f"      - Slippage: {risk_config.slippage*10000:.1f} pips")
    print(f"      - Commission: {risk_config.commission*10000:.1f} pips")
    print(f"      - Total por trade: ~{risk_config.get_total_cost()*10000:.1f} pips (normal)")

    env = GoldEnv(
        df_h1=df_h1,
        df_m1=df_m1,
        feature_columns=feature_cols,
        initial_balance=10000.0,
        risk_config=risk_config,
        episode_length=episode_length,
        randomize_start=True,
        entry_penalty=entry_penalty,
    )

    print(f"    Observation space: {env.observation_space.shape}")
    print(f"    Action space: {env.action_space}")
    print(f"    Entry penalty: {entry_penalty}")

    # 4. Crear modelo PPO-LSTM
    print("\n[MODEL] Creando modelo PPO-LSTM...")
    try:
        from sb3_contrib import RecurrentPPO

        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=env,
            learning_rate=3e-5,   # Bajo para estabilidad
            n_steps=2048,         # ~2 semanas de data por update
            batch_size=128,
            n_epochs=10,
            gamma=0.999,          # Vision muy largo plazo
            gae_lambda=0.95,
            ent_coef=0.05,        # MUY ALTO para forzar exploracion (evitar convergencia prematura)
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={
                "lstm_hidden_size": 256,
                "n_lstm_layers": 1,
                "enable_critic_lstm": True,
                "net_arch": {'pi': [128, 64], 'vf': [128, 64]}
            },
            verbose=0,
            seed=42
        )
        print("    Modelo creado exitosamente")
        print(f"    LSTM hidden size: 256")
        print(f"    Learning rate: 3e-5")
        print(f"    Gamma: 0.999")
        print(f"    Entropy coef: 0.02 (aumentado para exploracion)")

    except ImportError:
        print("    [X] ERROR: sb3-contrib no instalado")
        return None, None

    # 5. Entrenar con diagnostico detallado
    print(f"\n[TRAIN] ENTRENANDO ({timesteps:,} steps)...")
    print("    Esto tomara varios minutos...\n")

    callback = DetailedDiagnosticCallback(check_freq=check_freq)

    start_time = datetime.now()
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[!] Entrenamiento interrumpido por usuario")

    training_time = datetime.now() - start_time
    print(f"\n[TIME] Tiempo de entrenamiento: {training_time}")

    # 6. Evaluacion final detallada
    print("\n" + "="*80)
    print("[EVAL] EVALUACION FINAL DETALLADA")
    print("="*80)

    final_stats = evaluate_model_detailed(model, env, n_episodes=10)

    # 7. Diagnostico de problemas y recomendaciones
    print("\n" + "="*80)
    print("[DIAGNOSIS] ANALISIS Y RECOMENDACIONES")
    print("="*80)

    analyze_and_recommend(callback, final_stats)

    # 8. Guardar modelo si es bueno
    if final_stats['avg_sharpe'] > 0.5:
        model_path = os.path.join(os.path.dirname(__file__), 'best_model.zip')
        model.save(model_path)
        print(f"\n[SAVED] Modelo guardado en: {model_path}")

    return model, callback, final_stats


def evaluate_model_detailed(model, env, n_episodes: int = 10):
    """Evaluacion detallada del modelo."""
    print("\n[EVAL] Evaluando modelo entrenado...")

    all_stats = []
    all_trades = []

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

        # Recopilar estadisticas
        stats = env.get_episode_stats()
        all_stats.append(stats)

        # Trades de este episodio
        for trade in env.risk_manager.trade_history:
            all_trades.append(trade)

        print(f"\n    Ep {ep+1}/{n_episodes}:")
        print(f"    - Return: {stats['total_return']*100:>6.2f}%")
        print(f"    - Sharpe: {stats['sharpe_ratio']:>6.3f}")
        print(f"    - Trades: {stats['total_trades']:>3d} (Est. {stats['trades_per_week_est']:.1f}/week)")
        print(f"    - Win Rate: {stats['win_rate']*100:>5.1f}%")
        print(f"    - Actions: HOLD={stats['hold_pct']:.0f}% BUY={stats['buy_pct']:.0f}% SELL={stats['sell_pct']:.0f}%")

    # Resumen
    print("\n" + "-"*60)
    print("[SUMMARY] RESUMEN DE EVALUACION:")

    avg_return = np.mean([s['total_return'] for s in all_stats])
    avg_sharpe = np.mean([s['sharpe_ratio'] for s in all_stats])
    avg_trades = np.mean([s['total_trades'] for s in all_stats])
    avg_tpw = np.mean([s['trades_per_week_est'] for s in all_stats])
    avg_winrate = np.mean([s['win_rate'] for s in all_stats])
    avg_hold = np.mean([s['hold_pct'] for s in all_stats])

    print(f"    Return promedio:     {avg_return*100:>7.2f}%")
    print(f"    Sharpe promedio:     {avg_sharpe:>7.3f}")
    print(f"    Trades promedio:     {avg_trades:>7.1f}")
    print(f"    Trades/semana:       {avg_tpw:>7.1f}")
    print(f"    Win rate:            {avg_winrate*100:>7.1f}%")
    print(f"    Hold %:              {avg_hold:>7.1f}%")

    # Analisis de trades
    if all_trades:
        print(f"\n[TRADES] ANALISIS DE {len(all_trades)} TRADES:")

        pnls = [t['pnl'] for t in all_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        print(f"    Ganadores: {len(wins)} ({len(wins)/len(pnls)*100:.1f}%)")
        print(f"    Perdedores: {len(losses)} ({len(losses)/len(pnls)*100:.1f}%)")

        if wins:
            print(f"    Ganancia promedio: ${np.mean(wins):.2f}")
        if losses:
            print(f"    Perdida promedio: ${np.mean(losses):.2f}")

        print(f"    PnL total: ${sum(pnls):.2f}")

        # Razones de cierre
        reasons = defaultdict(int)
        for t in all_trades:
            reasons[t.get('exit_reason', 'UNKNOWN')] += 1

        print(f"\n    Razones de cierre:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    - {reason}: {count} ({count/len(all_trades)*100:.1f}%)")

    return {
        'avg_return': avg_return,
        'avg_sharpe': avg_sharpe,
        'avg_trades': avg_trades,
        'avg_tpw': avg_tpw,
        'avg_winrate': avg_winrate,
        'avg_hold': avg_hold,
        'all_trades': all_trades,
        'all_stats': all_stats
    }


def analyze_and_recommend(callback, final_stats):
    """Analizar resultados y dar recomendaciones."""
    problems = []
    recommendations = []

    # 1. Analizar frecuencia de trades
    avg_tpw = final_stats['avg_tpw']
    if avg_tpw < 1:
        problems.append("[X] Muy pocos trades (< 1/semana)")
        recommendations.append("-> Reducir entry_penalty (actualmente muy alto)")
        recommendations.append("-> Aumentar ent_coef para mas exploracion")
    elif avg_tpw > 20:
        problems.append("[X] Demasiados trades (> 20/semana)")
        recommendations.append("-> Aumentar entry_penalty")
        recommendations.append("-> Reducir ent_coef")
    else:
        print(f"[OK] Frecuencia de trades en rango: {avg_tpw:.1f}/semana")

    # 2. Analizar win rate
    if final_stats['avg_winrate'] < 0.4:
        problems.append(f"[X] Win rate bajo ({final_stats['avg_winrate']*100:.1f}%)")
        recommendations.append("-> Revisar features - puede que no sean predictivas")
        recommendations.append("-> Considerar ajustar TP/SL ratios")
    elif final_stats['avg_winrate'] > 0.6:
        print(f"[OK] Win rate bueno: {final_stats['avg_winrate']*100:.1f}%")

    # 3. Analizar Sharpe
    if final_stats['avg_sharpe'] < 0.5:
        problems.append(f"[X] Sharpe bajo ({final_stats['avg_sharpe']:.3f})")
        recommendations.append("-> Necesita mas entrenamiento")
        recommendations.append("-> Considerar ajustar gamma para vision mas largo plazo")
    elif final_stats['avg_sharpe'] > 1.5:
        print(f"[OK] Sharpe excelente: {final_stats['avg_sharpe']:.3f}")

    # 4. Analizar balance de acciones
    if final_stats['avg_hold'] > 95:
        problems.append("[X] Modelo demasiado conservador (>95% HOLD)")
        recommendations.append("-> Aumentar ent_coef significativamente")
        recommendations.append("-> Reducir entry_penalty")
    elif final_stats['avg_hold'] < 50:
        problems.append("[X] Modelo demasiado activo (<50% HOLD)")
        recommendations.append("-> Aumentar entry_penalty")

    # 5. Analizar razones de cierre
    if final_stats['all_trades']:
        force_close_count = sum(1 for t in final_stats['all_trades'] if t.get('exit_reason') == 'FORCE_CLOSE')
        force_close_pct = force_close_count / len(final_stats['all_trades']) * 100

        if force_close_pct > 50:
            problems.append(f"[X] {force_close_pct:.0f}% trades cerrados por FORCE_CLOSE")
            recommendations.append("-> Episodios muy cortos o SL/TP muy lejanos")
            recommendations.append("-> Considerar ajustar episode_length o TP/SL")

    # Imprimir diagnostico
    if problems:
        print("\n[!] PROBLEMAS DETECTADOS:")
        for p in problems:
            print(f"    {p}")

        print("\n[*] RECOMENDACIONES:")
        for r in recommendations:
            print(f"    {r}")
    else:
        print("\n[OK] No se detectaron problemas criticos")

    # Configuracion sugerida
    print("\n" + "="*60)
    print("[CONFIG] PROXIMOS PASOS:")
    print("="*60)

    if final_stats['avg_sharpe'] < 1.0:
        print("""
    1. ENTRENAR MAS TIEMPO:
       - Aumentar timesteps a 1-2M si hay mejora constante

    2. AJUSTAR HIPERPARAMETROS:
       - Si muy pocos trades: entry_penalty = 0.005
       - Si demasiados trades: entry_penalty = 0.02
       - Si no explora: ent_coef = 0.01

    3. MEJORAR FEATURES:
       - Agregar mas indicadores de momentum
       - Considerar features de volumen
       - Agregar correlaciones macro (DXY, yields)
        """)
    else:
        print("""
    [EXCELENTE] El modelo muestra buen rendimiento.

    PROXIMOS PASOS:
    1. Guardar el modelo
    2. Hacer backtest en datos out-of-sample
    3. Paper trading para validar
        """)

    # Guardar reporte
    save_report(callback, final_stats)


def save_report(callback, final_stats):
    """Guardar reporte detallado."""
    report_path = os.path.join(os.path.dirname(__file__), 'diagnostic_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("XAU-SNIPER DIAGNOSTIC REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("FINAL METRICS:\n")
        f.write(f"  Avg Return: {final_stats['avg_return']*100:.2f}%\n")
        f.write(f"  Avg Sharpe: {final_stats['avg_sharpe']:.3f}\n")
        f.write(f"  Avg Trades: {final_stats['avg_trades']:.1f}\n")
        f.write(f"  Trades/Week: {final_stats['avg_tpw']:.1f}\n")
        f.write(f"  Win Rate: {final_stats['avg_winrate']*100:.1f}%\n")

        f.write("\nACTION DISTRIBUTION:\n")
        total = sum(callback.action_counts.values())
        for action, count in sorted(callback.action_counts.items()):
            pct = count / total * 100 if total > 0 else 0
            f.write(f"  Action {action}: {count:,} ({pct:.1f}%)\n")

        f.write(f"\nEPISODES COMPLETED: {len(callback.episode_rewards)}\n")
        if callback.episode_rewards:
            f.write(f"  Best Reward: {callback.best_reward:.4f}\n")
            f.write(f"  Improvements: {len(callback.improvement_steps)}\n")

    print(f"\n[SAVED] Reporte guardado en: {report_path}")


if __name__ == "__main__":
    # Crear directorio de diagnosticos si no existe
    os.makedirs(os.path.dirname(__file__), exist_ok=True)

    # Ejecutar diagnostico completo
    # Episode length = 120 H1 bars = 1 semana de trading (24h * 5 dias)
    # Target: 1-20 trades por semana
    model, callback, stats = run_full_diagnostic(
        timesteps=500000,         # 500k steps
        episode_length=120,       # 1 semana = 120 H1 bars (CORREGIDO de 840)
        check_freq=50000,         # Diagnostico cada 50k
        entry_penalty=0.01        # Alta penalizacion para sniper
    )
