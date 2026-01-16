"""
Gold Trading Environment for PPO-LSTM
======================================

Gym-compatible environment for XAU/USD trading.
OPTIMIZED FOR SNIPER STRATEGY: 1-20 trades per week max.

Architecture:
- H1 data: Observation space (what the AI sees)
- M1 data: Execution simulation (internal risk management)

Actions:
- 0: HOLD/WAIT
- 1: BUY/LONG
- 2: SELL/SHORT

Note: Gold is historically bullish (~70% of time), so the model
may naturally favor LONG positions. This is expected behavior.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from datetime import timedelta
import logging

from .risk_manager import RiskManager, RiskConfig, PositionSide, PositionState

logger = logging.getLogger(__name__)


class GoldEnv(gym.Env):
    """
    Gold Trading Environment - SNIPER STRATEGY.

    Observation: H1 features (normalized)
    Action: Discrete(3) - HOLD, BUY, SELL
    Reward: Sharpe-optimized with HIGH entry penalty for sniper behavior

    Target: 1-20 trades per week (very selective)
    """

    metadata = {'render_modes': ['human']}

    # Action constants
    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(
        self,
        df_h1: pd.DataFrame,
        df_m1: Optional[pd.DataFrame] = None,
        feature_columns: Optional[list] = None,
        initial_balance: float = 10000.0,
        risk_config: Optional[RiskConfig] = None,
        episode_length: Optional[int] = None,
        randomize_start: bool = True,
        render_mode: Optional[str] = None,
        # Sniper parameters
        entry_penalty: float = 0.0,  # Zero penalty to encourage max frequency
        target_trades_per_week: int = 12,  # Target ~12 trades/week (~300/year)
        gold_bullish_bias: float = 0.7,  # Gold is bullish ~70% of time
    ):
        """
        Initialize Gold trading environment.

        Args:
            df_h1: Hourly data with features
            df_m1: Minute data for execution (optional)
            feature_columns: List of columns to use as observations
            initial_balance: Starting capital
            risk_config: Risk management configuration
            episode_length: Max steps per episode (None = full data)
            randomize_start: Randomize starting point each reset
            render_mode: Rendering mode
            entry_penalty: Penalty for opening trades (higher = fewer trades)
            target_trades_per_week: Target number of trades per week
            gold_bullish_bias: Expected bullish ratio for gold
        """
        super().__init__()

        self.render_mode = render_mode

        # Sniper parameters
        self.entry_penalty = entry_penalty
        self.target_trades_per_week = target_trades_per_week
        self.gold_bullish_bias = gold_bullish_bias

        # Store data
        self.df_h1_raw = df_h1.copy()
        self.df_m1_raw = df_m1.copy() if df_m1 is not None else None

        # Prepare data
        self._prepare_data()

        # Feature columns
        if feature_columns is None:
            # Auto-detect feature columns (exclude OHLCV)
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'time']
            self.feature_columns = [c for c in self.df_h1.columns if c not in exclude_cols]
        else:
            self.feature_columns = feature_columns

        self.n_features = len(self.feature_columns)

        # Configuration
        self.initial_balance = initial_balance
        self.episode_length = episode_length
        self.randomize_start = randomize_start

        # Risk manager
        self.risk_manager = RiskManager(risk_config or RiskConfig())

        # Define spaces
        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL

        # Observation: features + position info + market context + behavior info + entry conditions + trade memory
        # Position info: [in_position, position_side, unrealized_pnl_norm, duration_norm]
        # Market context: [recent_volatility, trend_strength]
        # Behavior info: [trade_ratio, trade_saturation, steps_since_trade, win_rate, atr_filter]
        # Entry conditions: [entry_rsi, entry_macd, entry_atr, entry_trend, price_vs_entry]
        # Trade memory: [last_win_rsi, last_win_macd, last_win_trend, last_win_side,
        #                last_loss_rsi, last_loss_macd, last_loss_trend, last_loss_side,
        #                similar_to_win]
        obs_dim = self.n_features + 6 + 5 + 5 + 9  # +5 behavior, +5 entry, +9 trade memory
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # State variables
        self.current_step = 0
        self.start_step = 0
        self.balance = initial_balance
        self.equity_curve = []
        self.prev_equity = initial_balance

        # Reward tracking
        self.just_opened_trade = False
        self.just_closed_trade = False
        self.last_trade_pnl = 0.0
        self.last_trade_entry_price = 0.0

        # Episode statistics for diagnostics
        self.episode_actions = []
        self.episode_rewards = []
        self.consecutive_holds = 0
        self.max_consecutive_holds = 0

        # Behavior tracking (for observation space)
        self.steps_since_last_trade = 0
        self.episode_win_rate = 0.0

        # Entry conditions tracking (para que el modelo aprenda patrones ganadores)
        self.entry_conditions = {
            'rsi': 0.0,
            'macd': 0.0,
            'atr': 0.0,
            'trend': 0.0,
            'price': 0.0
        }

        # Historial de ultimos trades - para que el modelo aprenda de resultados
        # Guarda condiciones de entrada + resultado (1=win, 0=loss)
        self.trade_memory = []  # Lista de dicts con condiciones y resultado
        self.last_win_conditions = {'rsi': 0.0, 'macd': 0.0, 'trend': 0.0, 'side': 0.0}
        self.last_loss_conditions = {'rsi': 0.0, 'macd': 0.0, 'trend': 0.0, 'side': 0.0}

        # Calculate hours per week for trade frequency
        self.hours_per_week = 24 * 5  # 5 trading days

        # Weekly bonus tracking (give bonus ONCE when reaching 5 trades/week)
        self.weekly_trades_count = 0  # Trades in current week
        self.current_week_number = -1  # Track which week we're in
        self.weekly_bonus_given = False  # Has bonus been given this week?

        # Differential Sharpe Ratio (Moody & Saffell 1998)
        self.dsr_A = 0.0  # EMA de returns
        self.dsr_B = 0.0  # EMA de returns^2
        self.dsr_eta = 0.01  # Factor de decaimiento

        # Drawdown tracking
        self.max_equity = initial_balance
        self.drawdown_threshold = 0.02  # 2% drawdown permitido
        self.drawdown_lambda = 10.0  # Multiplicador de penalidad

        # Cooldown: 1 bar H1 despues de cerrar trade (evitar re-entradas impulsivas)
        # Cooldown: 4 bars H1 despues de cerrar trade (e.g. 4 hours)
        self.cooldown_bars = 1  # 1 hour cooldown - aggressive for ~300 trades/year target
        self.bars_since_last_close = self.cooldown_bars  # Empieza sin cooldown

        logger.info(f"GoldEnv initialized: {len(self.df_h1)} H1 bars, "
                   f"{self.n_features} features, "
                   f"M1 data: {'Yes' if self.df_m1 is not None else 'No'}, "
                   f"Entry penalty: {entry_penalty}")

    def _prepare_data(self):
        """Prepare and synchronize data."""
        df = self.df_h1_raw.copy()

        # Ensure datetime index
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')

        # Remove timezone if present
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Sort by time
        df = df.sort_index()

        # Remove weekends and holidays (no trading)
        df = df[df.index.dayofweek < 5]

        self.df_h1 = df

        # Prepare M1 data
        if self.df_m1_raw is not None:
            m1 = self.df_m1_raw.copy()
            if 'time' in m1.columns:
                m1['time'] = pd.to_datetime(m1['time'])
                m1 = m1.set_index('time')
            if m1.index.tz is not None:
                m1.index = m1.index.tz_localize(None)
            m1 = m1.sort_index()
            self.df_m1 = m1

            # Calculate average volume for panic detection
            self.avg_m1_volume = m1['volume'].mean()
        else:
            self.df_m1 = None
            self.avg_m1_volume = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Determine start position
        max_start = len(self.df_h1) - (self.episode_length or 1000) - 1
        max_start = max(200, max_start)  # Need warmup period

        if self.randomize_start and max_start > 200:
            self.start_step = self.np_random.integers(200, max_start)
        else:
            self.start_step = 200

        self.current_step = self.start_step

        # Reset state
        self.balance = self.initial_balance
        self.equity_curve = [self.initial_balance]
        self.prev_equity = self.initial_balance
        self.risk_manager.reset()

        # Reset reward flags
        self.just_opened_trade = False
        self.just_closed_trade = False
        self.last_trade_pnl = 0.0
        self.last_trade_entry_price = 0.0

        # Reset episode statistics
        self.episode_actions = []
        self.episode_rewards = []
        self.consecutive_holds = 0
        self.max_consecutive_holds = 0

        # Reset behavior tracking
        self.steps_since_last_trade = 0
        self.episode_win_rate = 0.0

        # Reset entry conditions
        self.entry_conditions = {
            'rsi': 0.0, 'macd': 0.0, 'atr': 0.0, 'trend': 0.0, 'price': 0.0
        }

        # Reset trade memory
        self.trade_memory = []
        self.last_win_conditions = {'rsi': 0.0, 'macd': 0.0, 'trend': 0.0, 'side': 0.0}
        self.last_loss_conditions = {'rsi': 0.0, 'macd': 0.0, 'trend': 0.0, 'side': 0.0}

        # Reset DSR (Differential Sharpe Ratio)
        self.dsr_A = 0.0
        self.dsr_B = 0.0

        # Reset drawdown tracking
        self.max_equity = self.initial_balance

        # Reset cooldown
        self.bars_since_last_close = self.cooldown_bars  # Empieza sin cooldown

        # Reset weekly bonus tracking
        self.weekly_trades_count = 0
        self.current_week_number = -1
        self.weekly_bonus_given = False

        # Set average volume for risk manager
        self.risk_manager.avg_volume = self.avg_m1_volume

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: 0=HOLD, 1=BUY, 2=SELL

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.just_opened_trade = False
        self.just_closed_trade = False
        self.last_trade_pnl = 0.0

        # Track actions
        self.episode_actions.append(action)

        # Get current H1 bar data (necesario para trend alignment)
        current_bar = self.df_h1.iloc[self.current_step]
        current_time = self.df_h1.index[self.current_step]
        current_price = current_bar['close']
        reward = 0.0

        # COOLDOWN: Forzar HOLD si estamos en periodo de cooldown despues de cerrar trade
        if self.bars_since_last_close < self.cooldown_bars:
            action = self.HOLD  # Forzar HOLD durante cooldown

        # Trend Alignment Removed - Let Agent Learn
        # Check if action is valid given trend is handled by reward function, not override

        # Track consecutive holds
        if action == self.HOLD:
            self.consecutive_holds += 1
            self.max_consecutive_holds = max(self.max_consecutive_holds, self.consecutive_holds)
        else:
            self.consecutive_holds = 0

        # Store previous equity
        self.prev_equity = self._get_current_equity(current_price)

        # Incrementar contador de cooldown
        self.bars_since_last_close += 1

        # 1. Handle action (open position if signaled)
        if action in [self.BUY, self.SELL]:
            # CONDITIONS TO OPEN:
            # 1. No positions (FLAT)
            # OR
            # 2. Existing positions are safe (ALL TRAILING) AND same side
            
            can_open = False
            if self.risk_manager.is_flat:
                can_open = True
            elif self.risk_manager.all_trailing:
                # Check side consistency for pyramiding
                current_side = self.risk_manager.positions[0].side
                if (action == self.BUY and current_side == PositionSide.LONG) or \
                   (action == self.SELL and current_side == PositionSide.SHORT):
                    can_open = True
            
            if can_open:
                side = PositionSide.LONG if action == self.BUY else PositionSide.SHORT
                self.risk_manager.open_position(
                    side=side,
                    entry_price=current_price,
                    entry_time=self.current_step,
                    entry_date=str(current_time),
                    features=current_bar.to_dict()
                )
                self.just_opened_trade = True
                self.last_trade_entry_price = current_price
                self.steps_since_last_trade = 0  # Reset counter

                # GUARDAR condiciones de entrada para que el modelo aprenda patrones
                self.entry_conditions = {
                    'rsi': float(current_bar.get('rsi_norm', 0)) if 'rsi_norm' in current_bar.index else 0.0,
                    'macd': float(current_bar.get('macd_norm', 0)) if 'macd_norm' in current_bar.index else 0.0,
                    'atr': float(current_bar.get('atr_ratio', 1)) if 'atr_ratio' in current_bar.index else 1.0,
                    'trend': float(current_bar.get('trend_dir', 0)) if 'trend_dir' in current_bar.index else 0.0,
                    'price': current_price
                }
            
            # If opposite signal while in position, CLOSE ALL (Reversal/Exit)
            elif not self.risk_manager.is_flat:
                 # Check if signal is opposite to current positions
                 current_side = self.risk_manager.positions[0].side
                 if (action == self.BUY and current_side == PositionSide.SHORT) or \
                    (action == self.SELL and current_side == PositionSide.LONG):
                     
                     # Close all positions
                     closed, reason, pnl = self.risk_manager.close_all_positions(
                         current_price, 
                         "SIGNAL", 
                         exit_date=str(current_time),
                         features=current_bar.to_dict()
                     )
                     if closed:
                         self.just_closed_trade = True
                         self.last_trade_pnl = pnl
                         self.balance += pnl
                         self._update_win_rate()
                         
                         # EFECTUAR DEFENSIVE REWARD CALCULATION HERE
                         # Check if closing was a smart move (saved money)
                         reward += self._calculate_defensive_reward(current_side, current_price)

        else:
            # Increment steps since last trade when HOLDing
            self.steps_since_last_trade += 1

        # 2. Simulate M1 execution (if we have M1 data and position is open)
        if self.df_m1 is not None and not self.risk_manager.is_flat:
            closed, reason, pnl = self._simulate_m1_execution(current_time, current_bar.to_dict())
            if closed:
                self.just_closed_trade = True
                self.last_trade_pnl = pnl
                self.balance += pnl
                self._update_win_rate()
                self.bars_since_last_close = 0  # Reset cooldown
        elif not self.risk_manager.is_flat:
            # No M1 data - use H1 bar for risk management
            closed, reason, pnl = self.risk_manager.update_position(
                current_price,
                current_bar['high'],
                current_bar['low'],
                current_bar.get('volume', 0),
                current_date=str(current_time),
                features=current_bar.to_dict()
            )
            if closed:
                self.just_closed_trade = True
                self.last_trade_pnl = pnl
                self.balance += pnl
                self._update_win_rate()
                self.bars_since_last_close = 0  # Reset cooldown

        # 3. Move to next step
        self.current_step += 1

        # 4. Check termination
        max_step = len(self.df_h1) - 1
        if self.episode_length:
            max_step = min(max_step, self.start_step + self.episode_length)

        terminated = self.current_step >= max_step
        truncated = False

        # Force close at end of episode
        if terminated and not self.risk_manager.is_flat:
            next_price = self.df_h1.iloc[min(self.current_step, len(self.df_h1)-1)]['close']
            next_time = self.df_h1.index[min(self.current_step, len(self.df_h1)-1)]
            
            _, _, pnl = self.risk_manager.close_all_positions(
                next_price, 
                "FORCE_CLOSE", 
                exit_date=str(next_time),
                features=self.df_h1.iloc[min(self.current_step, len(self.df_h1)-1)].to_dict()
            )
            self.balance += pnl
            self.just_closed_trade = True
            self.last_trade_pnl = pnl

        # 5. Update max equity for drawdown tracking
        current_equity = self._get_current_equity(current_price)
        self.max_equity = max(self.max_equity, current_equity)

        # 6. Calculate reward
        reward += self._calculate_reward(current_price)
        self.episode_rewards.append(reward)

        # 6. Update equity curve
        current_equity = self._get_current_equity(current_price)
        self.equity_curve.append(current_equity)

        # 7. Get next observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _calculate_defensive_reward(self, side, close_price) -> float:
        """
        Calculate reward for 'good defensive close'.
        Checks if price went against the position in the next 4 hours.
        """
        # Look ahead 4 hours
        try:
            future_slice = self.df_h1.iloc[self.current_step + 1 : self.current_step + 5]
            if len(future_slice) == 0:
                return 0.0
                
            future_prices = future_slice['close']
            
            if side == PositionSide.LONG:
                # If we were LONG, we saved money if price dropped
                min_future = future_prices.min()
                saved_loss_pct = (close_price - min_future) / close_price
            else:
                # If we were SHORT, we saved money if price rose
                max_future = future_prices.max()
                saved_loss_pct = (max_future - close_price) / close_price
                
            # Reward if we saved a significant loss (>0.1%)
            if saved_loss_pct > 0.001:
                return saved_loss_pct * 20.0  # +2.0 reward for saving 1%
            
            return 0.0
        except:
            return 0.0

    def _simulate_m1_execution(self, h1_time: pd.Timestamp, h1_features: dict = None) -> Tuple[bool, str, float]:
        """
        Simulate execution on M1 data for the current H1 bar.

        Iterates through all M1 bars within the H1 timeframe.
        """
        # Get M1 slice for this hour
        end_time = h1_time + timedelta(hours=1)
        m1_slice = self.df_m1.loc[h1_time:end_time]

        if len(m1_slice) == 0:
            return False, "", 0.0

        # Iterate through M1 bars
        for idx, m1_bar in m1_slice.iterrows():
            closed, reason, pnl = self.risk_manager.update_position(
                current_price=m1_bar['close'],
                current_high=m1_bar['high'],
                current_low=m1_bar['low'],
                current_volume=m1_bar.get('volume', 0),
                current_date=str(idx),
                features=h1_features or {}
            )

            if closed:
                return True, reason, pnl

        return False, "", 0.0

    def _calculate_differential_sharpe(self, return_t: float) -> float:
        """
        Calcula el Differential Sharpe Ratio (Moody & Saffell 1998).

        DSR balancea retorno vs riesgo, incentivando trades consistentes.
        Formula: DSR_t = (B * delta_A - 0.5 * A * delta_B) / (B - A^2)^1.5
        """
        delta_A = return_t - self.dsr_A
        delta_B = return_t**2 - self.dsr_B

        denom = self.dsr_B - self.dsr_A**2
        if denom > 1e-8:  # Evitar division por cero
            dsr = (self.dsr_B * delta_A - 0.5 * self.dsr_A * delta_B) / (denom ** 1.5)
        else:
            dsr = return_t  # Fallback al return simple

        # Actualizar EMAs
        self.dsr_A += self.dsr_eta * delta_A
        self.dsr_B += self.dsr_eta * delta_B

        return dsr

    def _calculate_drawdown_penalty(self, current_price: float) -> float:
        """
        Penaliza drawdown excesivo.

        Cuando equity cae por debajo del maximo historico mas del threshold,
        aplica penalidad proporcional.
        """
        if self.max_equity <= 0:
            return 0.0

        current_equity = self.balance + self.risk_manager.position.unrealized_pnl
        drawdown = (self.max_equity - current_equity) / self.max_equity

        if drawdown > self.drawdown_threshold:
            return -self.drawdown_lambda * (drawdown - self.drawdown_threshold)
        return 0.0

    def _calculate_reward(self, current_price: float) -> float:
        """
        REWARD SYSTEM basado en investigacion + logica del usuario.

        Principios:
        1. HOLD con posicion GANADORA = PREMIAR (dejar correr ganancias)
        2. HOLD con posicion PERDEDORA = PENALIZAR (cortar perdidas)
        3. Al cerrar trade = DSR + bonus/penalty por resultado
        4. DRAWDOWN = Penalidad por secuencias de perdidas
        5. COOLDOWN = 1 bar forzado despues de cerrar (implementado en step)
        """
        reward = 0.0
        # REMOVED legacy position check: position = self.risk_manager.position

        # 1. DRAWDOWN PENALTY - Siempre activo
        reward += self._calculate_drawdown_penalty(current_price)

        # 2. PATIENCE REWARD (New) & HOLD REWARDS
        if self.risk_manager.is_flat:
            # DYNAMIC INACTIVITY THRESHOLD: Adapts to market conditions
            # More pressure during active sessions, more patience during slow periods

            # Get current time and market conditions from dataframe
            step_idx = min(self.current_step, len(self.df_h1) - 1)
            current_bar = self.df_h1.iloc[step_idx]
            current_dt = self.df_h1.index[step_idx]
            current_dow = current_dt.dayofweek  # 0=Monday
            current_hour = current_dt.hour

            # Get session indicators from features
            is_overlap = current_bar.get('is_overlap', 0.0) > 0.5 if 'is_overlap' in current_bar.index else False
            is_london = current_bar.get('is_london', 0.0) > 0.5 if 'is_london' in current_bar.index else False
            is_newyork = current_bar.get('is_newyork', 0.0) > 0.5 if 'is_newyork' in current_bar.index else False
            is_asia = current_bar.get('is_asia', 0.0) > 0.5 if 'is_asia' in current_bar.index else False
            atr_ratio = current_bar.get('atr_ratio', 1.0) if 'atr_ratio' in current_bar.index else 1.0

            # DYNAMIC THRESHOLD based on session
            if is_overlap:
                inactivity_threshold = 4   # London/NY overlap: high activity, more pressure
            elif is_london or is_newyork:
                inactivity_threshold = 6   # Active session: moderate pressure
            elif is_asia:
                inactivity_threshold = 16  # Asia: slow market, more patience
            else:
                inactivity_threshold = 8   # Default

            # Adjust threshold based on volatility
            if atr_ratio > 1.2:
                inactivity_threshold *= 0.75  # High volatility: reduce threshold (more pressure)
            elif atr_ratio < 0.7:
                inactivity_threshold *= 1.5   # Low volatility: increase threshold (more patience)

            # Check if we just passed a weekend (Monday early hours)
            # Don't penalize for weekend gap - market was closed
            is_monday_morning = (current_dow == 0 and current_hour < 8)

            # Only apply inactivity penalty if NOT monday morning
            # (because weekend hours don't count as trading opportunity)
            if self.steps_since_last_trade > inactivity_threshold and not is_monday_morning:
                # Penalty slope: -0.005 per hour
                reward -= 0.005 * (self.steps_since_last_trade - inactivity_threshold)
        
        # 3. REWARD/PENALTY POR HOLD CON POSICION ABIERTA
        else:
            # Use 'state' of the last position as proxy or check all
            # Here we use total unrealized PnL normalized
            unrealized_pnl = self.risk_manager.total_unrealized_pnl
            pnl_pct = unrealized_pnl / self.initial_balance # Approximate
            
            # Use the BEST position's state for rewards (optimistic)
            # This encourages adding to winners
            best_pnl_pct = 0.0
            any_trailing = False
            max_duration = 0
            
            for pos in self.risk_manager.positions:
                if pos.entry_price > 0:
                     if pos.side == PositionSide.LONG:
                         curr_pnl_pct = (current_price - pos.entry_price) / pos.entry_price
                     else:
                         curr_pnl_pct = (pos.entry_price - current_price) / pos.entry_price
                     
                     if curr_pnl_pct > best_pnl_pct:
                         best_pnl_pct = curr_pnl_pct
                
                if pos.state == PositionState.TRAILING:
                    any_trailing = True
                max_duration = max(max_duration, pos.duration_bars)

            if best_pnl_pct > 0:
                # POSICION GANADORA - Premiar HOLD (dejar correr ganancias)
                reward += best_pnl_pct * 5.0
                
                # Bonus extra si esta en estado TRAILING
                if any_trailing:
                    reward += 0.02 + (best_pnl_pct * 10.0)
            else:
                # POSICION PERDEDORA - Penalizar HOLD
                duration_mult = 1 + (max_duration * 0.05)
                reward += pnl_pct * 3.0 * duration_mult

        # 4. ENTRY PENALTY / ACTION BONUS
        if self.just_opened_trade:
            reward -= self.entry_penalty

            # ACTION BONUS: Small reward for taking action (encourages trading)
            # BUT only if market conditions are favorable (quality matters)
            current_bar = self.df_h1.iloc[min(self.current_step, len(self.df_h1) - 1)]

            # Check favorable conditions
            atr_ok = current_bar.get('atr_ratio', 1.0) >= 0.7 if 'atr_ratio' in current_bar.index else True
            not_asia = current_bar.get('is_asia', 0.0) < 0.5 if 'is_asia' in current_bar.index else True
            has_trend = abs(current_bar.get('trend_dir', 0)) > 0 if 'trend_dir' in current_bar.index else True

            # Bonus tiers based on conditions met
            conditions_met = sum([atr_ok, not_asia, has_trend])
            if conditions_met >= 3:
                reward += 0.1  # Good conditions: full bonus
            elif conditions_met >= 2:
                reward += 0.05  # Decent conditions: partial bonus
            # No bonus if conditions are poor (quality control)

            # WEEKLY TARGET BONUS: Reward given ONCE when reaching 5 trades in a week
            # Get current week number (year * 52 + week_of_year)
            step_idx = min(self.current_step, len(self.df_h1) - 1)
            current_dt = self.df_h1.index[step_idx]
            week_number = current_dt.year * 52 + current_dt.isocalendar()[1]

            # Check if we're in a new week
            if week_number != self.current_week_number:
                # New week started - reset counters
                self.current_week_number = week_number
                self.weekly_trades_count = 0
                self.weekly_bonus_given = False

            # Increment weekly trade count
            self.weekly_trades_count += 1

            # Give bonus ONCE when reaching 5 trades this week
            if self.weekly_trades_count >= 5 and not self.weekly_bonus_given:
                reward += 1.0  # Bonus for reaching weekly target
                self.weekly_bonus_given = True  # Don't give again this week

        # 5. REWARD AL CERRAR TRADE - DSR + bonus/penalty

        # 3. REWARD AL CERRAR TRADE - DSR + bonus/penalty
        if self.just_closed_trade:
            # Calcular return del trade
            if self.last_trade_entry_price > 0:
                pnl_pct = self.last_trade_pnl / self.last_trade_entry_price
            else:
                pnl_pct = 0.0

            # 3. DIFFERENTIAL SHARPE RATIO
            # Balancea retorno vs riesgo, incentivando trades consistentes
            dsr = self._calculate_differential_sharpe(pnl_pct)

            # Escalar DSR para que tenga impacto significativo
            reward += dsr * 50.0

            # 4. PROGRESSIVE JACKPOT REWARDS (Exponential scaling)
            # Incentivize catching major trend moves (>1%, >2%, >3%)
            if pnl_pct >= 0.03:    # >= 3.0% (JACKPOT)
                reward += 100.0    # Massive reward for life-changing trades
            elif pnl_pct >= 0.02:  # >= 2.0% (MEGA WIN)
                reward += 60.0
            elif pnl_pct >= 0.01:  # >= 1.0% (BIG WIN)
                reward += 25.0
            elif pnl_pct >= 0.005: # >= 0.5% (GOOD WIN)
                reward += 10.0
            elif pnl_pct >= 0.0025: # >= 0.25% (BASE TARGET)
                reward += 5.0
            elif pnl_pct > 0:      # Small profit
                reward += 1.0
            elif pnl_pct < -0.003:  # < -0.3% (bad loss)
                reward -= 5.0
            elif pnl_pct < -0.001:  # < -0.1%
                reward -= 2.0
            elif pnl_pct < 0:  # Cualquier perdida
                reward -= 1.0

            # 5. Bonus por cerrar en trailing (capturar tendencia)
            last_trade = self.risk_manager.trade_history[-1] if self.risk_manager.trade_history else None
            if last_trade:
                final_state = last_trade.get('final_state', '')
                if final_state == 'TRAILING' and pnl_pct > 0:
                    reward += 3.0  # Bonus por capturar tendencia

            # 6. CAPTURE EFFICIENCY - Que tan bien capturo el movimiento
            if last_trade:
                entry_price = last_trade['entry_price']
                exit_price = last_trade['exit_price']
                max_favorable = last_trade.get('max_favorable', exit_price)
                max_adverse = last_trade.get('max_adverse', exit_price)
                is_long = last_trade['side'] == 'LONG'

                if pnl_pct > 0:  # TRADE GANADOR
                    # Calcular que % del movimiento maximo capturo
                    if is_long:
                        max_possible_gain = max_favorable - entry_price
                        actual_gain = exit_price - entry_price
                    else:  # SHORT
                        max_possible_gain = entry_price - max_favorable
                        actual_gain = entry_price - exit_price

                    if max_possible_gain > 0:
                        capture_ratio = actual_gain / max_possible_gain
                        # Premiar si capturo >70% del movimiento
                        if capture_ratio >= 0.8:
                            reward += 2.0  # Excelente captura
                        elif capture_ratio >= 0.6:
                            reward += 1.0  # Buena captura
                        elif capture_ratio < 0.4:
                            reward -= 1.0  # Dejo mucho en la mesa

                else:  # TRADE PERDEDOR
                    # Evaluar si corto perdidas bien (salio antes del peor momento)
                    if is_long:
                        worst_loss = entry_price - max_adverse  # Cuanto pudo perder
                        actual_loss = entry_price - exit_price  # Cuanto perdio
                    else:  # SHORT
                        worst_loss = max_adverse - entry_price
                        actual_loss = exit_price - entry_price

                    if worst_loss > 0:
                        loss_ratio = actual_loss / worst_loss  # 1.0 = peor momento, <1 = corto antes
                        # Premiar si salio antes del peor momento
                        if loss_ratio < 0.5:
                            reward += 2.0  # Excelente corte de perdidas
                        elif loss_ratio < 0.8:
                            reward += 1.0  # Buen corte
                        elif loss_ratio >= 0.95:
                            reward -= 1.0  # Salio en el peor momento

        return float(reward)

    def _update_win_rate(self):
        """Update episode win rate and save trade conditions for learning."""
        trades = self.risk_manager.trade_history
        if trades:
            wins = sum(1 for t in trades if t['pnl'] > 0)
            self.episode_win_rate = wins / len(trades)

            # Guardar condiciones del ultimo trade para que el modelo aprenda
            last_trade = trades[-1]
            is_win = last_trade['pnl'] > 0
            side_binary = 1.0 if last_trade['side'] == 'LONG' else 0.0

            # Normalizar condiciones a 0-1
            # RSI norm esta en [-1,1], convertir a [0,1]
            rsi_01 = (self.entry_conditions['rsi'] + 1) / 2
            # MACD norm esta en [-1,1], convertir a [0,1]
            macd_01 = (self.entry_conditions['macd'] + 1) / 2
            # Trend dir esta en [-1,1], convertir a [0,1] (1=bullish, 0=bearish)
            trend_01 = (self.entry_conditions['trend'] + 1) / 2

            trade_record = {
                'rsi': rsi_01,
                'macd': macd_01,
                'trend': trend_01,
                'side': side_binary,
                'is_win': 1.0 if is_win else 0.0
            }
            self.trade_memory.append(trade_record)

            # Actualizar condiciones del ultimo win/loss
            if is_win:
                self.last_win_conditions = {
                    'rsi': rsi_01,
                    'macd': macd_01,
                    'trend': trend_01,
                    'side': side_binary
                }
            else:
                self.last_loss_conditions = {
                    'rsi': rsi_01,
                    'macd': macd_01,
                    'trend': trend_01,
                    'side': side_binary
                }

    def _get_current_equity(self, current_price: float) -> float:
        """Get current equity including unrealized PnL."""
        unrealized = self.risk_manager.total_unrealized_pnl
        return self.balance + unrealized

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        if self.current_step >= len(self.df_h1):
            self.current_step = len(self.df_h1) - 1

        current_bar = self.df_h1.iloc[self.current_step]

        # Get feature values
        features = []
        for col in self.feature_columns:
            if col in current_bar.index:
                features.append(float(current_bar[col]))
            else:
                features.append(0.0)

        # Add position info
        # Aggregate info from multiple positions
        in_position = 0.0 if self.risk_manager.is_flat else 1.0
        
        # Side: 0=None, 0.5=Long, 1.0=Short (approx)
        position_side = 0.0
        if not self.risk_manager.is_flat:
            side = self.risk_manager.positions[0].side
            position_side = 0.5 if side == PositionSide.LONG else 1.0
            
        unrealized_pnl = self.risk_manager.total_unrealized_pnl / self.initial_balance
        
        # Duration: use max duration
        max_dur = 0
        if not self.risk_manager.is_flat:
             max_dur = max(p.duration_bars for p in self.risk_manager.positions)
        duration = min(max_dur / 100.0, 1.0)

        # Add market context
        # Recent volatility (from atr_ratio if available)
        recent_vol = current_bar.get('atr_ratio', 1.0) if 'atr_ratio' in current_bar.index else 1.0
        # Trend strength (from adx_norm if available)
        trend_strength = current_bar.get('adx_norm', 0.5) if 'adx_norm' in current_bar.index else 0.5

        position_info = [in_position, position_side, unrealized_pnl, duration, recent_vol, trend_strength]

        # Add behavior info - CRITICO para que el modelo aprenda frecuencia
        n_trades = len(self.risk_manager.trade_history)

        # 1. Trade ratio vs target (0 = no trades, 1 = at target, >1 = over target)
        trade_ratio = n_trades / max(1, self.target_trades_per_week)

        # 2. Trade saturation (0-1, 1 means at or over 20 trades)
        trade_saturation = min(n_trades / 20.0, 1.0)

        # 3. Steps since last trade (normalized, cap at 100 steps)
        steps_since_trade = min(self.steps_since_last_trade / 100.0, 1.0)

        # 4. Current episode win rate (0-1)
        win_rate = self.episode_win_rate

        # 5. ATR filter signal (1 = good to trade, 0 = dead market)
        atr_ratio = current_bar.get('atr_ratio', 1.0) if 'atr_ratio' in current_bar.index else 1.0
        atr_filter = 1.0 if atr_ratio >= 0.8 else 0.0

        behavior_info = [trade_ratio, trade_saturation, steps_since_trade, win_rate, atr_filter]

        # Entry conditions - para que el modelo aprenda patrones ganadores
        # Use first position for entry conditions reference
        if not self.risk_manager.is_flat and self.entry_conditions['price'] > 0:
            entry_price = self.entry_conditions['price']
            current_price = current_bar['close']
            price_vs_entry = (current_price - entry_price) / entry_price  # % change since entry
            # Normalizar a 0-1
            entry_rsi_01 = (self.entry_conditions['rsi'] + 1) / 2
            entry_macd_01 = (self.entry_conditions['macd'] + 1) / 2
            entry_trend_01 = (self.entry_conditions['trend'] + 1) / 2
            entry_info = [
                entry_rsi_01,
                entry_macd_01,
                min(self.entry_conditions['atr'] / 2, 1.0),  # ATR normalizado
                entry_trend_01,
                max(min(price_vs_entry * 100, 1.0), -1.0)  # Price change clipped
            ]
        else:
            entry_info = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Trade memory - condiciones de ultimo win vs loss para que el modelo aprenda
        # Last win conditions (normalized 0-1)
        last_win = [
            self.last_win_conditions['rsi'],
            self.last_win_conditions['macd'],
            self.last_win_conditions['trend'],
            self.last_win_conditions['side']
        ]
        # Last loss conditions (normalized 0-1)
        last_loss = [
            self.last_loss_conditions['rsi'],
            self.last_loss_conditions['macd'],
            self.last_loss_conditions['trend'],
            self.last_loss_conditions['side']
        ]

        # Similaridad de condiciones actuales con ultimo win
        # Si las condiciones actuales son similares al ultimo win = buena senal
        current_rsi_01 = (current_bar.get('rsi_norm', 0) + 1) / 2 if 'rsi_norm' in current_bar.index else 0.5
        current_macd_01 = (current_bar.get('macd_norm', 0) + 1) / 2 if 'macd_norm' in current_bar.index else 0.5
        current_trend_01 = (current_bar.get('trend_dir', 0) + 1) / 2 if 'trend_dir' in current_bar.index else 0.5

        if self.last_win_conditions['rsi'] > 0:  # Si hay un win registrado
            # Calcular similaridad (1 = muy similar, 0 = muy diferente)
            rsi_diff = abs(current_rsi_01 - self.last_win_conditions['rsi'])
            macd_diff = abs(current_macd_01 - self.last_win_conditions['macd'])
            trend_diff = abs(current_trend_01 - self.last_win_conditions['trend'])
            similar_to_win = 1.0 - (rsi_diff + macd_diff + trend_diff) / 3
        else:
            similar_to_win = 0.5  # Neutral si no hay wins aun

        trade_memory_info = last_win + last_loss + [similar_to_win]

        obs = np.array(features + position_info + behavior_info + entry_info + trade_memory_info, dtype=np.float32)

        # Clip extreme values
        obs = np.clip(obs, -10.0, 10.0)

        # Replace NaN with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get detailed info dictionary for diagnostics."""
        position = self.risk_manager.position
        current_price = self.df_h1.iloc[min(self.current_step, len(self.df_h1)-1)]['close']

        # Calculate trade frequency
        episode_steps = len(self.episode_actions)
        total_trades = len(self.risk_manager.trade_history)
        trades_per_hour = total_trades / max(episode_steps, 1)
        trades_per_week_estimate = trades_per_hour * self.hours_per_week

        # Action distribution
        action_counts = {
            'hold': self.episode_actions.count(0),
            'buy': self.episode_actions.count(1),
            'sell': self.episode_actions.count(2)
        }

        return {
            'step': self.current_step,
            'episode_step': episode_steps,
            'balance': self.balance,
            'equity': self._get_current_equity(current_price),
            'position_state': position.state.name,
            'position_side': position.side.name,
            'unrealized_pnl': position.unrealized_pnl,
            'trade_count': total_trades,
            'trades_per_week_est': trades_per_week_estimate,
            'action_distribution': action_counts,
            'max_consecutive_holds': self.max_consecutive_holds,
            'trade_stats': self.risk_manager.get_trade_stats(),
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0
        }

    def render(self):
        """Render current state."""
        if self.render_mode == 'human':
            info = self._get_info()
            print(f"Step {info['step']}: Balance=${info['balance']:.2f}, "
                  f"Equity=${info['equity']:.2f}, "
                  f"Position={info['position_state']}, "
                  f"Trades={info['trade_count']}, "
                  f"Est.Trades/Week={info['trades_per_week_est']:.1f}")

    def close(self):
        """Clean up resources."""
        pass

    def get_episode_stats(self) -> Dict[str, float]:
        """Get comprehensive statistics for the current episode."""
        equity = np.array(self.equity_curve)
        if len(equity) > 1:
            # Protect against division by zero
            equity_shifted = np.where(equity[:-1] == 0, 1, equity[:-1])
            returns = np.diff(equity) / equity_shifted
            # Replace any inf/nan with 0
            returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            returns = np.array([0.0])

        # Trade statistics
        trade_stats = self.risk_manager.get_trade_stats()

        # Action statistics
        episode_steps = len(self.episode_actions)
        hold_pct = self.episode_actions.count(0) / max(episode_steps, 1) * 100
        buy_pct = self.episode_actions.count(1) / max(episode_steps, 1) * 100
        sell_pct = self.episode_actions.count(2) / max(episode_steps, 1) * 100

        # Trade frequency
        total_trades = len(self.risk_manager.trade_history)
        trades_per_hour = total_trades / max(episode_steps, 1)
        trades_per_week_est = trades_per_hour * self.hours_per_week

        # Calculate total return with protection
        if len(equity) > 0 and equity[0] != 0:
            total_return = (equity[-1] / equity[0]) - 1
        else:
            total_return = 0.0
        total_return = 0.0 if np.isnan(total_return) or np.isinf(total_return) else total_return

        stats = {
            # Returns
            'total_return': total_return,
            'sharpe_ratio': self._calculate_sharpe(returns),
            'sortino_ratio': self._calculate_sortino(returns),
            'max_drawdown': self._calculate_max_drawdown(equity),

            # Trades
            'total_trades': total_trades,
            'trades_per_week_est': trades_per_week_est,
            'win_rate': trade_stats['win_rate'],
            'profit_factor': trade_stats['profit_factor'],
            'avg_trade_pnl': trade_stats['avg_pnl'],
            'total_gains': trade_stats.get('total_gains', 0),
            'total_losses': trade_stats.get('total_losses', 0),

            # Actions
            'hold_pct': hold_pct,
            'buy_pct': buy_pct,
            'sell_pct': sell_pct,
            'max_consecutive_holds': self.max_consecutive_holds,

            # Rewards
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'total_reward': sum(self.episode_rewards) if self.episode_rewards else 0,
        }

        return stats

    def _calculate_sharpe(self, returns: np.ndarray, risk_free: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - risk_free
        if np.std(returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252 * 24)  # Annualized

    def _calculate_sortino(self, returns: np.ndarray, risk_free: float = 0.0) -> float:
        """Calculate Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - risk_free
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 99.99 if np.mean(excess_returns) > 0 else 0.0
        result = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252 * 24)
        return result if not np.isnan(result) and not np.isinf(result) else 0.0

    def _calculate_max_drawdown(self, equity: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(equity) < 2:
            return 0.0
        peak = np.maximum.accumulate(equity)
        # Avoid division by zero
        peak = np.where(peak == 0, 1, peak)
        drawdown = (peak - equity) / peak
        result = float(np.max(drawdown))
        return result if not np.isnan(result) and not np.isinf(result) else 0.0


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import os

    print("=" * 70)
    print("GOLD ENVIRONMENT TEST (SNIPER STRATEGY)")
    print("=" * 70)

    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    h1_path = os.path.join(data_dir, 'xauusd_h1.csv')
    m1_path = os.path.join(data_dir, 'xauusd_m1.csv')

    df_h1 = pd.read_csv(h1_path)
    print(f"Loaded H1: {len(df_h1)} rows")

    # Load M1 (subset for testing)
    if os.path.exists(m1_path):
        df_m1 = pd.read_csv(m1_path, nrows=100000)
        print(f"Loaded M1: {len(df_m1)} rows")
    else:
        df_m1 = None
        print("M1 data not found")

    # Add features to H1
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from features.h1_features import add_features_h1, get_feature_columns

    df_h1 = add_features_h1(df_h1)
    feature_cols = get_feature_columns()

    # Create environment with sniper settings
    env = GoldEnv(
        df_h1=df_h1,
        df_m1=df_m1,
        feature_columns=feature_cols,
        episode_length=1000,
        randomize_start=False,
        render_mode='human',
        entry_penalty=0.01,  # High penalty for sniper
    )

    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Run test episode
    print("\nRunning test episode...")
    obs, info = env.reset()
    print(f"Initial obs shape: {obs.shape}")

    total_reward = 0
    done = False
    step = 0

    while not done and step < 100:
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        step += 1

        if step % 20 == 0:
            env.render()

    print(f"\nEpisode finished after {step} steps")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Episode stats: {env.get_episode_stats()}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
