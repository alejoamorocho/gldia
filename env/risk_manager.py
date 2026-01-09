"""
Risk Manager for XAU-SNIPER
============================

Implements the "Sniper" risk management system:
- 0.3% profit target
- Trailing stop with trigger at 0.3%
- State machine: RISK -> TRIGGER -> TRAILING

This logic is handled by the environment, not the AI.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PositionState(Enum):
    """State machine for position management."""
    FLAT = 0        # No position
    RISK = 1        # Position open, in initial risk phase
    TRIGGERED = 2   # TP hit, stop moved to breakeven+
    TRAILING = 3    # Price advancing, trailing stop active


class PositionSide(Enum):
    """Position direction."""
    NONE = 0
    LONG = 1
    SHORT = 2


@dataclass
class PositionInfo:
    """Current position information."""
    side: PositionSide = PositionSide.NONE
    entry_price: float = 0.0
    entry_time: int = 0
    size: float = 1.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    state: PositionState = PositionState.FLAT
    max_price: float = 0.0  # For trailing (highest price for long)
    min_price: float = 0.0  # For trailing (lowest price for short)
    unrealized_pnl: float = 0.0
    duration_bars: int = 0
    # Enhanced context
    entry_date: str = ""
    entry_features: dict = None


@dataclass
class RiskConfig:
    """Risk management configuration."""
    # Target profit (0.3% = 0.003) - TRIGGER for trailing
    tp_target: float = 0.003

    # Initial stop loss (0.3% = 0.003) - R:R 1:1
    # Con trailing, trades ganadores pueden ser 1:5 o mas
    sl_initial: float = 0.003

    # Trailing distance after trigger (0.15%)
    # Mas ajustado para capturar ganancias del trailing
    trailing_distance: float = 0.0015

    # Breakeven level after trigger (0.15%)
    # Asegura ganancia minima despues del trigger
    breakeven_level: float = 0.0015

    # =========================================================================
    # COSTOS DE TRADING REALISTAS (basados en broker real)
    # =========================================================================
    # Spread base: 3 pips = 0.0003 (como porcentaje del precio)
    spread: float = 0.0003

    # Slippage base: 1-3 pips = 0.0001-0.0003
    slippage: float = 0.0002  # 2 pips promedio

    # Commission: 0.5 pips = 0.00005
    commission: float = 0.00005

    # Spread maximo para operar (filtro)
    max_spread: float = 0.0005  # 5 pips - rechazar si spread > esto

    # Multiplicadores de volatilidad
    spread_vol_multiplier: float = 2.0   # En alta volatilidad
    slippage_vol_multiplier: float = 3.0  # En alta volatilidad

    # Para backtest conservador (50% peor que historico)
    backtest_spread_mult: float = 1.5

    # Use ATR-based stops instead of fixed percentage
    use_atr_stops: bool = False
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 2.0

    # Panic exit on extreme volume
    panic_volume_multiplier: float = 10.0

    def get_total_cost(self, is_high_volatility: bool = False) -> float:
        """
        Calcula el costo total por trade como porcentaje.

        En condiciones normales: ~3.5 pips (0.00035)
        En alta volatilidad: hasta ~10+ pips (0.001+)
        """
        spread = self.spread * (self.spread_vol_multiplier if is_high_volatility else 1.0)
        slip = self.slippage * (self.slippage_vol_multiplier if is_high_volatility else 1.0)
        return spread + slip + self.commission


class RiskManager:
    """
    Manages position risk for the Sniper strategy.

    Features:
    - Fixed TP at 0.3%, SL at 0.15% (2:1 ratio)
    - Trailing stop activates after TP is hit
    - Panic exit on extreme volume spikes
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        """Initialize risk manager."""
        self.config = config or RiskConfig()
        self.positions = []  # List of PositionInfo
        self.trade_history = []
        self.avg_volume = 0.0

        logger.info(f"RiskManager initialized: TP={self.config.tp_target*100:.2f}%, "
                   f"SL={self.config.sl_initial*100:.2f}%")

    @property
    def position(self) -> PositionInfo:
        """
        Legacy property for backward compatibility / single-position logic.
        Returns the last opened position or a flat position.
        """
        if not self.positions:
            return PositionInfo()
        return self.positions[-1]
    
    @property
    def total_unrealized_pnl(self) -> float:
        """Sum of unrealized PnL of all positions."""
        return sum(p.unrealized_pnl for p in self.positions)
        
    @property
    def is_flat(self) -> bool:
        return len(self.positions) == 0

    @property
    def all_trailing(self) -> bool:
        """True if we have positions and ALL are in TRAILING state."""
        if not self.positions:
            return False
        return all(p.state == PositionState.TRAILING for p in self.positions)

    def reset(self):
        """Reset position state."""
        self.positions = []
        self.trade_history = []

    def open_position(
        self,
        side: PositionSide,
        entry_price: float,
        entry_time: int,
        size: float = 1.0,
        atr: Optional[float] = None,
        is_high_volatility: bool = False,
        entry_date: str = "",
        features: dict = None
    ) -> PositionInfo:
        """
        Open a new position.

        Args:
            side: LONG or SHORT
            entry_price: Entry price
            entry_time: Bar index at entry
            size: Position size
            atr: Current ATR for dynamic stops
            is_high_volatility: If True, apply higher spread/slippage

        Returns:
            Updated position info
        """
        # Restriction: Only allow same-side pyramiding
        if self.positions:
            if self.positions[0].side != side:
                logger.warning("Attempted to open opposite side position (Hedging not supported)")
                return self.position
            
            # OPTIONAL: Max positions limit (could be in config)
            if len(self.positions) >= 3:
                return self.position

        # Calculate stops

        # Calculate stops
        if self.config.use_atr_stops and atr is not None:
            sl_distance = atr * self.config.atr_sl_multiplier
            tp_distance = atr * self.config.atr_tp_multiplier
        else:
            sl_distance = entry_price * self.config.sl_initial
            tp_distance = entry_price * self.config.tp_target

        # Set SL/TP based on direction
        if side == PositionSide.LONG:
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # SHORT
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        # Apply costs as percentage of entry price
        # Costo total: spread + slippage + commission (~3.5 pips normal, ~10+ alta vol)
        total_cost_pct = self.config.get_total_cost(is_high_volatility)
        entry_cost = entry_price * total_cost_pct * size

        new_pos = PositionInfo(
            side=side,
            entry_price=entry_price,
            entry_time=entry_time,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            state=PositionState.RISK,
            max_price=entry_price,
            min_price=entry_price,
            unrealized_pnl=-entry_cost,  # Start with costs
            duration_bars=0,
            entry_date=entry_date,
            entry_features=features or {}
        )
        
        self.positions.append(new_pos)

        logger.debug(f"Opened {side.name} at {entry_price:.2f}, "
                    f"SL={stop_loss:.2f}, TP={take_profit:.2f}, "
                    f"Cost={entry_cost:.2f} ({total_cost_pct*100:.3f}%)")
        
        return new_pos

    def update_position(
        self,
        current_price: float,
        current_high: float,
        current_low: float,
        current_volume: float = 0.0,
        current_date: str = "",
        features: dict = None
    ) -> Tuple[bool, str, float]:
        """
        Update position with new price data.

        Called for each M1 bar within the H1 step.

        Args:
            current_price: Current close price
            current_high: Bar high
            current_low: Bar low
            current_volume: Bar volume

        Returns:
            Tuple of (closed, reason, realized_pnl)
        """
        """
        Update all positions with new price data.
        Returns: (closed_any, reason, total_realized_pnl)
        """
        if not self.positions:
            return False, "", 0.0

        total_pnl = 0.0
        closed_any = False
        last_reason = ""
        
        # Iterate over copy to allow removal
        for pos in self.positions[:]:
            pos.duration_bars += 1

            # Update max/min prices for trailing
            pos.max_price = max(pos.max_price, current_high)
            pos.min_price = min(pos.min_price, current_low)

            # Calculate unrealized PnL
            if pos.side == PositionSide.LONG:
                pos.unrealized_pnl = (
                    (current_price - pos.entry_price) * pos.size
                    - self.config.commission - self.config.slippage
                )
            else:
                pos.unrealized_pnl = (
                    (pos.entry_price - current_price) * pos.size
                    - self.config.commission - self.config.slippage
                )

            # Check panic exit (extreme volume)
            if current_volume > self.avg_volume * self.config.panic_volume_multiplier:
                _, reason, pnl = self._close_specific_position(pos, current_price, "PANIC")
                total_pnl += pnl
                closed_any = True
                last_reason = "PANIC"
                continue

            # Check stop loss
            if self._check_stop_loss(pos, current_low, current_high):
                exit_price = pos.stop_loss
                _, reason, pnl = self._close_specific_position(pos, exit_price, "STOP_LOSS")
                total_pnl += pnl
                closed_any = True
                last_reason = "STOP_LOSS"
                continue

            # Check take profit and update state
            if self._check_take_profit(pos, current_high, current_low):
                self._handle_tp_trigger(pos)

            # Update trailing stop if in trailing state
            if pos.state == PositionState.TRAILING:
                self._update_trailing_stop(pos)

        return closed_any, last_reason, total_pnl

    def _check_stop_loss(self, pos: PositionInfo, bar_low: float, bar_high: float) -> bool:
        """Check if stop loss was hit."""
        if pos.side == PositionSide.LONG:
            return bar_low <= pos.stop_loss
        else:  # SHORT
            return bar_high >= pos.stop_loss

    def _check_take_profit(self, pos: PositionInfo, bar_high: float, bar_low: float) -> bool:
        """Check if take profit was hit."""
        if pos.state != PositionState.RISK:
            return False

        if pos.side == PositionSide.LONG:
            return bar_high >= pos.take_profit
        else:  # SHORT
            return bar_low <= pos.take_profit

    def _handle_tp_trigger(self, pos: PositionInfo):
        """Handle take profit trigger - move to TRIGGERED state."""
        if pos.state != PositionState.RISK:
            return

        # Move stop to breakeven + buffer
        breakeven_buffer = pos.entry_price * self.config.breakeven_level

        if pos.side == PositionSide.LONG:
            pos.stop_loss = pos.entry_price + breakeven_buffer
        else:  # SHORT
            pos.stop_loss = pos.entry_price - breakeven_buffer

        pos.state = PositionState.TRIGGERED

        # Check if we should start trailing
        self._update_trailing_stop(pos)

    def _update_trailing_stop(self, pos: PositionInfo):
        """Update trailing stop based on price movement."""
        trailing_dist = pos.entry_price * self.config.trailing_distance

        if pos.side == PositionSide.LONG:
            new_sl = pos.max_price - trailing_dist
            if new_sl > pos.stop_loss:
                pos.stop_loss = new_sl
                pos.state = PositionState.TRAILING
        else:  # SHORT
            new_sl = pos.min_price + trailing_dist
            if new_sl < pos.stop_loss:
                pos.stop_loss = new_sl
                pos.state = PositionState.TRAILING

    def close_all_positions(self, exit_price: float, reason: str, is_high_volatility: bool = False, exit_date: str = "", features: dict = None) -> Tuple[bool, str, float]:
        """Close ALL positions."""
        if not self.positions:
            return False, "", 0.0

        total_pnl = 0.0
        
        # Close all
        for pos in self.positions[:]:
            _, _, pnl = self._close_specific_position(pos, exit_price, reason, is_high_volatility, exit_date, features)
            total_pnl += pnl

        return True, reason, total_pnl

    def _close_specific_position(
        self,
        pos: PositionInfo,
        exit_price: float,
        reason: str,
        is_high_volatility: bool = False,
        exit_date: str = "",
        features: dict = None
    ) -> Tuple[bool, str, float]:
        """Close a specific position and record trade."""
        entry_price = pos.entry_price
        size = pos.size

        # Calculate raw PnL
        if pos.side == PositionSide.LONG:
            raw_pnl = (exit_price - entry_price) * size
        else:
            raw_pnl = (entry_price - exit_price) * size

        # Subtract exit costs
        exit_cost_pct = self.config.get_total_cost(is_high_volatility)
        exit_cost = exit_price * exit_cost_pct * size
        pnl = raw_pnl - exit_cost
        
        # Calculate PnL percentage
        pnl_pct = pnl / entry_price if entry_price > 0 else 0

        # Record trade
        trade_record = {
            'side': pos.side.name,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'raw_pnl': raw_pnl,
            'total_costs': exit_cost + abs(pos.unrealized_pnl if pos.unrealized_pnl < 0 else 0),
            'duration_bars': pos.duration_bars,
            'max_favorable': pos.max_price if pos.side == PositionSide.LONG else pos.min_price,
            'max_adverse': pos.min_price if pos.side == PositionSide.LONG else pos.max_price,
            'exit_reason': reason,
            'final_state': pos.state.name,
            'entry_date': pos.entry_date,
            'exit_date': exit_date,
            'entry_features': pos.entry_features,
            'exit_features': features or {}
        }
        self.trade_history.append(trade_record)

        # Remove from list
        if pos in self.positions:
            self.positions.remove(pos)

        return True, reason, pnl

    def force_close(self, current_price: float) -> Tuple[bool, str, float]:
        """Force close position (e.g., end of episode)."""
        if not self.positions:
            return False, "", 0.0
        return self.close_all_positions(current_price, "FORCE_CLOSE")

    def get_position_pnl_pct(self, current_price: float) -> float:
        """Get best current position PnL as percentage (for rewards)."""
        if not self.positions:
            return 0.0

        # Return the PnL of the BEST position (to encourage adding to winners)
        pnls = []
        for pos in self.positions:
            if pos.side == PositionSide.LONG:
                pnls.append((current_price - pos.entry_price) / pos.entry_price)
            else:
                pnls.append((pos.entry_price - current_price) / pos.entry_price)
        
        return max(pnls)

    def get_trade_stats(self) -> dict:
        """Get statistics from trade history."""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'total_pnl': 0.0,
                'profit_factor': 0.0,
                'avg_duration': 0.0
            }

        pnls = [t['pnl'] for t in self.trade_history]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0

        return {
            'total_trades': len(self.trade_history),
            'win_rate': len(wins) / len(self.trade_history) if self.trade_history else 0.0,
            'avg_pnl': np.mean(pnls),
            'total_pnl': sum(pnls),
            'total_gains': total_wins,  # Exposed for reporting
            'total_losses': total_losses, # Exposed for reporting
            'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf'),
            'avg_duration': np.mean([t['duration_bars'] for t in self.trade_history])
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RISK MANAGER TEST")
    print("=" * 70)

    # Create risk manager
    config = RiskConfig(tp_target=0.003, sl_initial=0.0015)
    rm = RiskManager(config)

    # Simulate a trade
    entry_price = 2000.0
    print(f"\nOpening LONG at {entry_price}")

    rm.open_position(PositionSide.LONG, entry_price, 0)
    print(f"Position: {rm.position}")
    print(f"SL: {rm.position.stop_loss:.2f} ({(rm.position.stop_loss - entry_price)/entry_price*100:.2f}%)")
    print(f"TP: {rm.position.take_profit:.2f} ({(rm.position.take_profit - entry_price)/entry_price*100:.2f}%)")

    # Simulate price movement
    print("\nSimulating price movement...")
    prices = [
        (2001, 2002, 2000),  # Slight up
        (2003, 2005, 2002),  # Up more
        (2005, 2006, 2004),  # Hit TP level
        (2007, 2008, 2006),  # Continue up
        (2005, 2007, 2004),  # Pullback
    ]

    for i, (close, high, low) in enumerate(prices):
        closed, reason, pnl = rm.update_position(close, high, low)
        print(f"Bar {i+1}: close={close}, state={rm.position.state.name}, "
              f"SL={rm.position.stop_loss:.2f}, PnL={rm.position.unrealized_pnl:.2f}")

        if closed:
            print(f"  -> CLOSED: {reason}, PnL={pnl:.2f}")
            break

    # Print trade stats
    print(f"\nTrade stats: {rm.get_trade_stats()}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
