"""
Mean Reversion Features for Gold Trading
=========================================

Analisis mostro que el oro tiene:
- Autocorrelacion negativa (reversion a la media)
- Senales tecnicas funcionan para 30min, no para 1h+
- Necesitamos features que capturen CUANDO revertir

Features que funcionan para mean reversion:
1. Oversold/Overbought extremos (entrada en extremos)
2. Distancia de precio a medias (mean reversion)
3. Volatilidad expandida (oportunidad de reversion)
4. Niveles de soporte/resistencia cercanos
"""

import numpy as np
import pandas as pd
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def compute_mean_reversion_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """
    Compute features optimized for mean reversion trading.

    Returns DataFrame with ~30 mean reversion features.
    """
    features = pd.DataFrame(index=df.index)
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df.get('volume', pd.Series(0, index=df.index))

    # =========================================================================
    # 1. DISTANCIA A MEDIAS (Mean Reversion Core)
    # =========================================================================

    # Diferentes MAs
    for period in [10, 20, 50, 100]:
        ma = close.rolling(period).mean()

        # Distancia porcentual al MA
        dist = (close - ma) / ma
        features[f'{prefix}dist_ma{period}'] = dist

        # Z-score de la distancia (normalizado)
        dist_std = dist.rolling(50).std()
        features[f'{prefix}zscore_ma{period}'] = dist / dist_std.replace(0, np.nan)

        # Senal: Muy lejos del MA (oportunidad reversion)
        features[f'{prefix}extreme_above_ma{period}'] = (dist > dist.rolling(100).quantile(0.95)).astype(float)
        features[f'{prefix}extreme_below_ma{period}'] = (dist < dist.rolling(100).quantile(0.05)).astype(float)

    # =========================================================================
    # 2. RSI EXTREMOS (Classic Mean Reversion)
    # =========================================================================

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    for period in [7, 14, 21]:
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # RSI normalizado
        features[f'{prefix}rsi{period}'] = (rsi - 50) / 50  # -1 a 1

        # Extremos de RSI (principales senales)
        features[f'{prefix}rsi{period}_oversold'] = (rsi < 25).astype(float)
        features[f'{prefix}rsi{period}_overbought'] = (rsi > 75).astype(float)
        features[f'{prefix}rsi{period}_extreme_oversold'] = (rsi < 15).astype(float)
        features[f'{prefix}rsi{period}_extreme_overbought'] = (rsi > 85).astype(float)

    # =========================================================================
    # 3. BOLLINGER BANDS (Volatility Mean Reversion)
    # =========================================================================

    for period in [20, 50]:
        ma = close.rolling(period).mean()
        std = close.rolling(period).std()

        upper = ma + 2 * std
        lower = ma - 2 * std

        # Posicion dentro de las bandas (-1 a 1)
        bb_position = (close - lower) / (upper - lower)
        features[f'{prefix}bb{period}_position'] = bb_position * 2 - 1  # -1 a 1

        # Fuera de bandas (oportunidad reversion)
        features[f'{prefix}bb{period}_above_upper'] = (close > upper).astype(float)
        features[f'{prefix}bb{period}_below_lower'] = (close < lower).astype(float)

        # Bandas expandidas (alta volatilidad = buena oportunidad)
        bb_width = (upper - lower) / ma
        features[f'{prefix}bb{period}_width'] = bb_width
        features[f'{prefix}bb{period}_expanded'] = (bb_width > bb_width.rolling(50).quantile(0.8)).astype(float)

    # =========================================================================
    # 4. RETROCESO DE FIBONACCI (Niveles de Reversion)
    # =========================================================================

    # Swing high/low de ultimos N periodos
    for lookback in [20, 50]:
        swing_high = high.rolling(lookback).max()
        swing_low = low.rolling(lookback).min()

        # Porcentaje de retroceso
        range_size = swing_high - swing_low
        retracement = (swing_high - close) / range_size.replace(0, np.nan)

        features[f'{prefix}fib{lookback}_retracement'] = retracement

        # Cerca de niveles clave de Fibonacci
        features[f'{prefix}fib{lookback}_near_382'] = (abs(retracement - 0.382) < 0.05).astype(float)
        features[f'{prefix}fib{lookback}_near_500'] = (abs(retracement - 0.500) < 0.05).astype(float)
        features[f'{prefix}fib{lookback}_near_618'] = (abs(retracement - 0.618) < 0.05).astype(float)

    # =========================================================================
    # 5. VELOCIDAD DE MOVIMIENTO (Reversion Timing)
    # =========================================================================

    returns = close.pct_change()

    for period in [5, 10, 20]:
        # Retorno acumulado
        cum_ret = returns.rolling(period).sum()
        features[f'{prefix}cum_ret_{period}'] = cum_ret

        # Movimiento rapido (oportunidad reversion)
        ret_percentile = cum_ret.rolling(100).rank(pct=True)
        features[f'{prefix}fast_up_{period}'] = (ret_percentile > 0.95).astype(float)
        features[f'{prefix}fast_down_{period}'] = (ret_percentile < 0.05).astype(float)

    # =========================================================================
    # 6. ESTRUCTURA DE MERCADO
    # =========================================================================

    # Cuantas barras desde el ultimo high/low
    highs = high == high.rolling(20).max()
    lows = low == low.rolling(20).min()

    # Barras desde ultimo swing
    features[f'{prefix}bars_since_high'] = highs.astype(int).groupby((~highs).cumsum()).cumcount().astype(float) / 20
    features[f'{prefix}bars_since_low'] = lows.astype(int).groupby((~lows).cumsum()).cumcount().astype(float) / 20

    # =========================================================================
    # 7. COMBINACIONES DE SENALES (ALTA PROBABILIDAD)
    # =========================================================================

    # RSI oversold + cerca de lower BB + retroceso > 61.8%
    rsi14 = features.get(f'{prefix}rsi14', 0)
    bb_below = features.get(f'{prefix}bb20_below_lower', 0)
    fib_deep = features.get(f'{prefix}fib20_retracement', 0) > 0.618

    features[f'{prefix}strong_buy_signal'] = (
        (features.get(f'{prefix}rsi14_oversold', 0) > 0) &
        (bb_below > 0) &
        fib_deep
    ).astype(float)

    # RSI overbought + cerca de upper BB + retroceso < 38.2%
    bb_above = features.get(f'{prefix}bb20_above_upper', 0)
    fib_shallow = features.get(f'{prefix}fib20_retracement', 0) < 0.382

    features[f'{prefix}strong_sell_signal'] = (
        (features.get(f'{prefix}rsi14_overbought', 0) > 0) &
        (bb_above > 0) &
        fib_shallow
    ).astype(float)

    # Limpiar NaNs
    features = features.fillna(0)

    logger.info(f"Computed {len(features.columns)} mean reversion features for {prefix}")

    return features


def make_mean_reversion_features(
    base_timeframe: str = 'M5',
    data_dir: str = 'data'
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Create mean reversion focused features.

    Returns:
        X: Feature matrix (N x F)
        returns: Return array (N,)
        timestamps: DatetimeIndex
    """
    import os

    print("="*70)
    print("MEAN REVERSION FEATURE SYSTEM")
    print("="*70)

    # 1. Load base data
    print("\n1. Loading base data...")
    base_file = os.path.join(data_dir, f'xauusd_{base_timeframe.lower()}.csv')
    df = pd.read_csv(base_file, parse_dates=['time'], index_col='time')
    print(f"   Loaded {len(df)} bars")

    # 2. Compute returns
    print("\n2. Computing returns...")
    returns = df['close'].pct_change().fillna(0).values

    # Helper to make index timezone naive
    def make_tz_naive(idx):
        idx = pd.to_datetime(idx)
        if idx.tz is not None:
            return idx.tz_localize(None)
        return idx

    # 3. Compute mean reversion features for multiple timeframes
    print("\n3. Computing mean reversion features...")

    all_features = []

    # Base timeframe - make index tz naive
    df.index = make_tz_naive(df.index)
    mr_features = compute_mean_reversion_features(df, prefix=f'{base_timeframe}_')
    mr_features.index = make_tz_naive(mr_features.index)
    all_features.append(mr_features)
    print(f"   {base_timeframe}: {len(mr_features.columns)} features")

    # Higher timeframes
    for tf, period in [('H1', 12), ('H4', 48)]:
        tf_file = os.path.join(data_dir, f'xauusd_{tf.lower()}.csv')
        if os.path.exists(tf_file):
            tf_df = pd.read_csv(tf_file, parse_dates=['time'], index_col='time')
            tf_df.index = make_tz_naive(tf_df.index)
            tf_features = compute_mean_reversion_features(tf_df, prefix=f'{tf}_')
            tf_features.index = make_tz_naive(tf_features.index)

            # Resample to base timeframe
            tf_features = tf_features.reindex(df.index, method='ffill').fillna(0)
            all_features.append(tf_features)
            print(f"   {tf}: {len(tf_features.columns)} features")

    # 4. Combine all features
    print("\n4. Combining features...")
    combined = pd.concat(all_features, axis=1)

    # Drop rows with NaN
    valid_mask = ~combined.isna().any(axis=1)
    combined = combined[valid_mask]
    returns = returns[valid_mask.values]
    timestamps = df.index[valid_mask]

    print(f"\n   Total features: {combined.shape[1]}")
    print(f"   Total samples: {len(combined)}")

    X = combined.values.astype(np.float32)

    return X, returns.astype(np.float32), timestamps


if __name__ == "__main__":
    X, returns, ts = make_mean_reversion_features()
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Returns shape: {returns.shape}")
