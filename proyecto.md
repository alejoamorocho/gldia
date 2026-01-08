# XAU-SNIPER: Agente PPO-LSTM para Trading de Oro

> **Filosofia:** "Sniper" - Baja frecuencia, alta precision
> **Meta:** ~1 operacion semanal de alta calidad | Objetivo dinamico > 0.3% | Gestion de Riesgo M1

---

## 1. Arquitectura General

```
+------------------+     +------------------+     +------------------+
|   DATA LAYER     | --> |   FEATURE ENG    | --> |    PPO-LSTM      |
|  H1 (Cerebro)    |     |  Normalizacion   |     |  RecurrentPPO    |
|  M1 (Ejecutor)   |     |  Indicadores     |     |  MlpLstmPolicy   |
+------------------+     +------------------+     +------------------+
                                                          |
                                                          v
                         +------------------+     +------------------+
                         |   RISK MANAGER   | <-- |   GOLD ENV       |
                         |  Trailing Stop   |     |  Gym Interface   |
                         |  0.3% Target     |     |  M1 Simulation   |
                         +------------------+     +------------------+
```

---

## 2. Estructura de Datos Dual (H1 + M1)

### 2.1 df_h1 (El Cerebro)
- **Uso:** Observation Space de la red neuronal
- **Frecuencia:** 1 Hora
- **Indicadores:** Tendencia Macro, Ciclos de Sesion, Volatilidad

### 2.2 df_m1 (El Ejecutor)
- **Uso:** Simulacion interna de riesgo en `step()`
- **Frecuencia:** 1 Minuto
- **Indicadores:** Price Action (OHLC), Volumen, Picos de volatilidad

### 2.3 Sincronizacion
```python
# Para cada paso t en H1, extraer slice de M1
m1_slice = df_m1[t : t + timedelta(hours=1)]
```

---

## 3. Features del Modelo (Inputs - H1)

Todas las features se calculan sobre `df_h1` y deben estar normalizadas.

| Categoria   | Feature       | Parametros      | Formula                        | Razon                        |
|-------------|---------------|-----------------|--------------------------------|------------------------------|
| Mercado     | `log_ret`     | -               | `np.log(close/close.shift(1))`| Input base estacionario      |
| Volatilidad | `atr_ratio`   | ATR(14), SMA(48)| `ATR_14 / SMA_48(ATR)`        | FILTRO: < 1.0 = mercado muerto|
| Tendencia   | `dist_ema200` | EMA(200)        | `(close - EMA_200) / close`   | Ubicacion vs tendencia macro |
| Tendencia   | `dist_ema50`  | EMA(50)         | `(close - EMA_50) / close`    | Soporte dinamico             |
| Momentum    | `rsi_norm`    | RSI(14)         | `(RSI / 100) - 0.5`           | Sobrecompra/Sobreventa       |
| Fuerza      | `adx_norm`    | ADX(14)         | `ADX / 100`                   | < 0.25 = evitar (Rango)      |
| Tiempo      | `hour_sin`    | -               | `sin(2 * pi * hour / 24)`     | Ciclo Londres/NY             |
| Tiempo      | `hour_cos`    | -               | `cos(2 * pi * hour / 24)`     | Complemento ciclico          |

### Features Macro Adicionales (Opcional)
- `dxy_ret`: Retorno diario del DXY (correlacion inversa con oro)
- `us10y_level`: Nivel de yields (costo de oportunidad)
- `event_proximity`: Horas hasta proximo evento economico

---

## 4. Espacio de Acciones

```python
# Discrete(3) - Recomendado
actions = {
    0: "HOLD/WAIT",   # Esperar o mantener posicion
    1: "BUY/LONG",    # Abrir posicion larga
    2: "SELL/SHORT"   # Abrir posicion corta
}
```

---

## 5. Logica del Entorno (GoldEnv)

### 5.1 Algoritmo de step()
```python
def step(self, action):
    # 1. Decision H1
    if action in [1, 2] and not self.in_position:
        self._open_position(action)
        self._apply_costs()  # commission + slippage

    # 2. Simulacion M1 (si hay posicion)
    if self.in_position:
        m1_slice = self._get_m1_slice()
        for m1_bar in m1_slice:
            # Chequeo de panico
            if m1_bar.volume > self.avg_vol * 10:
                self._close_position("PANIC")
                break
            # Chequeo SL
            if self._hit_stop_loss(m1_bar):
                self._close_position("STOP_LOSS")
                break
            # Chequeo TP / Trailing
            if self._hit_take_profit(m1_bar):
                self._execute_trailing_logic()

    # 3. Calcular reward y next_obs
    reward = self._calculate_reward()
    obs = self._get_next_h1_observation()

    return obs, reward, done, info
```

---

## 6. Gestion de Riesgo "Sniper" (0.3%)

### 6.1 Parametros Iniciales
- **TP_Target:** 0.3% del precio de entrada
- **SL_Initial:** -0.15% (Ratio 2:1) o -1.5 * ATR

### 6.2 Maquina de Estados

```
Estado 1: RIESGO
  Precio: entrada → +0.29%
  SL: Fijo en -0.15%
         |
         | (precio toca +0.3%)
         v
Estado 2: TRIGGER
  Accion: Mover SL a +0.15%
  (Ganancia asegurada)
         |
         | (precio sigue subiendo)
         v
Estado 3: TRAILING
  SL persigue precio a distancia de 0.15%
  Ej: Si precio = +1.0%, SL = +0.85%
```

---

## 7. Funcion de Recompensa

```python
def calculate_reward(self):
    reward = 0

    # A. PENALIZACION DE ENTRADA (Hurdle Rate)
    if self.just_opened_trade:
        reward -= 0.0005  # 5 pips de castigo

    # B. RECOMPENSA DE PACIENCIA
    if self.in_position and self.unrealized_pnl > 0:
        reward += 0.00005 * self.duration_bars

    # C. JACKPOT (Objetivo 0.3%)
    if self.just_closed_trade and self.max_trade_pnl >= 0.003:
        reward += 2.0

    # D. OPTIMIZACION SHARPE/SORTINO
    step_pnl = self.current_equity - self.prev_equity

    if step_pnl < 0:
        reward += step_pnl * 3.0  # Castigo triple volatilidad negativa
    else:
        reward += step_pnl

    return reward
```

---

## 8. Configuracion del Modelo

| Hiperparametro | Valor           | Explicacion                          |
|----------------|-----------------|--------------------------------------|
| `policy`       | MlpLstmPolicy   | Habilita memoria temporal (LSTM)     |
| `learning_rate`| 3e-5            | Lento y estable                      |
| `n_steps`      | 2048            | ~3 meses de data H1 por batch        |
| `batch_size`   | 128             | Balance memoria/velocidad            |
| `gamma`        | 0.999           | Vision a muy largo plazo             |
| `gae_lambda`   | 0.95            | Suavizado de ventaja                 |
| `ent_coef`     | 0.005           | Baja entropia = determinismo         |

---

## 9. Estructura del Proyecto

```
GLDIA/
├── data/
│   ├── xauusd_h1.csv          # Datos horarios
│   ├── xauusd_m1.csv          # Datos por minuto
│   ├── dxy_daily.csv          # Indice dolar
│   ├── economic_calendar.py   # Calendario economico
│   └── fetch_correlations.py  # Datos macro
│
├── features/
│   ├── __init__.py
│   ├── h1_features.py         # Features para H1
│   └── mean_reversion_features.py
│
├── env/
│   ├── __init__.py
│   ├── gold_env.py            # Entorno Gym principal
│   ├── risk_manager.py        # Gestion de trailing stop
│   └── m1_executor.py         # Simulacion M1
│
├── models/
│   ├── __init__.py
│   └── ppo_lstm.py            # Configuracion PPO-LSTM
│
├── training/
│   ├── __init__.py
│   ├── train.py               # Script de entrenamiento
│   └── callbacks.py           # Callbacks personalizados
│
├── evaluation/
│   ├── __init__.py
│   ├── backtest.py            # Backtesting
│   └── metrics.py             # Sharpe, Sortino, etc.
│
├── proyecto.md                # Este archivo
├── requirements.txt           # Dependencias
└── main.py                    # Entry point
```

---

## 10. Metricas de Exito

| Metrica         | Objetivo   | Descripcion                    |
|-----------------|------------|--------------------------------|
| Sharpe Ratio    | > 1.5      | Retorno ajustado por riesgo    |
| Sortino Ratio   | > 2.0      | Riesgo solo de perdidas        |
| Profit Factor   | > 1.5      | Ganancias / Perdidas           |
| Max Drawdown    | < 10%      | Perdida maxima desde pico      |
| Win Rate        | > 55%      | Porcentaje de trades ganadores |
| Trades/Año      | ~50-120    | Frecuencia sniper              |

---

## 11. Checklist de Implementacion

### Fase 1: Datos y Features
- [ ] Limpiar data H1 y M1 (remover weekends, gaps)
- [ ] Implementar `add_features_h1()` con features de Seccion 3
- [ ] Verificar normalizacion (sin inf/NaN)
- [ ] Sincronizar H1 con M1

### Fase 2: Entorno Gym
- [ ] Crear clase `GoldEnv(gym.Env)`
- [ ] Implementar `__init__` con carga dual H1/M1
- [ ] Implementar `step()` con loop M1
- [ ] Implementar trailing stop (0.3% trigger)
- [ ] Implementar `calculate_reward()` con Sharpe

### Fase 3: Entrenamiento
- [ ] Configurar RecurrentPPO de sb3-contrib
- [ ] Definir Callbacks (EvalCallback, etc.)
- [ ] Entrenar 2M-5M pasos

### Fase 4: Validacion
- [ ] Backtest en datos out-of-sample (2023+)
- [ ] Calcular metricas: Sharpe, Sortino, MaxDD
- [ ] Analizar distribucion de trades

---

## 12. Dependencias

```
torch>=2.0.0
stable-baselines3>=2.0.0
sb3-contrib>=2.0.0
gymnasium>=0.29.0
pandas>=2.0.0
numpy>=1.24.0
ta-lib  # O alternativa: pandas-ta
matplotlib
seaborn
```
