# ANALISIS EXHAUSTIVO: XAU-SNIPER PPO-LSTM

> **Fecha:** 2026-01-06
> **Problema Principal:** El modelo sobre-opera (30% HOLD vs 80%+ requerido) y no aprende (28% win rate estancado)

---

## RESUMEN EJECUTIVO

El modelo actual tiene **7 problemas criticos** que impiden que aprenda el comportamiento "Sniper" deseado:

| # | Problema | Impacto | Severidad |
|---|----------|---------|-----------|
| 1 | Reward function no incentiva HOLD | Modelo no tiene razon para esperar | CRITICO |
| 2 | Entry penalty sin contrapeso | Solo castigo, sin reward por paciencia | CRITICO |
| 3 | Observation space incompleto | Modelo no sabe cuantos trades lleva | ALTO |
| 4 | Episode length mal calculado | 840 H1 = 7 semanas, no 1 semana | MEDIO |
| 5 | Features predictivas no usadas | DXY, calendario economico ausentes | MEDIO |
| 6 | Hiperparametros no alineados | ent_coef muy bajo para exploracion | MEDIO |
| 7 | Win rate estructuralmente bajo | SL/TP ratio no compensado | ALTO |

---

## 1. PROBLEMA CRITICO: REWARD FUNCTION

### 1.1 Situacion Actual (gold_env.py:365-439)

```python
def _calculate_reward(self, current_price: float) -> float:
    reward = 0.0

    # A. Entry penalty (UNICO castigo por operar)
    if self.just_opened_trade:
        reward -= self.entry_penalty  # -0.01

    # B. Patience reward (solo si HAY posicion abierta)
    if position.state != PositionState.FLAT and position.unrealized_pnl > 0:
        reward += 0.0001 * position.duration_bars  # max 0.01

    # C. Jackpot (raro, requiere 0.3% win)
    if self.just_closed_trade and pnl_pct >= 0.003:
        reward += 5.0

    # D. Step PnL (siempre activo, incluso sin posicion!)
    step_pnl_normalized = step_pnl / self.initial_balance
    reward += step_pnl_normalized * 0.5  # Esto no deberia existir sin posicion
```

### 1.2 Problemas Identificados

| Componente | Problema | Por que falla |
|------------|----------|---------------|
| Entry Penalty | Es el UNICO desincentivo para operar | No hay reward por NO operar |
| Patience Reward | Solo aplica CON posicion abierta | No hay reward por esperar SIN posicion |
| Step PnL | Se acumula aunque no haya posicion | Modelo recibe reward sin hacer nada util |
| Jackpot | 5.0 es grande pero ocurre ~5% de trades | Raro reward no cambia comportamiento base |

### 1.3 Analisis Matematico

**Escenario A: Modelo hace HOLD todo el episodio**
- Entry penalty: $0 (no abre trades)
- Patience: $0 (no hay posicion)
- Jackpot: $0 (no hay trades)
- Step PnL: ~$0 (sin posicion, equity no cambia)
- **Reward Total: ~0**

**Escenario B: Modelo hace 100 trades aleatorios**
- Entry penalty: -100 * 0.01 = -1.0
- Patience: ~+0.5 (en promedio por duracion)
- Jackpot: ~+5.0 * 5 trades ganadores = +25.0
- Step PnL: ~+2.0 (fluctuaciones)
- **Reward Total: ~26.5**

**Conclusion:** El modelo aprende que MAS TRADES = MAS REWARD, exactamente lo contrario de lo deseado.

### 1.4 Solucion Propuesta

```python
def _calculate_reward(self, current_price: float) -> float:
    reward = 0.0

    # A. REWARD POR PACIENCIA SIN POSICION (NUEVO - CRITICO)
    if self.risk_manager.position.state == PositionState.FLAT:
        # Reward por cada step que espera correctamente
        reward += 0.001  # Acumulativo: 840 steps * 0.001 = 0.84 base reward

        # Bonus por esperar en condiciones malas
        if self._is_bad_entry_condition():
            reward += 0.002  # Extra por evitar malas entradas

    # B. Entry penalty PROPORCIONAL a trades ya hechos
    if self.just_opened_trade:
        n_trades = len(self.risk_manager.trade_history)
        # Penalty escala: 1er trade = -0.01, 10mo trade = -0.1
        reward -= self.entry_penalty * (1 + n_trades * 0.5)

    # C. Patience reward mejorado (mantener posicion ganadora)
    if position.state != PositionState.FLAT and position.unrealized_pnl > 0:
        reward += 0.0005 * position.duration_bars  # 5x mas que antes

    # D. Jackpot (sin cambios, ya es bueno)
    # E. Step PnL SOLO si hay posicion (CORREGIDO)
    if position.state != PositionState.FLAT:
        reward += step_pnl_normalized * 0.5
```

---

## 2. PROBLEMA CRITICO: OBSERVATION SPACE

### 2.1 Situacion Actual (gold_env.py:448-486)

```python
def _get_observation(self) -> np.ndarray:
    # Features del mercado (43 features)
    features = [current_bar[col] for col in self.feature_columns]

    # Position info (6 features)
    position_info = [in_position, position_side, unrealized_pnl, duration,
                     recent_vol, trend_strength]

    # Total: 49 features
```

### 2.2 Features FALTANTES (criticas para Sniper)

| Feature | Descripcion | Por que es critica |
|---------|-------------|-------------------|
| `trades_this_episode` | Numero de trades abiertos en el episodio | Modelo debe saber si ya "uso su cuota" |
| `time_since_last_trade` | Barras desde ultimo trade | Previene trades consecutivos |
| `trades_per_week_ratio` | Ratio vs target (5 trades/semana) | Feedback directo sobre frecuencia |
| `win_rate_episode` | Win rate del episodio actual | Modelo aprende de su performance |
| `atr_filter` | ATR < 1.0 = mercado muerto | Proyecto.md lo menciona pero no se usa |

### 2.3 Solucion Propuesta

```python
def _get_observation(self) -> np.ndarray:
    # ... features existentes ...

    # NUEVAS features de comportamiento
    n_trades = len(self.risk_manager.trade_history)
    episode_steps = len(self.episode_actions)

    behavior_info = [
        # Frecuencia de trades
        n_trades / max(1, self.target_trades_per_week),  # Ratio vs objetivo
        min(n_trades / 20, 1.0),  # Saturacion (1.0 si >= 20 trades)

        # Tiempo desde ultimo trade
        self.steps_since_last_trade / 100,  # Normalizado

        # Win rate actual
        self.current_win_rate,

        # Filtro de mercado muerto
        1.0 if current_bar['atr_ratio'] < 1.0 else 0.0,  # Proyecto.md lo requiere
    ]

    obs = np.array(features + position_info + behavior_info, dtype=np.float32)
```

---

## 3. PROBLEMA: EPISODE LENGTH

### 3.1 Situacion Actual

```python
episode_length = 840  # Configurado en run_diagnostic.py
```

### 3.2 Calculo Incorrecto

| Calculo | Resultado |
|---------|-----------|
| 840 H1 barras | 840 horas |
| 840 / 24 | 35 dias |
| 35 / 5 trading days | **7 semanas** |

**Proyecto.md dice:** "1-20 trades por SEMANA"
**Episodio actual:** 7 semanas

Si el target es 5 trades/semana, en 7 semanas deberia haber ~35 trades.
Pero el modelo hace ~110 trades = ~16 trades/semana (muy alto pero no catastrofico).

### 3.3 Solucion

```python
# Opcion A: Episodio de 1 semana
episode_length = 24 * 5  # = 120 H1 barras = 1 semana de trading

# Opcion B: Mantener 840 pero ajustar target
target_trades_per_episode = 35  # 5/semana * 7 semanas
```

---

## 4. PROBLEMA: FEATURES NO UTILIZADAS

### 4.1 Features del proyecto.md NO implementadas

| Feature | Descripcion | Estado |
|---------|-------------|--------|
| `event_proximity` | Horas hasta proximo evento economico | NO IMPLEMENTADO |
| `dxy_ret` | Retorno diario del DXY | Implementado pero NO SE USA |
| `us10y_level` | Nivel de yields | NO IMPLEMENTADO |

### 4.2 En h1_features.py

```python
# Linea 379: DXY solo se agrega si se pasa dxy_df
if config.include_dxy and dxy_df is not None:
    df = _add_dxy_features(df, dxy_df)
```

**Problema:** En `run_diagnostic.py` NUNCA se pasa `dxy_df`:

```python
# Linea 261
df_h1 = add_features_h1(df_h1)  # Sin dxy_df!
```

### 4.3 Solucion

```python
# En run_diagnostic.py
dxy_path = os.path.join(data_dir, 'dxy_daily.csv')
dxy_df = pd.read_csv(dxy_path) if os.path.exists(dxy_path) else None
df_h1 = add_features_h1(df_h1, dxy_df=dxy_df)
```

---

## 5. PROBLEMA: WIN RATE ESTRUCTURAL

### 5.1 Analisis de TP/SL

| Parametro | Valor | Descripcion |
|-----------|-------|-------------|
| TP | 0.3% | +6 pips en oro (~$6/oz) |
| SL | 0.15% | -3 pips en oro (~$3/oz) |
| Ratio | 2:1 | Necesita >33% win rate para breakeven |

### 5.2 Win Rate Actual: 28%

Con 28% win rate y 2:1 ratio:
- 28 trades ganadores: +28 * 6 = +168
- 72 trades perdedores: -72 * 3 = -216
- **Resultado: -48 (perdida)**

### 5.3 Posibles Causas

1. **SL muy ajustado:** 0.15% = 3 pips. El ruido normal del oro puede tocar esto facilmente.
2. **Entradas aleatorias:** Sin features predictivas, el modelo entra aleatoriamente.
3. **No filtra mercado muerto:** ATR < 1.0 deberia prevenir entradas (proyecto.md lo requiere).

### 5.4 Solucion

```python
# Opcion A: Ajustar SL
risk_config = RiskConfig(
    tp_target=0.003,    # Mantener
    sl_initial=0.003,   # Aumentar a 1:1 temporalmente para diagnostico
)

# Opcion B: Filtro de ATR en reward
def _calculate_reward(self, current_price: float) -> float:
    atr_ratio = current_bar['atr_ratio']

    # Penalizar trades en mercado muerto
    if self.just_opened_trade and atr_ratio < 1.0:
        reward -= 0.05  # Penalidad extra por operar en Asia/mercado muerto
```

---

## 6. PROBLEMA: HIPERPARAMETROS

### 6.1 Configuracion Actual (run_diagnostic.py:303-323)

```python
model = RecurrentPPO(
    learning_rate=3e-5,   # OK
    n_steps=2048,         # OK
    gamma=0.999,          # OK
    ent_coef=0.005,       # PROBLEMA: Muy bajo para exploracion
)
```

### 6.2 Analisis de ent_coef

| ent_coef | Efecto |
|----------|--------|
| 0.005 | Muy bajo: modelo converge rapido a politica sub-optima |
| 0.01 | Recomendado: balance exploracion/explotacion |
| 0.02+ | Alto: mas exploracion pero entrenamiento mas lento |

**Problema:** Con ent_coef=0.005, el modelo rapidamente converge a "hacer muchos trades" porque es lo que inicialmente da reward (por el jackpot ocasional).

### 6.3 Solucion

```python
model = RecurrentPPO(
    ent_coef=0.02,  # Aumentar para mas exploracion
    # O usar decay:
    # ent_coef_schedule = lambda progress: 0.02 * (1 - progress * 0.5)
)
```

---

## 7. RESUMEN DE CAMBIOS REQUERIDOS

### Prioridad CRITICA (hacer primero)

1. **Modificar reward function** para dar reward por HOLD sin posicion
2. **Agregar features de comportamiento** al observation space
3. **Aumentar ent_coef** a 0.02

### Prioridad ALTA

4. **Implementar filtro ATR** (no operar si ATR < 1.0)
5. **Cargar DXY features** en diagnostic
6. **Ajustar episode_length** a 120 (1 semana)

### Prioridad MEDIA

7. **Ajustar SL** temporalmente para diagnostico
8. **Agregar calendario economico** si disponible

---

## 8. COMPARACION: PROYECTO.MD vs IMPLEMENTACION

| Aspecto | proyecto.md | Implementacion | Match |
|---------|-------------|----------------|-------|
| RSI Period | 21 | 21 | OK |
| RSI Thresholds | 75/25 | 75/25 | OK |
| MACD Fast | 16 | 16 | OK |
| MACD Slow | 34 | 34 | OK |
| MACD Signal | 13 | 13 | OK |
| Entry Penalty | 0.0005 (5 pips) | 0.01 | DIFERENTE |
| Patience Reward | Si hay posicion ganadora | Solo con posicion | OK |
| Jackpot 0.3% | reward += 2.0 | reward += 5.0 | DIFERENTE |
| ATR Filter | < 1.0 = mercado muerto | NO IMPLEMENTADO | FALTA |
| DXY Features | Opcional | No se carga | FALTA |
| event_proximity | Mencionado | NO IMPLEMENTADO | FALTA |
| Trades/Ano | 50-120 (~1-2/semana) | ~110/episodio(7sem) | ALTO |

---

## 9. SIGUIENTE PASO RECOMENDADO

Implementar los cambios en este orden:

```
1. Modificar gold_env.py:
   - Agregar reward por HOLD sin posicion (+0.001/step)
   - Agregar penalty escalable por numero de trades
   - Agregar features de comportamiento al obs

2. Modificar run_diagnostic.py:
   - Cargar DXY data
   - Cambiar episode_length a 120
   - Cambiar ent_coef a 0.02

3. Correr test de 100k steps para validar cambios

4. Si mejora, continuar con 500k-1M steps
```

---

## 10. METRICAS OBJETIVO

Despues de implementar cambios, el modelo debe mostrar:

| Metrica | Actual | Objetivo |
|---------|--------|----------|
| HOLD % | 30% | > 80% |
| Trades/semana | ~16 | 1-20 |
| Win Rate | 28% | > 40% |
| Tendencia | ESTANCADO | MEJORANDO |
| Sharpe | ~0 | > 0.5 |

---

*Documento generado automaticamente por analisis de codigo*
