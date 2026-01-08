# INVESTIGACION: Reward Shaping para RL Trading (Gold/Forex/CFD)

> **Fecha:** 2026-01-06
> **Objetivo:** Identificar mejores practicas para balancear rewards en trading con RL
> **Problema actual:** Modelo PPO-LSTM opera aleatoriamente (33% cada accion, 50% win rate)

---

## RESUMEN EJECUTIVO

Despues de investigar repositorios, papers academicos y proyectos open-source, identificamos **7 hallazgos criticos** que explican por que nuestro modelo no aprende y que soluciones han funcionado en la industria.

### Hallazgo Principal
> **"Performed experiments aimed at teaching the agent to conduct less frequent but confident trades did not produce the desired outcomes."**
> - Fuente: [D3F4LT4ST/RL-trading](https://github.com/D3F4LT4ST/RL-trading)

Este es exactamente nuestro problema. La investigacion confirma que **reward shaping tradicional NO es suficiente** para controlar la frecuencia de trading.

---

## 1. TIPOS DE REWARD FUNCTIONS EN TRADING RL

### 1.1 Reward Basico (Profit-Based)
```
r_t = log(portfolio_t / portfolio_{t-1})
```
- **Ventaja:** Simple, directo
- **Desventaja:** Rewards esparsos, no controla frecuencia
- **Usado por:** FinRL, la mayoria de implementaciones basicas

### 1.2 Differential Sharpe Ratio (DSR)
```
DSR_t = (A_{t-1} * delta_t - 0.5 * B_{t-1} * delta_t^2) / (B_{t-1} - A_{t-1}^2)^1.5
donde:
  A_t = A_{t-1} + eta * (R_t - A_{t-1})  # EMA de returns
  B_t = B_{t-1} + eta * (R_t^2 - B_{t-1})  # EMA de returns^2
```
- **Ventaja:** Risk-adjusted, penaliza volatilidad
- **Desventaja:** Complejo de implementar, inestable al inicio
- **Paper:** Moody & Saffell (1998) - "Reinforcement Learning for Trading"
- **Fuente:** [NeurIPS Paper](http://papers.neurips.cc/paper/1551-reinforcement-learning-for-trading.pdf)

### 1.3 Embedded Drawdown Constraint
```
r_t = return_t - lambda * max(0, drawdown_t - target_mdd)
```
- **Ventaja:** Controla directamente el drawdown maximo
- **Desventaja:** Puede ser muy conservador
- **Paper:** Wu et al. (2022) - [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1568494622004082)

### 1.4 Multi-Reward Composito
```
r_t = w1 * Sharpe + w2 * Sortino + w3 * Calmar
con pesos dinamicos basados en performance
```
- **Ventaja:** Captura multiples objetivos
- **Paper:** [Springer 2025](https://link.springer.com/article/10.1007/s44196-025-00875-8)

### 1.5 Trailing Reward + PnL (Tsantekidis et al.)
```
r_t = PnL_t + trailing_reward_t
donde trailing_reward = bonus por mantener posicion ganadora
```
- **Ventaja:** Aborda el problema de rewards esparsos
- **Resultado:** Aumento en Sharpe Ratio
- **Paper:** [arXiv 2411.01456](https://arxiv.org/abs/2411.01456)

---

## 2. EL PROBLEMA DEL OVER-TRADING

### 2.1 Por que ocurre
| Causa | Explicacion |
|-------|-------------|
| Rewards densos | Feedback constante incentiva accion constante |
| Sin costo de oportunidad | No hay penalidad por "no esperar" |
| Exploration alta | ent_coef alto causa acciones aleatorias |
| Commission insuficiente | Costos no son lo suficientemente prohibitivos |

### 2.2 Soluciones Encontradas

#### A. RL-Cooldown (Proyecto Especifico)
- **Repositorio:** [Degergokalp/RL-cooldown](https://github.com/Degergokalp/RL-cooldown)
- **Enfoque:** Agente secundario que decide cuando "enfriarse" (skip trades)
- **Resultado:** Mejora performance al evitar trades en condiciones desfavorables

#### B. Event-Driven Sparse Rewards (ETDQN)
- **Paper:** [ScienceDirect 2023](https://www.sciencedirect.com/science/article/abs/pii/S0957417423023990)
- **Enfoque:** Reward SOLO al cerrar trade, no durante
- **Tecnicas adicionales:**
  - Prioritized Experience Replay
  - Hindsight Experience Replay
  - Noisy Linear Layers para exploracion
  - Dueling Network Architecture

#### C. Turnover Penalty Explicito
> "Turnover is a secret regularizer. Costs alone weren't enough; explicitly penalizing turnover curbed churn and improved OOS smoothness."
- **Implementacion:**
```python
turnover_penalty = -lambda * abs(position_change)
reward += turnover_penalty
```

#### D. Action Masking / Cooldown Period
```python
if steps_since_last_trade < cooldown_period:
    # Forzar HOLD, no permitir BUY/SELL
    available_actions = [HOLD]
```

---

## 3. CONFIGURACION DE PPO PARA TRADING

### 3.1 Entropy Coefficient (ent_coef)

| Valor | Efecto | Recomendacion |
|-------|--------|---------------|
| < 0.005 | Convergencia rapida a politica sub-optima | NO para trading |
| 0.01-0.02 | Balance exploration/exploitation | RECOMENDADO inicial |
| > 0.05 | Demasiada exploracion, comportamiento aleatorio | NO |

**Mejor practica:** Usar adaptive entropy (axPPO)
- Empezar alto (0.02-0.05) y decaer basado en performance
- **Paper:** [arXiv 2405.04664](https://arxiv.org/html/2405.04664v1)

### 3.2 Gamma (Discount Factor)
- **Trading intraday:** gamma = 0.99
- **Trading swing/position:** gamma = 0.999
- **Nuestro caso (H1):** gamma = 0.999 es correcto

### 3.3 Workflow Recomendado
> "LSTM helps - but only after stability. Start with PPO-MLP to debug, then switch to PPO-LSTM."

Esto sugiere que deberiamos **primero hacer funcionar PPO-MLP basico** antes de agregar complejidad LSTM.

---

## 4. OBSERVACIONES CLAVE DE REPOSITORIOS

### 4.1 FinRL Framework
- **Reward:** `r(s,a,s') = v' - v` (cambio en valor de portfolio)
- **Action Space:** `{-k,...,-1, 0, 1,...,k}` donde k = shares
- **Soporte:** PPO, A2C, DDPG, TD3, SAC
- **Repo:** [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL)

### 4.2 D3F4LT4ST/RL-trading (Forex PPO)
- **Hallazgo critico:** Introducir comisiones redujo severamente la rentabilidad
- **DQN/A2C:** Convergieron a buy-and-hold
- **PPO:** Continuo high-frequency trading a pesar de comisiones
- **Conclusion:** "Standard reward shaping alone cannot effectively discourage overtrading"

### 4.3 TomatoFT/Forex-Automation (PPO, TD3, DDPG)
- **Paper:** RIVF 2022
- **Enfoque:** Ensemble de multiples algoritmos
- **Seleccion:** Mejor agente basado en Sharpe Ratio

### 4.4 trading-rl (ICASSP 2019)
- **Enfoque:** Price Trailing con DQN
- **Repo:** [Kostis-S-Z/trading-rl](https://github.com/Kostis-S-Z/trading-rl)

---

## 5. ANALISIS: POR QUE NUESTRO MODELO FALLA

### 5.1 Comparacion con Best Practices

| Aspecto | Nuestro Modelo | Best Practice | Gap |
|---------|----------------|---------------|-----|
| Reward base | PnL + penalties | Sharpe/Sortino | CAMBIAR |
| Drawdown | No se penaliza | Embedded constraint | AGREGAR |
| Turnover | Entry penalty fijo | Penalty escalable | MEJORAR |
| Sparse rewards | Reward cada step | Solo al cerrar trade | CONSIDERAR |
| Exploration | ent_coef=0.02 | Adaptive decay | IMPLEMENTAR |
| Action masking | No | Cooldown period | AGREGAR |
| Arquitectura | PPO-LSTM directo | PPO-MLP primero | PROBAR |

### 5.2 Problemas Especificos Identificados

1. **Reward demasiado denso:** Damos reward cada step, incluso sin posicion
2. **Sin cooldown:** Modelo puede abrir trade inmediatamente despues de cerrar
3. **Drawdown no penalizado:** No hay pain por secuencia de losses
4. **ent_coef estatico:** No decae con el tiempo
5. **Complejidad innecesaria:** LSTM puede estar complicando el aprendizaje

---

## 6. SOLUCIONES PROPUESTAS (Priorizado)

### PRIORIDAD 1: Simplificar (Alto Impacto, Bajo Esfuerzo)

#### A. Probar PPO-MLP primero
```python
policy_kwargs = dict(
    net_arch=[256, 256],  # Sin LSTM
)
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs)
```

#### B. Implementar cooldown obligatorio
```python
# En gold_env.py
self.cooldown_period = 5  # Minimo 5 H1 bars entre trades

def step(self, action):
    if self.steps_since_last_trade < self.cooldown_period:
        action = 0  # Forzar HOLD
```

### PRIORIDAD 2: Reward Restructuring (Alto Impacto, Medio Esfuerzo)

#### A. Cambiar a Sparse Rewards
```python
def _calculate_reward(self):
    if self.just_closed_trade:
        # Reward SOLO al cerrar
        pnl = self.last_trade_pnl
        if pnl > 0:
            return pnl / self.initial_balance * 100  # Escalar
        else:
            return pnl / self.initial_balance * 200  # Penalty 2x
    else:
        return 0  # CERO reward mientras espera
```

#### B. Agregar Drawdown Penalty
```python
def _calculate_drawdown_penalty(self):
    if self.max_equity > 0:
        current_dd = (self.max_equity - self.equity) / self.max_equity
        if current_dd > 0.02:  # >2% drawdown
            return -current_dd * 10  # Penalty proporcional
    return 0
```

### PRIORIDAD 3: Adaptive Exploration (Medio Impacto, Medio Esfuerzo)

```python
# Entropy coefficient decay
def ent_coef_schedule(progress_remaining):
    # Empieza en 0.05, termina en 0.005
    return 0.05 * progress_remaining + 0.005 * (1 - progress_remaining)

model = PPO(..., ent_coef=ent_coef_schedule)
```

### PRIORIDAD 4: Differential Sharpe Ratio (Alto Impacto, Alto Esfuerzo)

```python
class DifferentialSharpeReward:
    def __init__(self, eta=0.01):
        self.eta = eta
        self.A = 0  # EMA de returns
        self.B = 0  # EMA de returns^2

    def calculate(self, return_t):
        delta_A = return_t - self.A
        delta_B = return_t**2 - self.B

        if self.B - self.A**2 > 0:
            dsr = (self.B * delta_A - 0.5 * self.A * delta_B) / \
                  (self.B - self.A**2)**1.5
        else:
            dsr = 0

        # Update EMAs
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B

        return dsr
```

---

## 7. PLAN DE ACCION RECOMENDADO

### Fase 1: Simplificacion (1-2 horas)
1. Cambiar a PPO-MLP (sin LSTM)
2. Implementar cooldown de 5 bars
3. Cambiar a sparse rewards (solo al cerrar trade)
4. Probar 100k steps

### Fase 2: Risk-Adjusted Rewards (2-3 horas)
1. Agregar drawdown penalty
2. Implementar Differential Sharpe Ratio
3. Agregar turnover penalty explicito
4. Probar 200k steps

### Fase 3: Fine-tuning (2-3 horas)
1. Implementar adaptive entropy decay
2. Si Fase 1-2 funcionan, agregar LSTM de vuelta
3. Optimizar hiperparametros
4. Probar 500k-1M steps

### Fase 4: Validacion (1-2 horas)
1. Backtest en datos out-of-sample
2. Verificar que no hay overfitting
3. Documentar resultados finales

---

## 8. FUENTES Y REFERENCIAS

### Papers Academicos
- [Moody & Saffell - Reinforcement Learning for Trading (NeurIPS)](http://papers.neurips.cc/paper/1551-reinforcement-learning-for-trading.pdf)
- [Embedded Drawdown Constraint (ScienceDirect 2022)](https://www.sciencedirect.com/science/article/abs/pii/S1568494622004082)
- [ETDQN Sparse Rewards (ScienceDirect 2023)](https://www.sciencedirect.com/science/article/abs/pii/S0957417423023990)
- [Auxiliary Task for Forex PPO (arXiv 2024)](https://arxiv.org/abs/2411.01456)
- [Adaptive PPO Exploration (arXiv 2024)](https://arxiv.org/html/2405.04664v1)
- [Multi-Reward Portfolio Optimization (Springer 2025)](https://link.springer.com/article/10.1007/s44196-025-00875-8)
- [Sharpe Ratio Based Reward (Springer 2023)](https://link.springer.com/chapter/10.1007/978-3-031-34111-3_2)

### Repositorios GitHub
- [FinRL Framework](https://github.com/AI4Finance-Foundation/FinRL)
- [D3F4LT4ST/RL-trading](https://github.com/D3F4LT4ST/RL-trading)
- [TomatoFT/Forex-Automation](https://github.com/TomatoFT/Forex-Trading-Automation-with-Deep-Reinforcement-Learning)
- [Kostis-S-Z/trading-rl](https://github.com/Kostis-S-Z/trading-rl)
- [kayuksel/forex-rl-challenge](https://github.com/kayuksel/forex-rl-challenge)
- [RL-cooldown](https://github.com/Degergokalp/RL-cooldown)

### Documentacion
- [Stable Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [FinRL Documentation](https://finrl.readthedocs.io/en/latest/)

---

## 9. CONCLUSION

El problema de over-trading en RL es **bien documentado y dificil de resolver**. La investigacion muestra que:

1. **Reward shaping tradicional NO es suficiente** - Multiples proyectos confirman esto
2. **Sparse rewards funcionan mejor** - Feedback solo al cerrar trades
3. **Cooldown periods son efectivos** - Forzar espera minima entre trades
4. **Risk-adjusted rewards son necesarios** - Sharpe/Sortino > PnL puro
5. **Simplificar primero** - PPO-MLP antes de PPO-LSTM
6. **Adaptive exploration** - ent_coef debe decaer con el tiempo

El camino mas prometedor es **simplificar el problema primero** (cooldown + sparse rewards + PPO-MLP), validar que funciona, y luego agregar complejidad gradualmente.

---

*Documento generado: 2026-01-06*
*Investigacion realizada con WebSearch sobre repositorios y papers academicos*
