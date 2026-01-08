"""
PPO-LSTM Model Configuration for XAU-SNIPER
=============================================

Uses RecurrentPPO from sb3-contrib for temporal memory.

Key settings optimized for:
- Low frequency trading (~1 trade/week)
- Long-term value (gamma=0.999)
- Stable learning (low lr)
- Deterministic behavior (low entropy)
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass
class PPOLSTMConfig:
    """Configuration for PPO-LSTM model."""

    # Policy architecture
    policy: str = "MlpLstmPolicy"
    lstm_hidden_size: int = 256
    n_lstm_layers: int = 1
    shared_lstm: bool = True  # Share LSTM between actor and critic

    # MLP layers after LSTM
    net_arch: Dict[str, Any] = field(default_factory=lambda: {
        'pi': [128, 64],   # Policy network
        'vf': [128, 64]    # Value network
    })

    # Learning parameters
    learning_rate: float = 3e-5  # Slow and stable
    n_steps: int = 2048          # Steps per update (~3 months H1)
    batch_size: int = 128        # Mini-batch size
    n_epochs: int = 10           # Epochs per update

    # Discount and advantage
    gamma: float = 0.999         # Long-term vision
    gae_lambda: float = 0.95     # GAE smoothing

    # PPO clipping
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None  # Value function clipping

    # Entropy and regularization
    ent_coef: float = 0.005      # Low entropy = deterministic
    vf_coef: float = 0.5         # Value function weight
    max_grad_norm: float = 0.5   # Gradient clipping

    # LSTM specific
    enable_critic_lstm: bool = False
    lstm_num_layers: int = 1

    # Training
    seed: int = 42
    device: str = "auto"
    verbose: int = 1

    # Tensorboard
    tensorboard_log: Optional[str] = None # "./logs/tensorboard/"


def create_ppo_lstm_model(
    env,
    config: Optional[PPOLSTMConfig] = None,
    load_path: Optional[str] = None
):
    """
    Create or load a PPO-LSTM model.

    Args:
        env: Gymnasium environment
        config: Model configuration
        load_path: Path to load existing model

    Returns:
        RecurrentPPO model
    """
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError:
        raise ImportError(
            "sb3-contrib is required for RecurrentPPO. "
            "Install with: pip install sb3-contrib"
        )

    if config is None:
        config = PPOLSTMConfig()

    # Load existing model
    if load_path and os.path.exists(load_path):
        logger.info(f"Loading model from {load_path}")
        model = RecurrentPPO.load(load_path, env=env)
        return model

    # Policy kwargs for LSTM architecture
    policy_kwargs = {
        "lstm_hidden_size": config.lstm_hidden_size,
        "n_lstm_layers": config.n_lstm_layers,
        "shared_lstm": config.shared_lstm,
        "enable_critic_lstm": config.enable_critic_lstm,
        "net_arch": config.net_arch,
    }

    # Create model
    model = RecurrentPPO(
        policy=config.policy,
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        clip_range_vf=config.clip_range_vf,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=policy_kwargs,
        seed=config.seed,
        device=config.device,
        verbose=config.verbose,
        tensorboard_log=config.tensorboard_log,
    )

    logger.info(f"Created PPO-LSTM model with config: {config}")
    logger.info(f"Policy architecture: {policy_kwargs}")

    return model


def get_linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Create a learning rate schedule that decreases linearly.

    Args:
        initial_value: Initial learning rate

    Returns:
        Schedule function
    """
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return schedule


def get_exponential_schedule(
    initial_value: float,
    decay_rate: float = 0.9
) -> Callable[[float], float]:
    """
    Create a learning rate schedule that decreases exponentially.

    Args:
        initial_value: Initial learning rate
        decay_rate: Decay rate per step

    Returns:
        Schedule function
    """
    def schedule(progress_remaining: float) -> float:
        return initial_value * (decay_rate ** (1 - progress_remaining))

    return schedule


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PPO-LSTM MODEL TEST")
    print("=" * 70)

    # Check if sb3-contrib is available
    try:
        from sb3_contrib import RecurrentPPO
        print("sb3-contrib installed successfully")
    except ImportError:
        print("ERROR: sb3-contrib not installed")
        print("Install with: pip install sb3-contrib")
        exit(1)

    # Create a dummy environment for testing
    import gymnasium as gym
    from gymnasium import spaces
    import numpy as np

    class DummyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(low=-1, high=1, shape=(30,), dtype=np.float32)
            self.action_space = spaces.Discrete(3)
            self.step_count = 0

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.step_count = 0
            return np.zeros(30, dtype=np.float32), {}

        def step(self, action):
            self.step_count += 1
            obs = np.random.randn(30).astype(np.float32)
            reward = np.random.randn()
            terminated = self.step_count >= 100
            return obs, reward, terminated, False, {}

    # Create environment
    env = DummyEnv()

    # Create model
    config = PPOLSTMConfig(
        learning_rate=3e-5,
        n_steps=256,  # Small for testing
        batch_size=32,
        verbose=1
    )

    model = create_ppo_lstm_model(env, config)

    print(f"\nModel created successfully")
    print(f"Policy: {model.policy}")
    print(f"Device: {model.device}")

    # Quick training test
    print("\nRunning quick training test (1000 steps)...")
    model.learn(total_timesteps=1000, progress_bar=True)

    print("\nTraining complete!")

    # Test prediction
    obs, _ = env.reset()
    lstm_states = None
    episode_start = True

    for i in range(10):
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=True
        )
        obs, reward, terminated, truncated, info = env.step(action)
        episode_start = terminated or truncated
        print(f"Step {i+1}: action={action}, reward={reward:.4f}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
