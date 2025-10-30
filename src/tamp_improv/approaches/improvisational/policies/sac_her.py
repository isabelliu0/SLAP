"""SAC+HER policy for end-to-end task learning."""

import os
import pickle
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer

from tamp_improv.approaches.improvisational.policies.base import (
    ActType,
    ObsType,
    Policy,
    TrainingData,
)
from tamp_improv.approaches.improvisational.policies.goal_rl import (
    HERTrainingProgressCallback,
)


@dataclass
class SACHERConfig:
    """Configuration for SAC+HER policy."""

    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1000000
    n_sampled_goal: int = 4
    goal_selection_strategy: str = "future"
    device: str = "cuda"


class SACHERPolicy(Policy[ObsType, ActType]):
    """SAC+HER policy for end-to-end task learning."""

    def __init__(self, seed: int, config: SACHERConfig | None = None):
        """Initialize policy."""
        super().__init__(seed)
        self.config = config or SACHERConfig()
        if self.config.device == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device
        self.model: SAC | None = None

    def initialize(self, env: gym.Env) -> None:
        """Initialize policy with environment."""

    def can_initiate(self) -> bool:
        """Check whether the policy can be executed."""
        return self.model is not None

    def get_action(self, obs: ObsType) -> ActType:
        """Get action from policy."""
        assert self.model is not None
        if isinstance(obs, dict):
            action, _ = self.model.predict(obs, deterministic=True)
        else:
            if hasattr(obs, "nodes"):
                flattened_obs = obs.nodes.flatten()
                action, _ = self.model.predict(flattened_obs, deterministic=True)
            else:
                raise ValueError("Invalid observation format")
        return action  # type: ignore[return-value]

    def train(self, env: gym.Env, train_data: TrainingData | None = None) -> None:
        """Train the policy using SAC+HER."""
        print("\nTraining SAC+HER policy...")
        replay_buffer_kwargs = {
            "n_sampled_goal": self.config.n_sampled_goal,
            "goal_selection_strategy": self.config.goal_selection_strategy,
        }
        max_episode_steps = getattr(env, "max_steps", 500)
        learning_starts = max_episode_steps * 2
        self.model = SAC(
            (
                "MultiInputPolicy"
                if isinstance(env.observation_space, gym.spaces.Dict)
                else "MlpPolicy"
            ),
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            buffer_size=self.config.buffer_size,
            learning_starts=learning_starts,
            device=self.device,
            seed=self._seed,
            verbose=1,
        )
        total_timesteps = 1_000_000  # Adjust as needed
        print(f"Training for {total_timesteps} timesteps...")
        callback = HERTrainingProgressCallback(check_freq=10000)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save(self, path: str) -> None:
        """Save policy."""
        assert self.model is not None
        os.makedirs(path, exist_ok=True)
        self.model.save(f"{path}/model")
        with open(f"{path}/model_observation_space.pkl", "wb") as f:
            pickle.dump(self.model.observation_space, f)
        with open(f"{path}/model_action_space.pkl", "wb") as f:
            pickle.dump(self.model.action_space, f)

    def load(self, path: str) -> None:
        """Load policy."""
        obs_space_path = f"{path}/model_observation_space.pkl"
        action_space_path = f"{path}/model_action_space.pkl"

        with open(obs_space_path, "rb") as f:
            observation_space = pickle.load(f)
        with open(action_space_path, "rb") as f:
            action_space = pickle.load(f)

        class DummyEnv(gym.Env):  # pylint: disable=abstract-method
            """Dummy environment for loading SAC model."""

            def __init__(self, observation_space, action_space):
                self.observation_space = observation_space
                self.action_space = action_space

            def compute_reward(
                self, achieved_goal, _desired_goal, _info, _indices=None
            ):
                """Compute reward based on achieved goal."""
                if isinstance(achieved_goal, np.ndarray):
                    return np.zeros(achieved_goal.shape[0], dtype=np.float32)
                return 0.0

            def reset(self, **_kwargs):
                if isinstance(self.observation_space, gym.spaces.Dict):
                    obs = {}
                    for key, space in self.observation_space.spaces.items():
                        obs[key] = np.zeros(space.shape, dtype=space.dtype)
                else:
                    obs = np.zeros(
                        self.observation_space.shape, dtype=self.observation_space.dtype
                    )
                return obs, {}

            def step(self, action):
                obs, _ = self.reset()
                return obs, 0.0, False, False, {}

        dummy_env = DummyEnv(observation_space, action_space)  # type: ignore[no-untyped-call]  # pylint:disable=line-too-long
        self.model = SAC.load(f"{path}/model", env=dummy_env, device=self.device)
