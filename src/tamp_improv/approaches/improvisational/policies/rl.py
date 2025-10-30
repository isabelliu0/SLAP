"""RL-based policy implementation."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray
from relational_structs import GroundAtom
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from torch import Tensor

from tamp_improv.approaches.improvisational.policies.base import (
    ActType,
    ObsType,
    Policy,
    PolicyContext,
    TrainingData,
)
from tamp_improv.utils.gpu_utils import DeviceContext


@dataclass
class RLConfig:
    """Configuration for RL policy."""

    learning_rate: float = 1e-4
    batch_size: int = 32
    n_epochs: int = 5
    gamma: float = 0.99
    ent_coef: float = 0.01
    device: str = "cuda"
    deterministic: bool = False


class TrainingProgressCallback(BaseCallback):
    """Callback to track training progress."""

    def __init__(
        self,
        check_freq: int = 100,
        verbose: int = 1,
        early_stopping: bool = False,
        early_stopping_patience: int = 1,
        early_stopping_threshold: float = 0.7,
        policy_key: str | None = None,
        save_checkpoints: bool = True,
        checkpoint_dir: str | None = None,
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.success_history: list[bool] = []
        self.episode_lengths: list[int] = []
        self.episode_rewards: list[float] = []
        self.current_length = 0
        self.current_reward = 0.0

        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.policy_key = policy_key
        self.plateau_count: int = 0
        self.success_rates: list[float] = []

        self.save_checkpoints = save_checkpoints
        self.checkpoint_dir = checkpoint_dir
        self.checkpoints: list[tuple[int, float, str]] = []
        self.best_checkpoint_path: str | None = None

    def _on_step(self) -> bool:
        self.current_length += 1
        self.current_reward += self.locals["rewards"][0]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        if dones[0]:
            success = not infos[0].get("TimeLimit.truncated", False)
            self.success_history.append(success)
            self.episode_lengths.append(self.current_length)
            self.episode_rewards.append(self.current_reward)

            self.current_length = 0
            self.current_reward = 0.0

            n_episodes = len(self.success_history)
            if n_episodes % self.check_freq == 0:
                recent_successes = self.success_history[-self.check_freq :]
                recent_lengths = self.episode_lengths[-self.check_freq :]
                recent_rewards = self.episode_rewards[-self.check_freq :]
                success_rate = sum(recent_successes) / len(recent_successes)
                print("\nTraining Progress:")
                print(f"Episodes: {n_episodes}")
                print(f"Recent Success%: {success_rate:.2%}")
                print(f"Recent Avg Episode Length: {np.mean(recent_lengths):.2f}")
                print(f"Recent Avg Reward: {np.mean(recent_rewards):.2f}")

                if self.save_checkpoints and self.checkpoint_dir is not None:
                    checkpoint_path = Path(self.checkpoint_dir)
                    checkpoint_path.mkdir(parents=True, exist_ok=True)
                    policy_suffix = f"_{self.policy_key}" if self.policy_key else ""
                    save_filename = f"checkpoint_ep_{n_episodes}{policy_suffix}"
                    save_path = str(checkpoint_path / save_filename)
                    self.model.save(save_path)
                    self.checkpoints.append((n_episodes, success_rate, save_path))

                if self.early_stopping:
                    self.success_rates.append(success_rate)
                    if len(self.success_rates) >= 3:
                        recent_rates = self.success_rates[-3:]
                        if all(
                            r >= self.early_stopping_threshold for r in recent_rates
                        ):
                            self.plateau_count += 1
                            print(
                                f"Plateau detected: {self.plateau_count}/{self.early_stopping_patience}"  # pylint: disable=line-too-long
                            )
                        if self.plateau_count >= self.early_stopping_patience:
                            policy_info = (
                                f" for {self.policy_key}" if self.policy_key else ""
                            )
                            print(
                                f"Early stopping{policy_info}: Success rate consistently above {self.early_stopping_threshold:.0%}"  # pylint: disable=line-too-long
                            )
                            return False  # Stop training

        return True

    def _on_training_end(self) -> None:
        """Print final training statistics."""
        print("\nFinal Training Results:")
        if self.success_history:
            print(f"Overall Success Rate: {self._get_success_rate:.2%}")
            print(f"Overall Avg Episode Length: {self._get_avg_episode_length:.2f}")
            print(f"Overall Avg Reward: {self._get_avg_reward:.2f}")
            if (
                self.early_stopping
                and self.plateau_count >= self.early_stopping_patience
            ):
                print("Training stopped early due to plateau in performance.")
        else:
            print("No episodes completed during training.")

        if self.save_checkpoints and self.checkpoints:
            best_checkpoint = self._select_stable_checkpoint()
            if best_checkpoint:
                best_episode, best_success_rate, best_path = best_checkpoint
                print(
                    f"\nBest checkpoint: Episode {best_episode} (success rate: {best_success_rate:.2%})"  # pylint: disable=line-too-long
                )
                self.best_checkpoint_path = best_path
                for _, _, path in self.checkpoints:
                    if path != best_path:
                        zip_path = Path(f"{path}.zip")
                        if zip_path.exists():
                            zip_path.unlink()

    def _select_stable_checkpoint(
        self, min_success_rate: float = 0.1
    ) -> tuple[int, float, str] | None:
        """Select checkpoint with stable performance."""
        if not self.checkpoints:
            return None

        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x[0])
        stable_candidates: list[tuple[int, float, str]] = []
        for i, (episode, success_rate, path) in enumerate(sorted_checkpoints):
            is_stable = True
            if i > 0:
                _, prev_success_rate, _ = sorted_checkpoints[i - 1]
                if prev_success_rate < min_success_rate:
                    is_stable = False
            if i < len(sorted_checkpoints) - 1:
                _, next_success_rate, _ = sorted_checkpoints[i + 1]
                if next_success_rate < min_success_rate:
                    is_stable = False
            if is_stable:
                stable_candidates.append((episode, success_rate, path))

        if stable_candidates:
            best_stable = max(stable_candidates, key=lambda x: x[1])
            return best_stable

        return None

    @property
    def _get_success_rate(self) -> float:
        """Get the success rate over all training."""
        if not self.success_history:
            return 0.0
        return float(sum(self.success_history) / len(self.success_history))

    @property
    def _get_avg_episode_length(self) -> float:
        """Get the average episode length over all training."""
        if not self.episode_lengths:
            return 0.0
        return float(np.mean(self.episode_lengths))

    @property
    def _get_avg_reward(self) -> float:
        """Get the average reward over all training."""
        if not self.episode_rewards:
            return 0.0
        return float(np.mean(self.episode_rewards))


class RLPolicy(Policy[ObsType, ActType]):
    """RL policy using PPO."""

    def __init__(self, seed: int, config: RLConfig | None = None) -> None:
        """Initialize policy."""
        super().__init__(seed)
        self.config = config or RLConfig()
        self.device_ctx = DeviceContext(self.config.device)

        self._torch_generator = torch.Generator(device=self.device_ctx.device)
        self._torch_generator.manual_seed(seed)

        self.model: PPO | None = None

        self.trained_shortcuts: set[tuple[frozenset, frozenset]] = set()
        self._current_atoms: set[GroundAtom] | None = None
        self._goal_atoms: set[GroundAtom] | None = None

    def initialize(self, env: gym.Env) -> None:
        """Initialize policy with environment."""
        if self.model is None:
            self.model = PPO(
                "MlpPolicy",
                env,
                learning_rate=self.config.learning_rate,
                n_steps=100,  # Default value, will be updated in train()
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs,
                gamma=self.config.gamma,
                ent_coef=self.config.ent_coef,
                device=self.device_ctx.device,
                seed=self._seed,
                verbose=1,
            )

    def can_initiate(self):
        """Check whether the policy can be executed given the current
        context."""
        if self.model is None:
            return False
        if not self.trained_shortcuts:
            return True
        if self._current_atoms is None or self._goal_atoms is None:
            return False
        current_sig = (frozenset(self._current_atoms), frozenset(self._goal_atoms))
        return current_sig in self.trained_shortcuts

    def configure_context(self, context: PolicyContext[ObsType, ActType]) -> None:
        """Configure policy with context information."""
        self._current_atoms = context.current_atoms
        self._goal_atoms = context.goal_atoms

    def train(
        self,
        env: gym.Env,
        train_data: TrainingData | None,
        callback: BaseCallback | None = None,
    ) -> None:
        """Train policy."""
        # Handle pure RL training without training data
        if train_data is None:
            print("\nTraining in pure RL mode with direct environment interaction")
            if hasattr(env, "max_episode_steps"):
                max_steps = env.max_episode_steps
            elif hasattr(env, "env") and hasattr(env.env, "max_episode_steps"):
                max_steps = env.env.max_episode_steps
            else:
                raise ValueError(
                    "Environment does not have max_episode_steps attribute"
                )
            wrapped_env = FlattenGraphObsWrapper(env)
            self.model = PPO(
                "MlpPolicy",
                wrapped_env,
                learning_rate=self.config.learning_rate,
                n_steps=max_steps,
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs,
                gamma=self.config.gamma,
                ent_coef=self.config.ent_coef,
                device=self.device_ctx.device,
                seed=self._seed,
                verbose=1,
            )
            if callback is None:
                callback = TrainingProgressCallback()
            total_timesteps = 1_000_000  # Adjust as needed
            self.model.learn(total_timesteps=total_timesteps, callback=callback)
            return

        # Call base class train to initialize and configure env
        super().train(env, train_data)

        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.config.learning_rate,
            n_steps=train_data.config.get("max_training_steps_per_shortcut", 100),
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            ent_coef=self.config.ent_coef,
            device=self.device_ctx.device,
            seed=self._seed,
            verbose=1,
        )

        if callback is None:
            callback = TrainingProgressCallback(
                check_freq=train_data.config.get("training_record_interval", 100)
            )

        episodes_per_scenario = train_data.config.get("episodes_per_scenario", 2)
        max_steps = train_data.config.get("max_training_steps_per_shortcut", 100)
        total_timesteps = len(train_data.states) * episodes_per_scenario * max_steps

        print("Training Settings:")
        print(f"Max steps per episode: {max_steps}")
        print(f"Episodes per scenario: {episodes_per_scenario}")
        print(f"Total scenarios: {len(train_data.states)}")
        print(f"Total training timesteps: {total_timesteps}")

        self.model.learn(total_timesteps=total_timesteps, callback=callback)

        for current_atoms, goal_atoms in zip(
            train_data.current_atoms, train_data.goal_atoms
        ):
            sig = (frozenset(current_atoms), frozenset(goal_atoms))
            self.trained_shortcuts.add(sig)
        print(f"Policy trained on {len(self.trained_shortcuts)} unique shortcuts")

    def get_action(self, obs: ObsType) -> ActType:
        """Get action from policy."""
        if self.model is None:
            raise ValueError("Policy not trained or loaded")

        if hasattr(obs, "nodes"):
            obs_flat = obs.nodes.flatten()
            # Pad to match training observation space if needed
            assert self.model.observation_space.shape is not None
            expected_size = self.model.observation_space.shape[0]
            if len(obs_flat) < expected_size:
                padded = np.zeros(expected_size, dtype=np.float32)
                padded[: len(obs_flat)] = obs_flat
                obs_flat = padded
            elif len(obs_flat) > expected_size:
                obs_flat = obs_flat[:expected_size]
            obs_numpy = obs_flat
        else:
            obs_tensor = self.device_ctx(obs)
            obs_cpu = (
                obs_tensor.cpu() if isinstance(obs_tensor, Tensor) else obs_tensor
            )  # move to CPU for stable_baselines3
            obs_numpy = self.device_ctx.numpy(obs_cpu)

        with torch.no_grad():
            action, _ = self.model.predict(
                obs_numpy, deterministic=self.config.deterministic
            )

        # Convert back to original type
        if isinstance(obs, (int, float)):
            return cast(ActType, int(action[0]))
        return cast(ActType, action)

    def save(self, path: str) -> None:
        """Save policy."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load policy."""
        self.model = PPO.load(path)


class FlattenGraphObsWrapper(gym.ObservationWrapper):
    """Simple wrapper to flatten graph observations for stable-baselines3."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if hasattr(env.observation_space, "node_space"):
            max_size = 0
            for _ in range(10):
                sample_obs = env.observation_space.sample()
                flattened_size = sample_obs.nodes.flatten().shape[0]
                max_size = max(max_size, flattened_size)

            self.max_size: int | None = max_size
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(max_size,), dtype=np.float32
            )
        else:
            self.max_size = None

    def observation(
        self, observation: ObsType
    ) -> ObsType | NDArray[np.float32] | dict[str, np.ndarray]:
        if isinstance(observation, dict):
            result: dict[str, np.ndarray] = {}
            for key, value in observation.items():
                if hasattr(value, "nodes"):
                    flattened = value.nodes.flatten()
                    if self.max_size and len(flattened) < self.max_size:
                        padded = np.zeros(self.max_size, dtype=np.float32)
                        padded[: len(flattened)] = flattened
                        result[key] = padded
                    else:
                        result[key] = flattened
                else:
                    result[key] = value
            return result
        if hasattr(observation, "nodes"):
            flattened = observation.nodes.flatten()
            if self.max_size and len(flattened) < self.max_size:
                padded = np.zeros(self.max_size, dtype=np.float32)
                padded[: len(flattened)] = flattened
                return padded
            return flattened
        return observation
