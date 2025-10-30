"""Goal-conditioned RL policy for learning shortcuts."""

import os
import pickle
import time
from dataclasses import dataclass
from typing import cast

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from relational_structs import GroundAtom
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

from tamp_improv.approaches.improvisational.policies.base import (
    ActType,
    GoalConditionedTrainingData,
    ObsType,
    Policy,
    PolicyContext,
)
from tamp_improv.approaches.improvisational.policies.node_replay_buffer import (
    NodeBasedHerBuffer,
)
from tamp_improv.benchmarks.goal_wrapper import GoalConditionedWrapper


@dataclass
class GoalConditionedRLConfig:
    """Configuration for goal-conditioned RL policy."""

    algorithm: str = "SAC"  # "TD3" or "SAC"

    # Learning parameters
    learning_rate: float = 3e-4
    batch_size: int = (
        256  # transitions sampled from the replay buffer for each training update
    )
    buffer_size: int = 1000000
    n_sampled_goal: int = 4

    # Training parameters
    success_threshold: float = 0.01
    action_noise: float = 0.1

    # Device settings
    device: str = "cuda"


class HERTrainingProgressCallback(BaseCallback):
    """Callback to track training progress specifically for HER."""

    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.success_history: list[bool] = []
        self.episode_lengths: list[int] = []
        self.episode_rewards: list[float] = []
        self.goal_distances: list[float] = []
        self.current_length = 0
        self.current_reward = 0.0
        self.last_print_time = time.time()
        self.last_print_step = 0

    def _on_step(self) -> bool:
        """Called at each step of training."""
        self.current_length += 1
        self.current_reward += self.locals["rewards"][0]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        if infos and "goal_distance" in infos[0]:
            self.goal_distances.append(infos[0]["goal_distance"])

        if dones[0]:
            is_success = infos[0].get("is_success", False)
            self.success_history.append(is_success)
            self.episode_lengths.append(self.current_length)
            self.episode_rewards.append(self.current_reward)

            # Reset counters
            self.current_length = 0
            self.current_reward = 0.0

        if self.num_timesteps - self.last_print_step >= self.check_freq:
            elapsed = time.time() - self.last_print_time
            steps_per_second = self.check_freq / max(elapsed, 1e-6)
            self.last_print_time = time.time()
            self.last_print_step = self.num_timesteps

            print("\nHER Training Progress:")
            print(f"Timesteps: {self.num_timesteps}")
            print(f"Time elapsed: {elapsed:.2f}s, {steps_per_second:.1f} steps/s")

            if self.success_history:
                n_recent = min(100, len(self.success_history))
                recent_successes = self.success_history[-n_recent:]
                recent_lengths = self.episode_lengths[-n_recent:]
                recent_rewards = self.episode_rewards[-n_recent:]

                print(f"Episodes completed: {len(self.success_history)}")
                print(
                    f"Recent Success Rate: {sum(recent_successes)/max(len(recent_successes), 1):.2%}"  # pylint: disable=line-too-long
                )
                print(
                    f"Recent Avg Episode Length: {np.mean(recent_lengths) if recent_lengths else 0:.2f}"  # pylint: disable=line-too-long
                )
                print(
                    f"Recent Avg Reward: {np.mean(recent_rewards) if recent_rewards else 0:.2f}"  # pylint: disable=line-too-long
                )

            if self.goal_distances:
                n_recent = min(100, len(self.goal_distances))
                recent_distances = self.goal_distances[-n_recent:]
                print(f"Recent Avg Goal Distance: {np.mean(recent_distances):.4f}")
                print(f"Recent Min Goal Distance: {np.min(recent_distances):.4f}")

            # Print current state info
            if "infos" in self.locals and self.locals["infos"]:
                info = self.locals["infos"][0]
                print("Current episode info:")
                if "source_node_id" in info and "goal_node_id" in info:
                    print(
                        f"  Source → Goal: {info['source_node_id']} → {info['goal_node_id']}"  # pylint: disable=line-too-long
                    )
                if "goal_distance" in info:
                    print(f"  Goal distance: {info['goal_distance']:.4f}")
                if "is_success" in info:
                    print(f"  Success: {info['is_success']}")

            # Print buffer info
            buffer = self.locals.get("replay_buffer")
            if buffer is not None:
                print(f"Replay buffer size: {buffer.size()}/{buffer.buffer_size}")

        return True

    def _on_training_end(self) -> None:
        """Print final training statistics."""
        print("\nFinal HER Training Results:")
        if self.success_history:
            print(f"Total Episodes: {len(self.success_history)}")
            print(f"Final Success Rate: {self._get_success_rate:.2%}")
            print(f"Avg Episode Length: {self._get_avg_episode_length:.2f}")
            print(f"Avg Episode Reward: {self._get_avg_reward:.2f}")

            if self.goal_distances:
                print(f"Avg Goal Distance: {np.mean(self.goal_distances):.4f}")
                print(f"Min Goal Distance: {np.min(self.goal_distances):.4f}")
        else:
            print("No episodes completed during training.")

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


class GoalConditionedRLPolicy(Policy[ObsType, ActType]):
    """Goal-conditioned RL policy for learning shortcuts."""

    def __init__(self, seed: int, config: GoalConditionedRLConfig | None = None):
        """Initialize policy."""
        super().__init__(seed)
        self.config = config or GoalConditionedRLConfig()
        if self.config.device == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device
        self.model: SAC | TD3 | None = None
        self.node_states: dict[int, ObsType] = {}
        self.valid_shortcuts: list[tuple[int, int]] = []
        self.node_atoms: dict[int, set[GroundAtom]] = {}

    def initialize(self, env):
        """Initialize policy with environment."""

    def can_initiate(self) -> bool:
        """Check whether the policy can be executed."""
        return self.model is not None

    def configure_context(self, context: PolicyContext) -> None:
        """Configure policy with context information."""

    def get_action(self, obs: ObsType) -> ActType:
        """Get action from policy."""
        assert self.model is not None
        assert isinstance(
            obs, dict
        ), "Observation must be a dictionary, consistent with HER"
        action, _ = self.model.predict(obs, deterministic=True)
        return cast(ActType, action)

    def train(
        self,
        env: gym.Env,
        train_data: GoalConditionedTrainingData,  # type: ignore[override]
    ):
        """Train the policy."""
        print(f"\nTraining goal-conditioned RL policy ({self.config.algorithm})...")

        self.node_states = train_data.node_states
        self.valid_shortcuts = train_data.valid_shortcuts
        self.node_atoms = train_data.node_atoms
        print(
            f"Using {len(self.node_states)} node states, {len(self.valid_shortcuts)} valid shortcuts, and {len(self.node_atoms)} node atoms from training data"  # pylint: disable=line-too-long
        )
        goal_env = self._get_goal_env(env)
        use_atom_as_obs = goal_env.use_atom_as_obs

        # Use our custom buffer with node states
        replay_buffer_kwargs = {
            "node_states": self.node_states,
            "valid_shortcuts": self.valid_shortcuts,
            "n_sampled_goal": self.config.n_sampled_goal,
            "using_atom_as_obs": use_atom_as_obs,
        }
        if use_atom_as_obs:
            replay_buffer_kwargs["atom_to_index"] = goal_env.atom_to_index
            replay_buffer_kwargs["atom_vectors"] = goal_env.atom_vectors

        # Set learning_starts to be greater than max episode length for HER
        max_steps = train_data.config.get("max_steps", 50)
        learning_starts = max_steps * 2

        if self.config.algorithm == "SAC":
            self.model = SAC(
                "MultiInputPolicy",
                env,
                replay_buffer_class=NodeBasedHerBuffer,
                replay_buffer_kwargs=replay_buffer_kwargs,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                buffer_size=self.config.buffer_size,
                learning_starts=learning_starts,
                device=self.device,
                seed=self._seed,
                verbose=1,
            )
        elif self.config.algorithm == "TD3":
            assert env.action_space.shape is not None
            action_dim = env.action_space.shape[0]
            action_noise = NormalActionNoise(
                mean=np.zeros(action_dim),
                sigma=self.config.action_noise * np.ones(action_dim),
            )
            self.model = TD3(
                "MultiInputPolicy",
                env,
                replay_buffer_class=NodeBasedHerBuffer,
                replay_buffer_kwargs=replay_buffer_kwargs,
                action_noise=action_noise,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                buffer_size=self.config.buffer_size,
                learning_starts=learning_starts,
                device=self.device,
                seed=self._seed,
                verbose=1,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")

        episodes_per_scenario = train_data.config.get("episodes_per_scenario", 1)
        max_steps = train_data.config.get("max_steps", 50)
        total_timesteps = len(self.valid_shortcuts) * episodes_per_scenario * max_steps
        print(f"Training for {total_timesteps} timesteps...")

        callback = HERTrainingProgressCallback(
            check_freq=max_steps
            * train_data.config.get("training_record_interval", 100)
        )
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

        with open(f"{path}/node_states.pkl", "wb") as f:
            pickle.dump(self.node_states, f)
        with open(f"{path}/valid_shortcuts.pkl", "wb") as f:
            pickle.dump(self.valid_shortcuts, f)

    def load(self, path: str) -> None:
        """Load policy."""
        # Create a dummy env that matches the saved model's spaces
        # But after loading we will use real observations
        obs_space_path = f"{path}/model_observation_space.pkl"
        action_space_path = f"{path}/model_action_space.pkl"
        with open(obs_space_path, "rb") as f:
            observation_space = pickle.load(f)
        with open(action_space_path, "rb") as f:
            action_space = pickle.load(f)

        class DummyEnv(gym.Env):  # pylint: disable=abstract-method
            """Dummy environment to load the model."""

            def __init__(self, observation_space, action_space):
                self.observation_space = observation_space
                self.action_space = action_space

            def compute_reward(
                self, achieved_goal, _desired_goal, _info, _indices=None
            ):
                """Compute reward (return zeros of the right shape)."""
                if isinstance(achieved_goal, np.ndarray):
                    return np.zeros(achieved_goal.shape[0], dtype=np.float32)
                return 0.0

            def reset(self, **_kwargs):
                if isinstance(self.observation_space, spaces.Dict):
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

        dummy_env = DummyEnv(observation_space, action_space)  # type: ignore[no-untyped-call]  # pylint: disable=line-too-long

        if self.config.algorithm == "SAC":
            self.model = SAC.load(f"{path}/model", env=dummy_env, device=self.device)
        elif self.config.algorithm == "TD3":
            self.model = TD3.load(f"{path}/model", env=dummy_env, device=self.device)
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")

        with open(f"{path}/node_states.pkl", "rb") as f:
            self.node_states = pickle.load(f)
        with open(f"{path}/valid_shortcuts.pkl", "rb") as f:
            self.valid_shortcuts = pickle.load(f)

    def _get_goal_env(self, env: gym.Env) -> GoalConditionedWrapper:
        """Get the goal-conditioned environment."""
        current_env = env
        while hasattr(current_env, "env"):
            if isinstance(current_env, GoalConditionedWrapper):
                return current_env
            current_env = current_env.env
        raise ValueError(
            "GoalConditionedWrapper not found when using GoalConditionedRLPolicy."
        )
