"""Goal-conditioned wrapper for SAC+HER baseline."""

from typing import Any, TypeVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from relational_structs import GroundAtom
from task_then_motion_planning.structs import Perceiver

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class SACHERWrapper(gym.Wrapper):
    """Wrapper that converts environment to goal-conditioned format for SAC+HER
    baseline."""

    def __init__(
        self,
        env: gym.Env,
        perceiver: Perceiver,
        goal_atoms: set[GroundAtom],
        max_atom_size: int = 50,
        max_episode_steps: int = 500,
        success_threshold: float = 0.01,
        success_reward: float = 100.0,
        step_penalty: float = -1.0,
    ):
        """Initialize wrapper."""
        super().__init__(env)
        self.perceiver = perceiver
        self.goal_atoms = goal_atoms
        self.max_atom_size = max_atom_size
        self.max_episode_steps = max_episode_steps
        self.success_threshold = success_threshold
        self.success_reward = success_reward
        self.step_penalty = step_penalty
        self.steps = 0

        # Initialize atom to index mapping
        self.atom_to_index: dict[str, int] = {}
        self._next_index = 0

        # Create goal-conditioned observation space
        base_obs_space = env.observation_space
        if hasattr(base_obs_space, "node_space"):
            sample_obs = base_obs_space.sample()
            flattened_size = sample_obs.nodes.flatten().shape[0]
            base_obs_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(flattened_size,), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(
            {
                "observation": base_obs_space,
                "achieved_goal": gym.spaces.Box(
                    0, 1, shape=(max_atom_size,), dtype=np.float32
                ),
                "desired_goal": gym.spaces.Box(
                    0, 1, shape=(max_atom_size,), dtype=np.float32
                ),
            }
        )

        self.goal_atom_vector = self.create_atom_vector(goal_atoms)

    def reset(
        self, *, seed: int | None = None, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset environment."""
        self.steps = 0
        obs, info = self.env.reset(seed=seed, **kwargs)

        _, atoms, _ = self.perceiver.reset(obs, info)
        current_atom_vector = self.create_atom_vector(atoms)

        dict_obs = {
            "observation": self.flatten_obs(obs),
            "achieved_goal": current_atom_vector,
            "desired_goal": self.goal_atom_vector,
        }

        return dict_obs, info

    def step(
        self, action: ActType
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Step environment."""
        obs, _, terminated, truncated, info = self.env.step(action)
        self.steps += 1

        atoms = self.perceiver.step(obs)
        current_atom_vector = self.create_atom_vector(atoms)

        goal_indices = np.where(self.goal_atom_vector > 0.5)[0]
        goal_achieved = np.all(current_atom_vector[goal_indices] > 0.5)
        atoms_distance = np.sum(current_atom_vector[goal_indices] < 0.5)

        reward = self.success_reward if goal_achieved else self.step_penalty

        info.update(
            {
                "is_success": goal_achieved,
                "atoms_distance": atoms_distance,
            }
        )

        dict_obs = {
            "observation": self.flatten_obs(obs),
            "achieved_goal": current_atom_vector,
            "desired_goal": self.goal_atom_vector,
        }

        truncated = truncated or (self.steps >= self.max_episode_steps)

        return (
            dict_obs,
            reward,
            bool(terminated or goal_achieved),
            bool(truncated),
            info,
        )

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        _info: dict[str, Any],
        _indices=None,
    ) -> np.ndarray:
        """Compute reward for HER."""
        if achieved_goal.ndim == 1:
            achieved_goal = achieved_goal.reshape(1, -1)
        if desired_goal.ndim == 1:
            desired_goal = desired_goal.reshape(1, -1)

        rewards = np.zeros(achieved_goal.shape[0], dtype=np.float32)

        for i in range(achieved_goal.shape[0]):
            goal_indices = np.where(desired_goal[i] > 0.5)[0]
            goal_satisfied = np.all(achieved_goal[i][goal_indices] > 0.5)
            rewards[i] = self.success_reward if goal_satisfied else self.step_penalty

        return rewards

    def flatten_obs(self, obs: ObsType) -> np.ndarray:
        """Flatten observation."""
        if hasattr(obs, "nodes"):
            flattened = obs.nodes.flatten().astype(np.float32)
            assert isinstance(self.observation_space, spaces.Dict)
            obs_space = self.observation_space["observation"]
            assert isinstance(obs_space, spaces.Box)
            obs_shape = obs_space.shape
            assert obs_shape is not None, "Observation space shape cannot be None"
            expected_size = obs_shape[0]
            if len(flattened) < expected_size:
                padded = np.zeros(expected_size, dtype=np.float32)
                padded[: len(flattened)] = flattened
                return padded
            if len(flattened) > expected_size:
                return flattened[:expected_size]
            return flattened
        return np.array(obs, dtype=np.float32)

    def _get_atom_index(self, atom_str: str) -> int:
        """Get a unique index for this atom."""
        if atom_str in self.atom_to_index:
            return self.atom_to_index[atom_str]
        assert (
            self._next_index < self.max_atom_size
        ), f"No more space for new atom at index {self._next_index}. Increase max_atom_size (currently {self.max_atom_size})."  # pylint: disable=line-too-long
        idx = self._next_index
        self.atom_to_index[atom_str] = idx
        self._next_index += 1
        return idx

    def create_atom_vector(self, atoms: set[GroundAtom]) -> np.ndarray:
        """Create a multi-hot vector representation of the set of atoms."""
        vector = np.zeros(self.max_atom_size, dtype=np.float32)
        for atom in atoms:
            idx = self._get_atom_index(str(atom))
            vector[idx] = 1.0
        return vector
