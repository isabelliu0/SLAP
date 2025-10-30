"""Context-aware environment wrapper for improvisational TAMP."""

from typing import Any

import gymnasium as gym
import numpy as np
from relational_structs import GroundAtom
from task_then_motion_planning.structs import Perceiver

from tamp_improv.approaches.improvisational.policies.base import TrainingData
from tamp_improv.benchmarks.wrappers import ActType, ObsType


class ContextAwareWrapper(gym.Wrapper):
    """Wrapper that augments observations with context information."""

    def __init__(
        self,
        env: gym.Env,
        perceiver: Perceiver[ObsType],
        max_atom_size: int = 12,
        max_episode_steps: int = 50,
        success_reward: float = 10.0,
        step_penalty: float = -0.5,
    ) -> None:
        super().__init__(env)
        self.perceiver: Perceiver[ObsType] = perceiver
        self.max_episode_steps = max_episode_steps
        self.success_reward = success_reward
        self.step_penalty = step_penalty
        self.steps = 0

        self.training_states: list[ObsType] = []
        self.current_atoms_list: list[set[GroundAtom]] = []
        self.goal_atoms_list: list[set[GroundAtom]] = []
        self.current_episode_idx = 0

        self.current_goal_atoms: set[GroundAtom] = set()
        self.atom_to_index: dict[str, int] = {}
        self._next_index = 0

        self.num_context_features = max_atom_size
        if hasattr(env.observation_space, "node_space"):
            sample_obs = env.observation_space.sample()
            base_size = sample_obs.nodes.flatten().shape[0]
        else:
            assert env.observation_space.shape is not None
            base_size = env.observation_space.shape[0]
        total_size = base_size + max_atom_size
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float32
        )
        self.base_obs_size = base_size

    def configure_training(self, train_data: TrainingData) -> None:
        """Configure with training data."""
        self.training_states = train_data.states
        self.current_atoms_list = train_data.current_atoms
        self.goal_atoms_list = train_data.goal_atoms

        if "atom_to_index" in train_data.config and train_data.config["atom_to_index"]:
            self.atom_to_index = train_data.config["atom_to_index"]
            self._next_index = (
                max(self.atom_to_index.values()) + 1 if self.atom_to_index else 0
            )
        else:
            unique_atoms = set()
            for atoms_set in self.current_atoms_list + self.goal_atoms_list:
                for atom in atoms_set:
                    unique_atoms.add(str(atom))
            for atom_str in sorted(unique_atoms):
                self._get_atom_index(atom_str)

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to a random training episode."""
        self.steps = 0

        self.current_episode_idx = np.random.randint(0, len(self.training_states))
        start_state = self.training_states[self.current_episode_idx]
        self.current_goal_atoms = self.goal_atoms_list[self.current_episode_idx]

        if hasattr(self.env, "reset_from_state"):
            original_obs, info = self.env.reset_from_state(start_state, **kwargs)
        else:
            original_obs, info = self.env.reset(**kwargs)

        augmented_obs = self.augment_observation(original_obs)
        return augmented_obs, info

    def step(
        self, action: ActType
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step and check if goal is achieved."""
        original_obs, _, terminated, truncated, info = self.env.step(action)
        self.steps += 1

        current_atoms = self.perceiver.step(original_obs)
        goal_achieved = self.current_goal_atoms.issubset(current_atoms)

        reward = self.success_reward if goal_achieved else self.step_penalty

        info.update(
            {
                "is_success": goal_achieved,
                "goal_atoms": self.current_goal_atoms,
                "current_atoms": current_atoms,
            }
        )

        augmented_obs = self.augment_observation(original_obs)
        truncated = truncated or (self.steps >= self.max_episode_steps)

        return augmented_obs, reward, terminated or goal_achieved, truncated, info

    def augment_observation(self, obs: ObsType) -> np.ndarray:
        """Augment observation with goal context."""
        if hasattr(obs, "nodes"):
            base_obs = obs.nodes.flatten().astype(np.float32)
        else:
            base_obs = np.array(obs, dtype=np.float32).flatten()

        assert len(base_obs) <= self.base_obs_size
        if len(base_obs) < self.base_obs_size:
            padded_obs = np.zeros(self.base_obs_size, dtype=np.float32)
            padded_obs[: len(base_obs)] = base_obs
            base_obs = padded_obs

        context = np.zeros(self.num_context_features, dtype=np.float32)
        for atom in self.current_goal_atoms:
            idx = self._get_atom_index(str(atom))
            context[idx] = 1.0

        return np.concatenate([base_obs, context])

    def _get_atom_index(self, atom_str: str) -> int:
        """Get a unique index for this atom."""
        if atom_str in self.atom_to_index:
            return self.atom_to_index[atom_str]
        assert (
            self._next_index < self.num_context_features
        ), f"Too many for context size {self.num_context_features}"
        idx = self._next_index
        self.atom_to_index[atom_str] = idx
        self._next_index += 1
        return idx
