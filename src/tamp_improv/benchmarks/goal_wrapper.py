"""Goal-conditioned wrapper for learning shortcuts in TAMP."""

from typing import Any, TypeVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from relational_structs import GroundAtom
from task_then_motion_planning.structs import Perceiver

from tamp_improv.approaches.improvisational.policies.base import (
    GoalConditionedTrainingData,
)

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class GoalConditionedWrapper(gym.Wrapper):
    """Wrapper that converts an environment to a goal-conditioned format."""

    def __init__(
        self,
        env: gym.Env,
        node_states: dict[int, list[ObsType]],
        valid_shortcuts: list[tuple[int, int]],
        perceiver: Perceiver | None = None,
        node_atoms: dict[int, set[GroundAtom]] | None = None,
        max_atom_size: int = 12,
        use_atom_as_obs: bool = True,
        success_threshold: float = 0.01,
        success_reward: float = 10.0,
        step_penalty: float = -0.5,
        max_episode_steps: int = 50,
    ):
        """Initialize wrapper with node states."""
        super().__init__(env)
        self.node_states = node_states
        self.valid_shortcuts = valid_shortcuts or []
        self.perceiver = perceiver
        self.node_atoms = node_atoms or {}
        self.use_atom_as_obs = use_atom_as_obs
        self.max_atom_size = max_atom_size
        self.success_threshold = success_threshold
        self.success_reward = success_reward
        self.step_penalty = step_penalty
        self.max_episode_steps = max_episode_steps
        self.steps = 0

        if self.use_atom_as_obs and self.node_atoms is not None:
            assert (
                self.perceiver is not None
            ), "Perceiver must be provided when using atoms as observations"
            self.atom_to_index: dict[str, int] = {}
            self._next_index = 0

            self.atom_vectors: dict[int, np.ndarray] = {}
            for node_id, atoms in self.node_atoms.items():
                self.atom_vectors[node_id] = self.create_atom_vector(atoms)

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

        else:
            base_obs_space = env.observation_space
            if hasattr(env.observation_space, "node_space"):
                assert len(node_states) > 0, "Node states must not be empty"
                first_node_id = next(iter(node_states.keys()))
                first_state = (
                    node_states[first_node_id][0]
                    if isinstance(node_states[first_node_id], list)
                    else node_states[first_node_id]
                )
                if hasattr(first_state, "nodes"):
                    flattened_size = first_state.nodes.flatten().shape[0]
                else:
                    flattened_size = len(np.array(first_state).flatten())
                base_obs_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(flattened_size,), dtype=np.float32
                )
            self.observation_space = gym.spaces.Dict(
                {
                    "observation": base_obs_space,
                    "achieved_goal": base_obs_space,
                    "desired_goal": base_obs_space,
                }
            )

        self.current_node_id: int | None = None
        self.goal_node_id: int | None = None
        self.goal_state: ObsType | None = None
        self.goal_atom_vector: np.ndarray | None = None
        self.node_ids = sorted(list(node_states.keys()))

    def configure_training(self, train_data: GoalConditionedTrainingData) -> None:
        """Configure environment for training (for compatibility)."""
        assert hasattr(train_data, "node_states") and train_data.node_states is not None
        assert (
            hasattr(train_data, "valid_shortcuts")
            and train_data.valid_shortcuts is not None
        )
        self.node_states = train_data.node_states
        self.valid_shortcuts = train_data.valid_shortcuts
        self.node_atoms = train_data.node_atoms or {}
        self.max_episode_steps = train_data.config.get(
            "max_steps", self.max_episode_steps
        )

    def flatten_obs(self, obs: ObsType) -> np.ndarray:
        """Flatten graph observation for stable-baselines3."""
        if hasattr(obs, "nodes"):
            flattened = obs.nodes.flatten()
            assert isinstance(self.observation_space, spaces.Dict)
            assert self.observation_space["observation"].shape is not None
            expected_size = self.observation_space["observation"].shape[0]
            if len(flattened) < expected_size:
                padded = np.zeros(expected_size, dtype=np.float32)
                padded[: len(flattened)] = flattened
                return padded
            if len(flattened) > expected_size:
                return flattened[:expected_size]
            return flattened
        return np.array(obs, dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset environment and sample a goal."""
        self.steps = 0
        options = options or {}
        self.current_node_id = options.get("source_node_id")
        self.goal_node_id = options.get("goal_node_id")

        self._sample_valid_nodes()
        assert self.current_node_id in self.node_states, "Invalid source node ID"
        available_states = self.node_states[self.current_node_id]
        random_idx = np.random.randint(0, len(available_states))
        current_state = available_states[random_idx]

        if hasattr(self.env, "reset_from_state"):
            original_obs, info = self.env.reset_from_state(current_state, seed=seed)
        else:
            raise AttributeError(
                "The environment does not have a 'reset_from_state' method."
            )
        assert self.current_node_id is not None and self.goal_node_id is not None
        self.goal_state = self.node_states[self.goal_node_id][0]

        info.update(
            {
                "source_node_id": self.current_node_id,
                "goal_node_id": self.goal_node_id,
            }
        )

        if self.use_atom_as_obs:
            self.goal_atom_vector = self.atom_vectors[self.goal_node_id]
            current_atom_vector = self._get_current_atom_vector(original_obs)
            dict_obs = {
                "observation": self.flatten_obs(original_obs),
                "achieved_goal": current_atom_vector,
                "desired_goal": self.goal_atom_vector,
            }
        else:
            flattened_goal = self.flatten_obs(self.goal_state)
            dict_obs = {
                "observation": self.flatten_obs(original_obs),
                "achieved_goal": self.flatten_obs(original_obs),
                "desired_goal": flattened_goal,
            }

        return dict_obs, info

    def step(
        self, action: ActType
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Step environment and compute goal-conditioned rewards."""
        original_next_obs, _, terminated, truncated, info = self.env.step(action)
        self.steps += 1

        if self.use_atom_as_obs and self.goal_atom_vector is not None:
            current_atom_vector = self._get_current_atom_vector(original_next_obs)
            goal_indices = np.where(self.goal_atom_vector > 0.5)[0]
            goal_achieved = np.all(current_atom_vector[goal_indices] > 0.5)
            atoms_distance = np.sum(current_atom_vector[goal_indices] < 0.5)
            info.update(
                {
                    "atoms_distance": atoms_distance,
                    "is_success": goal_achieved,
                    "source_node_id": self.current_node_id,
                    "goal_node_id": self.goal_node_id,
                }
            )
            dict_obs = {
                "observation": self.flatten_obs(original_next_obs),
                "achieved_goal": current_atom_vector,
                "desired_goal": self.goal_atom_vector,
            }
        else:
            goal_distance = np.linalg.norm(
                self.flatten_obs(original_next_obs) - self.flatten_obs(self.goal_state)
            )
            goal_achieved = goal_distance < self.success_threshold
            info.update(
                {
                    "goal_distance": goal_distance,
                    "is_success": goal_achieved,
                    "source_node_id": self.current_node_id,
                    "goal_node_id": self.goal_node_id,
                }
            )
            dict_obs = {
                "observation": self.flatten_obs(original_next_obs),
                "achieved_goal": self.flatten_obs(original_next_obs),
                "desired_goal": self.flatten_obs(self.goal_state),
            }

        goal_reward = self.success_reward if goal_achieved else self.step_penalty
        goal_terminated = bool(goal_achieved)
        truncated = truncated or (self.steps >= self.max_episode_steps)
        return dict_obs, goal_reward, goal_terminated or terminated, truncated, info

    def compute_reward(
        self,
        achieved_goal: ObsType,
        desired_goal: ObsType,
        info: list[dict[str, Any]] | dict[str, Any],
        _indices: list[int] | None = None,
    ) -> np.ndarray:
        """Compute the reward for achieving a given goal."""
        if self.use_atom_as_obs:
            assert hasattr(achieved_goal, "shape")
            assert hasattr(
                desired_goal, "__getitem__"
            ), "desired_goal must be indexable"
            assert hasattr(
                achieved_goal, "__getitem__"
            ), "achieved_goal must be indexable"
            rewards = np.zeros(achieved_goal.shape[0])
            success = np.zeros(achieved_goal.shape[0], dtype=np.bool_)

            for i in range(achieved_goal.shape[0]):
                goal_indices = np.where(desired_goal[i] > 0.5)[0]
                goal_satisfied = np.all(achieved_goal[i][goal_indices] > 0.5)
                rewards[i] = (
                    self.success_reward if goal_satisfied else self.step_penalty
                )
                success[i] = goal_satisfied

            if isinstance(info, list):
                for i, info_dict in enumerate(info):
                    if i < len(success):
                        info_dict["is_success"] = bool(success[i])
            elif isinstance(info, dict):
                if len(success) > 0:
                    info["is_success"] = bool(success[0])
        else:
            distance = np.linalg.norm(
                np.array(achieved_goal) - np.array(desired_goal), axis=-1
            )
            rewards = np.where(
                distance < self.success_threshold,
                self.success_reward,
                self.step_penalty,
            )
            if isinstance(info, list):
                for i, info_dict in enumerate(info):
                    if i < len(distance):
                        info_dict["is_success"] = bool(
                            distance[i] < self.success_threshold
                        )
            elif isinstance(info, dict):
                if isinstance(distance, np.ndarray) and len(distance) > 0:
                    info["is_success"] = bool(distance[0] < self.success_threshold)
                else:
                    info["is_success"] = bool(distance < self.success_threshold)
        return rewards

    def _sample_valid_nodes(self) -> None:
        """Sample valid source and goal nodes."""
        assert self.node_ids
        if self.current_node_id is None:
            source_nodes = set(source_id for source_id, _ in self.valid_shortcuts)
            self.current_node_id = np.random.choice(list(source_nodes))
        if self.goal_node_id is None and self.current_node_id is not None:
            valid_targets = [
                target_id
                for source_id, target_id in self.valid_shortcuts
                if source_id == self.current_node_id
            ]
            assert (
                valid_targets
            ), "No valid target nodes found for the current source node"
            self.goal_node_id = np.random.choice(valid_targets)

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

    def _get_current_atom_vector(self, obs: np.ndarray) -> np.ndarray:
        """Get the multi-hot vector for the current atoms."""
        assert self.perceiver is not None
        atoms = self.perceiver.step(obs)
        return self.create_atom_vector(atoms)
