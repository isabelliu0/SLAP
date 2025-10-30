"""Custom replay buffer for node-based goal sampling."""

from typing import Any, Union

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from tamp_improv.approaches.improvisational.policies.base import ObsType


class NodeBasedHerBuffer(HerReplayBuffer):
    """Custom HER buffer that samples goals from planning graph node states.

    This buffer ensures that:
    1. Goals come from the collection of node states (G)
    2. The node ID of the goal state is larger than the initial source node ID (s_0)
    3. There's no direct non-shortcut edge between source and goal nodes
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        node_states: dict[int, ObsType],
        valid_shortcuts: list[tuple[int, int]],
        using_atom_as_obs: bool,
        atom_to_index: dict[str, int] | None = None,
        atom_vectors: dict[int, np.ndarray] | None = None,
        device: Union[torch.device, str] = "cuda",
        n_sampled_goal: int = 4,
        **kwargs,
    ):
        """Initialize the buffer with node states."""
        self.node_states = node_states or {}
        self.valid_shortcuts = valid_shortcuts or []
        self.node_ids = sorted(list(node_states.keys()))
        self.using_atom_as_obs = using_atom_as_obs
        self.atom_to_index = atom_to_index or {}
        self.atom_vectors = atom_vectors or {}
        self.n_sampled_goal = n_sampled_goal
        self.n_envs = kwargs.get("n_envs", 1)

        self.valid_targets: dict[int, list[int]] = {}
        for source_id, target_id in valid_shortcuts:
            if source_id not in self.valid_targets:
                self.valid_targets[source_id] = []
            self.valid_targets[source_id].append(target_id)

        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_sampled_goal=n_sampled_goal,
            handle_timeout_termination=False,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["node_states", "valid_shortcuts"]
            },
        )

        self.episode_source_ids = np.full((buffer_size,), -1, dtype=np.int32)

    def add(
        self,
        obs: dict[str, np.ndarray],  # type: ignore[override]
        next_obs: dict[str, np.ndarray],  # type: ignore[override]
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """Add a new transition to buffer."""
        is_new_episode = self.pos == 0 or np.any(self.dones[self.pos - 1])
        if is_new_episode:
            assert (
                len(infos) > 0 and "source_node_id" in infos[0]
            ), "Source node ID not found in info dict"
            for env_idx in range(self.n_envs):
                if self.pos == 0 or self.dones[self.pos - 1, env_idx]:
                    episode_idx = self.pos // self.n_envs
                    self.episode_source_ids[episode_idx] = infos[env_idx][
                        "source_node_id"
                    ]
        super().add(obs, next_obs, action, reward, done, infos)

    def _sample_goals(
        self,
        batch_indices: np.ndarray,
        env_indices: np.ndarray,
    ) -> np.ndarray:
        """Override goal sampling to use node states instead of future
        states."""
        assert len(batch_indices) > 0, "Empty batch indices"
        # Determine which episode each transition belongs to
        batch_ep_start = self.ep_start[batch_indices, env_indices]
        episode_indices = batch_ep_start // self.n_envs

        goals = np.zeros(
            (len(batch_indices),) + self.next_observations["achieved_goal"].shape[2:],
            dtype=self.next_observations["achieved_goal"].dtype,
        )

        # For each transition, sample a goal from node states
        for i, ep_idx in enumerate(episode_indices):
            source_node_id = self.episode_source_ids[ep_idx]
            assert source_node_id >= 0, "Source node ID not found in episode source IDs"
            goals[i] = self._sample_node_goal(source_node_id)

        return goals

    def _sample_node_goal(self, source_node_id: int) -> np.ndarray:
        """Sample a goal state from node states."""
        assert (
            source_node_id >= 0 and source_node_id in self.node_ids
        ), "Invalid source node ID"
        if source_node_id in self.valid_targets and self.valid_targets[source_node_id]:
            goal_id = np.random.choice(self.valid_targets[source_node_id])
            if self.using_atom_as_obs:
                return np.array(self.atom_vectors[goal_id])
            # node_states[goal_id] is a list of states, pick one randomly
            goal_states = self.node_states[goal_id]
            if isinstance(goal_states, list) and len(goal_states) > 0:
                random_state = np.random.choice(len(goal_states))
                selected_state = goal_states[random_state]
                if hasattr(selected_state, "nodes"):
                    return selected_state.nodes.flatten().astype(np.float32)
                return np.array(selected_state, dtype=np.float32)
            if hasattr(goal_states, "nodes"):
                return goal_states.nodes.flatten().astype(np.float32)
            return np.array(goal_states, dtype=np.float32)
        raise ValueError(f"No valid targets for source node {source_node_id}!")
