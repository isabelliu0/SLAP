"""General wrapper for environments supporting improvisational policies."""

from typing import Any, TypeVar

import gymnasium as gym
import numpy as np
from relational_structs import GroundAtom
from task_then_motion_planning.structs import Perceiver

from tamp_improv.approaches.improvisational.policies.base import TrainingData

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class ImprovWrapper(gym.Env):
    """General wrapper for training improvisational policies."""

    def __init__(
        self,
        base_env: gym.Env,
        perceiver: Perceiver[ObsType],
        max_episode_steps: int = 100,
        *,
        step_penalty: float = -0.1,
        achievement_bonus: float = 1.0,
        action_scale: float = 1.0,
    ) -> None:
        """Initialize wrapper with environment and perceiver."""
        self.env = base_env
        self.observation_space = base_env.observation_space
        self.action_scale = action_scale
        if isinstance(base_env.action_space, gym.spaces.Box):
            self.action_space = gym.spaces.Box(
                low=base_env.action_space.low * action_scale,
                high=base_env.action_space.high * action_scale,
                dtype=np.float32,
            )
        else:
            print("Warning: Action space is not Box, using original action space.")
            self.action_space = base_env.action_space
        self.max_episode_steps = max_episode_steps
        self.steps = 0
        self.perceiver = perceiver

        self.step_penalty = step_penalty
        self.achievement_bonus = achievement_bonus

        self.training_states: list[ObsType] = []
        self.current_atoms_list: list[set[GroundAtom]] = []
        self.goal_atoms_list: list[set[GroundAtom]] = []
        self.current_atom_set: set[GroundAtom] = set()
        self.goal_atom_set: set[GroundAtom] = set()
        self.current_training_idx: int = 0

        self.relevant_objects = None
        self.render_mode = base_env.render_mode

    def configure_training(
        self,
        training_data: TrainingData,
    ) -> None:
        """Configure environment for training phase."""
        self.training_states = training_data.states

        self.current_atoms_list = training_data.current_atoms
        self.goal_atoms_list = training_data.goal_atoms
        self.current_atom_set = (
            self.current_atoms_list[0] if self.current_atoms_list else set()
        )
        self.goal_atom_set = self.goal_atoms_list[0] if self.goal_atoms_list else set()

        self.current_training_idx = 0
        self.max_episode_steps = training_data.config.get(
            "max_training_steps_per_shortcut", self.max_episode_steps
        )

    def set_relevant_objects(self, objects):
        """Set relevant objects for observation extraction."""
        self.relevant_objects = objects

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset environment."""
        self.steps = 0

        if self.training_states:
            self.current_training_idx = self.current_training_idx % len(
                self.training_states
            )
            current_state = self.training_states[self.current_training_idx]

            self.current_atom_set = self.current_atoms_list[self.current_training_idx]
            self.goal_atom_set = self.goal_atoms_list[self.current_training_idx]

            if hasattr(self.env, "reset_from_state"):
                obs, info = self.env.reset_from_state(current_state, seed=seed)
            else:
                raise AttributeError(
                    "The environment does not have a 'reset_from_state' method."
                )

            if self.relevant_objects is not None:
                assert hasattr(self.env, "extract_relevant_object_features")  # type: ignore[unreachable]   # pylint: disable=line-too-long
                obs = self.env.extract_relevant_object_features(
                    obs, self.relevant_objects
                )

            self.current_training_idx += 1
            return obs, info

        return self.env.reset(seed=seed, options=options)

    def step(
        self,
        action: ActType,
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Step environment."""
        obs, _, _, truncated, info = self.env.step(action)
        self.steps += 1
        current_atoms = self.perceiver.step(obs)

        if self.relevant_objects is not None:
            assert hasattr(self.env, "extract_relevant_object_features")  # type: ignore[unreachable]   # pylint: disable=line-too-long
            obs = self.env.extract_relevant_object_features(obs, self.relevant_objects)

        achieved = self.goal_atom_set == current_atoms

        reward = self.step_penalty
        if achieved:
            reward += self.achievement_bonus

        terminated = achieved
        truncated = truncated or self.steps >= self.max_episode_steps

        return obs, reward, terminated, truncated, info

    def render(self) -> Any:
        """Render the environment."""
        return self.env.render()


class PureRLWrapper(gym.Wrapper):
    """Wrapper for training pure RL baselines without TAMP structure."""

    def __init__(
        self,
        env: gym.Env,
        perceiver: Perceiver[ObsType],
        goal_atoms: set[GroundAtom],
        *,
        max_episode_steps: int = 100,
        step_penalty: float = -0.1,
        achievement_bonus: float = 1.0,
        action_scale: float = 1.0,
    ) -> None:
        """Initialize wrapper for pure RL training."""
        super().__init__(env)
        self.perceiver = perceiver
        self.goal_atoms = goal_atoms
        self.max_episode_steps = max_episode_steps
        self.step_penalty = step_penalty
        self.achievement_bonus = achievement_bonus
        self.action_scale = action_scale
        self.steps = 0
        if isinstance(env.action_space, gym.spaces.Box):
            self.action_space = gym.spaces.Box(
                low=env.action_space.low * action_scale,
                high=env.action_space.high * action_scale,
                dtype=np.float32,
            )
        else:
            print("Warning: Action space is not Box, using original action space.")
            self.action_space = env.action_space
        self._render_mode = getattr(env, "render_mode", None)

    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        """Reset the environment."""
        self.steps = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Take a step in the environment."""
        obs, _, _, truncated, info = self.env.step(action)
        self.steps += 1

        current_atoms = self.perceiver.step(obs)
        achieved = self.goal_atoms.issubset(current_atoms)

        reward = self.step_penalty
        if achieved:
            reward += self.achievement_bonus

        terminated = achieved
        truncated = truncated or self.steps >= self.max_episode_steps

        return obs, reward, terminated, truncated, info
