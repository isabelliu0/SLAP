"""Hierarchical RL wrapper that combines low-level actions with TAMP skills."""

import itertools
from typing import Any, TypeVar, Union

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray
from relational_structs import GroundAtom
from task_then_motion_planning.structs import (
    GroundOperator,
    LiftedOperator,
    Object,
    Skill,
)

from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class HierarchicalRLWrapper(gym.Env):
    """Wrapper that enables hierarchical RL by augmenting action space with
    TAMP skills.

    The action space consists of:
    - Original low-level actions (continuous)
    - Skill selection (discrete)

    If a skill is selected, the wrapper executes the skill until completion or failure,
    then returns control to the RL agent.
    """

    def __init__(
        self,
        tamp_system: ImprovisationalTAMPSystem[ObsType, ActType],
        max_episode_steps: int = 100,
        max_skill_steps: int = 50,
        step_penalty: float = -0.1,
        achievement_bonus: float = 10.0,
        action_scale: float = 1.0,
        skill_failure_penalty: float = -1.0,
        single_step_skills: bool = True,
    ) -> None:
        """Initialize hierarchical wrapper."""
        self.tamp_system = tamp_system
        self.env = tamp_system.env
        self.perceiver = tamp_system.perceiver
        self.max_episode_steps = max_episode_steps
        self.max_skill_steps = max_skill_steps
        self.step_penalty = step_penalty
        self.achievement_bonus = achievement_bonus
        self.action_scale = action_scale
        self.skill_failure_penalty = skill_failure_penalty
        self.single_step_skills = single_step_skills

        self.steps = 0
        self.current_obs: Union[ObsType, None] = None
        self.goal_atoms: set[GroundAtom] = set()
        self.current_atoms: set[GroundAtom] = set()

        self.current_skill: Union[Skill, None] = None
        self.current_skill_operator: Union[GroundOperator, None] = None
        self.skill_steps_taken: int = 0

        self.observation_space = self.env.observation_space

        (
            self.ground_skill_operators,
            self.skill_to_index,
        ) = self.get_ground_operator_skills()
        self.skill_names = [op.short_str for op in self.ground_skill_operators]

        self._setup_action_space()

    def get_ground_operator_skills(
        self,
    ) -> tuple[list[GroundOperator], dict[GroundOperator, int]]:
        """Get skills from the TAMP system."""
        operators = self.tamp_system.components.operators
        objects = self.tamp_system.components.perceiver.get_objects()  # type: ignore[attr-defined]  # pylint: disable=line-too-long
        ground_skill_operators = []
        skill_op_to_index = {}
        for lifted_operator in operators:
            # find
            groundings = self._find_valid_groundings(lifted_operator, objects)
            if groundings:
                for grounding in groundings:
                    ground_skill_operator = lifted_operator.ground(grounding)
                    ground_skill_operators.append(ground_skill_operator)
                    skill_op_to_index[ground_skill_operator] = (
                        len(ground_skill_operators) - 1
                    )
        return ground_skill_operators, skill_op_to_index

    def _setup_action_space(self) -> None:
        """Set up the augmented action space."""
        if isinstance(self.env.action_space, Box):
            low_level_dim = (
                self.env.action_space.shape[0]
                if self.env.action_space.shape is not None
                else 0
            )
            low_level_low = self.env.action_space.low * self.action_scale
            low_level_high = self.env.action_space.high * self.action_scale
        else:
            raise ValueError("Base environment must have Box action space")

        num_skills = len(self.ground_skill_operators)
        skill_low = np.zeros(num_skills, dtype=np.float32)
        skill_high = np.ones(num_skills, dtype=np.float32)
        action_low = np.concatenate([low_level_low, skill_low])
        action_high = np.concatenate([low_level_high, skill_high])
        self.action_space = Box(low=action_low, high=action_high, dtype=np.float32)
        print(
            f"Action space: {low_level_dim}D low-level + {num_skills}D skill activations [0,1]"  # pylint: disable=line-too-long
        )

    def reset(
        self,
        *,
        seed: Union[int, None] = None,
        options: Union[dict[str, Any], None] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset environment."""
        self.steps = 0

        obs, info = self.tamp_system.reset(seed=seed)
        self.current_obs = obs

        self.current_skill = None
        self.current_skill_operator = None
        self.skill_steps_taken = 0

        _, current_atoms, goal_atoms = self.perceiver.reset(obs, info)
        self.current_atoms = current_atoms
        self.goal_atoms = goal_atoms

        info.update(
            {
                "goal_atoms": goal_atoms,
                "current_atoms": current_atoms,
                "available_skills": self.ground_skill_operators,
            }
        )

        return obs, info  # type: ignore

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Execute action (either low-level or skill).

        Args:
            action: [low_level_actions..., skill_0, skill_1, ...]
                   If max(skill_activations) > 0.5, use highest skill, else use low-level
        """
        if self.current_obs is None:
            raise RuntimeError("Environment not reset")

        base_action_dim = (
            self.env.action_space.shape[0]
            if self.env.action_space.shape is not None
            else 0
        )
        low_level_action = action[:base_action_dim]
        skill_activations = action[base_action_dim:]

        max_skill_activation = np.max(skill_activations)
        if max_skill_activation > 0.5:
            skill_idx = int(np.argmax(skill_activations))
            obs: ObsType
            obs, reward, terminated, truncated, info = self._execute_skill(skill_idx)
            info["action_type"] = (
                f"skill_{self.ground_skill_operators[skill_idx].short_str}"
            )
            info["skill_activation"] = float(skill_activations[skill_idx])
            info["max_skill_activation"] = float(max_skill_activation)
        else:
            obs, reward, terminated, truncated, info = self._execute_low_level_action(
                low_level_action
            )
            info["action_type"] = "low_level"
            info["max_skill_activation"] = float(max_skill_activation)

        self.current_obs = obs  # type: ignore

        self.current_atoms = self.perceiver.step(obs)  # type: ignore
        goal_achieved = self.goal_atoms.issubset(self.current_atoms)

        if goal_achieved:
            reward += self.achievement_bonus
            terminated = True

        truncated = truncated or self.steps >= self.max_episode_steps

        info.update(
            {
                "goal_achieved": goal_achieved,
                "current_atoms": self.current_atoms,
                "goal_atoms": self.goal_atoms,
                "steps": self.steps,
            }
        )

        return obs, reward, terminated, truncated, info

    def _execute_low_level_action(
        self, action: NDArray[np.float32]
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Execute a low-level action."""
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        reward = float(env_reward) + self.step_penalty
        return obs, reward, terminated, truncated, info

    def _execute_skill(
        self, skill_idx: int
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Execute a TAMP skill either single-step or until completion."""
        if self.single_step_skills:
            return self._execute_skill_single_step(skill_idx)
        return self._execute_skill_multi_step(skill_idx)

    def _execute_skill_single_step(
        self, skill_idx: int
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Execute one step of a skill for better video recording."""
        ground_operator = self.ground_skill_operators[skill_idx]

        if self.current_skill is None or self.current_skill_operator != ground_operator:
            assert self.current_obs is not None
            current_atoms = self.perceiver.step(self.current_obs)
            applicable = ground_operator.preconditions.issubset(current_atoms)

            if not applicable:
                self.steps += 1
                current_obs_typed: ObsType = self.current_obs  # type: ignore
                return (
                    current_obs_typed,
                    self.skill_failure_penalty,
                    False,
                    False,
                    {
                        "skill_applicable": False,
                        "skill_name": ground_operator.name,
                        "reason": "preconditions not met",
                    },
                )

            self.current_skill = self._get_skill(ground_operator)
            self.current_skill.reset(ground_operator)
            self.current_skill_operator = ground_operator
            self.skill_steps_taken = 0

        try:
            skill_action = self.current_skill.get_action(self.current_obs)
            obs, env_reward, terminated, truncated, _ = self.env.step(skill_action)

            self.steps += 1
            self.current_obs = obs
            self.skill_steps_taken += 1
            skill_reward = float(env_reward) + self.step_penalty

            new_atoms = self.perceiver.step(obs)
            add_ok = ground_operator.add_effects.issubset(new_atoms)
            delete_ok = ground_operator.delete_effects.isdisjoint(new_atoms)

            if add_ok and delete_ok:
                skill_steps = self.skill_steps_taken
                self._reset_skill_state()
                return (
                    obs,
                    skill_reward,
                    terminated,
                    truncated,
                    {
                        "skill_applicable": True,
                        "skill_completed": True,
                        "skill_name": ground_operator.name,
                        "skill_steps": skill_steps,
                    },
                )

            if self.skill_steps_taken >= self.max_skill_steps:
                skill_steps = self.skill_steps_taken
                self._reset_skill_state()
                return (
                    obs,
                    skill_reward + self.skill_failure_penalty,
                    terminated,
                    truncated,
                    {
                        "skill_applicable": True,
                        "skill_completed": False,
                        "skill_name": ground_operator.name,
                        "skill_steps": skill_steps,
                        "reason": "timeout",
                    },
                )

            return (
                obs,
                skill_reward,
                terminated,
                truncated,
                {
                    "skill_applicable": True,
                    "skill_completed": False,
                    "skill_name": ground_operator.name,
                    "skill_steps": self.skill_steps_taken,
                    "skill_in_progress": True,
                },
            )

        except Exception as e:
            self.steps += 1
            skill_steps = self.skill_steps_taken
            self._reset_skill_state()
            current_obs_typed: ObsType = self.current_obs  # type: ignore
            return (
                current_obs_typed,
                self.skill_failure_penalty,
                False,
                False,
                {
                    "skill_applicable": True,
                    "skill_completed": False,
                    "skill_name": ground_operator.name,
                    "skill_steps": skill_steps,
                    "error": str(e),
                },
            )

    def _execute_skill_multi_step(
        self, skill_idx: int
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Execute a TAMP skill until completion or failure (original
        behavior)."""
        assert self.current_obs is not None
        current_atoms = self.perceiver.step(self.current_obs)
        ground_operator = self.ground_skill_operators[skill_idx]
        applicable = ground_operator.preconditions.issubset(current_atoms)

        if not applicable:
            self.steps += 1
            current_obs_typed: ObsType = self.current_obs  # type: ignore
            return (
                current_obs_typed,
                self.skill_failure_penalty,
                False,
                False,
                {
                    "skill_applicable": False,
                    "skill_name": self.ground_skill_operators[skill_idx].name,
                    "reason": "preconditions not met",
                },
            )

        skill_steps = 0
        skill_reward = 0.0
        obs: ObsType = self.current_obs  # type: ignore
        skill = self._get_skill(ground_operator)
        skill.reset(ground_operator)

        while skill_steps < self.max_skill_steps:
            try:
                skill_action = skill.get_action(obs)

                obs, env_reward, terminated, truncated, _ = self.env.step(skill_action)
                self.steps += 1
                self.current_obs = obs  # type: ignore
                skill_reward += float(env_reward) + self.step_penalty
                skill_steps += 1

                new_atoms = self.perceiver.step(obs)  # type: ignore

                add_ok = ground_operator.add_effects.issubset(new_atoms)
                delete_ok = ground_operator.delete_effects.isdisjoint(new_atoms)
                if add_ok and delete_ok:
                    return (
                        obs,
                        skill_reward,
                        terminated,
                        truncated,
                        {
                            "skill_applicable": True,
                            "skill_completed": True,
                            "skill_name": self.ground_skill_operators[skill_idx].name,
                            "skill_steps": skill_steps,
                        },
                    )

                if terminated or truncated:
                    break

            except Exception as e:
                self.steps += 1
                return (
                    obs,
                    skill_reward + self.skill_failure_penalty,
                    False,
                    False,
                    {
                        "skill_applicable": True,
                        "skill_completed": False,
                        "skill_name": self.ground_skill_operators[skill_idx].name,
                        "skill_steps": skill_steps,
                        "error": str(e),
                    },
                )

        return (
            obs,
            skill_reward + self.skill_failure_penalty,
            terminated,
            truncated,
            {
                "skill_applicable": True,
                "skill_completed": False,
                "skill_name": self.ground_skill_operators[skill_idx].name,
                "skill_steps": skill_steps,
                "reason": "timeout",
            },
        )

    def _reset_skill_state(self) -> None:
        """Reset skill execution state."""
        self.current_skill = None
        self.current_skill_operator = None
        self.skill_steps_taken = 0

    def _get_skill(self, operator: GroundOperator) -> Skill:
        """Get skill that can execute operator."""
        skills = [s for s in self.tamp_system.skills if s.can_execute(operator)]
        assert skills, f"No skill found for operator {operator.name}"
        return skills[0]

    def _find_valid_groundings(
        self, lifted_op: LiftedOperator, objects: set[Object]
    ) -> list[tuple[Object, ...]]:
        """Find all valid groundings for a lifted operator."""
        objects_by_type: dict[Any, list[Object]] = {}
        for obj in objects:
            if obj.type not in objects_by_type:
                objects_by_type[obj.type] = []
            objects_by_type[obj.type].append(obj)

        param_types = []
        for param in lifted_op.parameters:
            param_types.append(f"{param.name} ({param.type.name})")

        param_objects = []
        for param in lifted_op.parameters:
            if param.type in objects_by_type:
                param_objects.append(objects_by_type[param.type])
            else:
                return []

        groundings = list(itertools.product(*param_objects))

        return groundings

    @property
    def render_mode(self):
        """Get render mode from base environment."""
        return getattr(self.env, "render_mode", None)

    def render(self) -> Any:
        """Render the environment."""
        if hasattr(self.env, "render"):
            return self.env.render()
        return None

    def close(self) -> None:
        """Close the environment."""
        if hasattr(self.env, "close"):
            self.env.close()  # type: ignore[no-untyped-call]
