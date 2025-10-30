"""ObstacleTower environment implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from pybullet_blocks.envs.obstacle_tower_env import (
    ObstacleTowerPyBulletObjectsEnv,
    ObstacleTowerSceneDescription,
)
from pybullet_blocks.planning_models.action import OPERATORS, SKILLS
from pybullet_blocks.planning_models.perception import (
    PREDICATES,
    TYPES,
    ObstacleTowerPyBulletObjectsPerceiver,
)
from relational_structs import PDDLDomain
from task_then_motion_planning.structs import Skill

from tamp_improv.benchmarks.base import (
    BaseTAMPSystem,
    ImprovisationalTAMPSystem,
    PlanningComponents,
)
from tamp_improv.benchmarks.wrappers import ImprovWrapper


@dataclass(frozen=True)
class ObstacleTowerPredicates:
    """Container for ObstacleTower predicates."""

    def __getitem__(self, key: str) -> Any:
        """Get predicate by name."""
        return next(p for p in PREDICATES if p.name == key)

    def as_set(self) -> set:
        """Convert to set of predicates."""
        return set(PREDICATES)


class BaseObstacleTowerTAMPSystem(
    BaseTAMPSystem[NDArray[np.float32], NDArray[np.float32]]
):
    """Base TAMP system for ObstacleTower environment."""

    def __init__(
        self,
        planning_components: PlanningComponents[NDArray[np.float32]],
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize ObstacleTower TAMP system."""
        self._render_mode = render_mode
        super().__init__(planning_components, name="ObstacleTowerTAMPSystem", seed=seed)

    def _create_env(self) -> gym.Env:
        """Create base environment."""
        scene_description = ObstacleTowerSceneDescription(
            num_obstacle_blocks=3,
            stack_blocks=True,
        )
        return ObstacleTowerPyBulletObjectsEnv(
            scene_description=scene_description,
            render_mode=self._render_mode,
            use_gui=False,
        )

    def _get_domain_name(self) -> str:
        """Get domain name."""
        return "obstacle-tower-domain"

    def get_domain(self) -> PDDLDomain:
        """Get PDDL domain."""
        return PDDLDomain(
            self._get_domain_name(),
            self.components.operators,
            self.components.predicate_container.as_set(),
            self.components.types,
        )

    @classmethod
    def create_default(
        cls,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> BaseObstacleTowerTAMPSystem:
        """Factory method for creating system with default components."""
        scene_description = ObstacleTowerSceneDescription(
            num_obstacle_blocks=3,
            stack_blocks=True,
        )
        sim = ObstacleTowerPyBulletObjectsEnv(
            scene_description=scene_description,
            render_mode=render_mode,
            use_gui=False,
        )
        pybullet_skills = {
            s(sim, max_motion_planning_time=0.1)  # type:ignore[abstract]
            for s in SKILLS
        }
        skills: set[Skill[NDArray[np.float32], NDArray[np.float32]]] = cast(
            set[Skill[NDArray[np.float32], NDArray[np.float32]]], pybullet_skills
        )
        perceiver = ObstacleTowerPyBulletObjectsPerceiver(sim)
        predicates = ObstacleTowerPredicates()
        system = cls(
            PlanningComponents(
                types=set(TYPES),
                predicate_container=predicates,
                operators=OPERATORS,
                skills=skills,
                perceiver=perceiver,
            ),
            seed=seed,
            render_mode=render_mode,
        )
        return system


class ObstacleTowerTAMPSystem(
    ImprovisationalTAMPSystem[NDArray[np.float32], NDArray[np.float32]],
    BaseObstacleTowerTAMPSystem,
):
    """TAMP system for ObstacleTower environment with improvisational policy
    learning enabled."""

    def __init__(
        self,
        planning_components: PlanningComponents[NDArray[np.float32]],
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize ObstacleTower TAMP system."""
        self._render_mode = render_mode
        super().__init__(planning_components, seed=seed, render_mode=render_mode)

    def _create_wrapped_env(
        self, components: PlanningComponents[NDArray[np.float32]]
    ) -> gym.Env:
        """Create wrapped environment for training."""
        return ImprovWrapper(
            base_env=self.env,
            perceiver=components.perceiver,
            step_penalty=-1.0,
            achievement_bonus=100.0,
            action_scale=0.015,
        )

    @classmethod
    def create_default(
        cls,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> ObstacleTowerTAMPSystem:
        """Factory method for creating improvisational system with default
        components."""
        scene_description = ObstacleTowerSceneDescription(
            num_obstacle_blocks=3,
            stack_blocks=True,
        )
        sim = ObstacleTowerPyBulletObjectsEnv(
            scene_description=scene_description,
            render_mode=render_mode,
            use_gui=False,
        )
        pybullet_skills = {
            s(sim, max_motion_planning_time=0.1)  # type:ignore[abstract]
            for s in SKILLS
        }
        skills: set[Skill[NDArray[np.float32], NDArray[np.float32]]] = cast(
            set[Skill[NDArray[np.float32], NDArray[np.float32]]], pybullet_skills
        )
        perceiver = ObstacleTowerPyBulletObjectsPerceiver(sim)
        predicates = ObstacleTowerPredicates()
        system = cls(
            PlanningComponents(
                types=set(TYPES),
                predicate_container=predicates,
                operators=OPERATORS,
                skills=skills,
                perceiver=perceiver,
            ),
            seed=seed,
            render_mode=render_mode,
        )
        return system
