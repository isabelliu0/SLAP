"""CleanupTable environment implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium.spaces import GraphInstance
from numpy.typing import NDArray
from pybullet_blocks.envs.cleanup_table_env import (
    CleanupTablePyBulletObjectsEnv,
    CleanupTableSceneDescription,
)
from pybullet_blocks.planning_models.action import OPERATORS_CLEANUP, SKILLS_CLEANUP
from pybullet_blocks.planning_models.perception import (
    CLEANUP_PREDICATES,
    TYPES,
    CleanupTablePyBulletObjectsPerceiver,
)
from relational_structs import PDDLDomain, Predicate
from task_then_motion_planning.structs import Skill

from tamp_improv.benchmarks.base import (
    BaseTAMPSystem,
    ImprovisationalTAMPSystem,
    PlanningComponents,
)
from tamp_improv.benchmarks.wrappers import ImprovWrapper


@dataclass(frozen=True)
class CleanupTablePredicates:
    """Container for CleanupTable predicates."""

    def __getitem__(self, key: str) -> Any:
        """Get predicate by name."""
        return next(p for p in CLEANUP_PREDICATES if p.name == key)

    def as_set(self) -> set[Predicate]:
        """Convert to set of predicates."""
        return set(CLEANUP_PREDICATES)


class BaseCleanupTableTAMPSystem(BaseTAMPSystem[GraphInstance, NDArray[np.float32]]):
    """Base TAMP system for cleanup table environment."""

    def __init__(
        self,
        planning_components: PlanningComponents[GraphInstance],
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize cleanup table TAMP system."""
        self._render_mode = render_mode
        super().__init__(planning_components, name="CleanupTableTAMPSystem", seed=seed)

    def _create_env(self) -> gym.Env:
        """Create base environment."""
        scene_description = CleanupTableSceneDescription(
            num_toys=3,
        )
        return CleanupTablePyBulletObjectsEnv(
            scene_description=scene_description,
            render_mode=self._render_mode,
            use_gui=False,
        )

    def _get_domain_name(self) -> str:
        """Get domain name."""
        return "cleanup-table-domain"

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
    ) -> BaseCleanupTableTAMPSystem:
        """Factory method for creating system with default components."""
        scene_description = CleanupTableSceneDescription(
            num_toys=3,
        )
        sim = CleanupTablePyBulletObjectsEnv(
            scene_description=scene_description,
            render_mode=render_mode,
            use_gui=False,
        )
        pybullet_skills = {
            s(sim, max_motion_planning_time=0.1)  # type:ignore[abstract]
            for s in SKILLS_CLEANUP
        }
        skills: set[Skill[GraphInstance, NDArray[np.float32]]] = cast(
            set[Skill[GraphInstance, NDArray[np.float32]]], pybullet_skills
        )
        perceiver = CleanupTablePyBulletObjectsPerceiver(sim)
        predicates = CleanupTablePredicates()
        system = cls(
            PlanningComponents(
                types=set(TYPES),
                predicate_container=predicates,
                operators=set(OPERATORS_CLEANUP),
                skills=skills,
                perceiver=perceiver,
            ),
            seed=seed,
            render_mode=render_mode,
        )
        return system


class CleanupTableTAMPSystem(
    ImprovisationalTAMPSystem[GraphInstance, NDArray[np.float32]],
    BaseCleanupTableTAMPSystem,
):
    """TAMP system for cleanup table environment with improvisational policy
    learning enabled."""

    def __init__(
        self,
        planning_components: PlanningComponents[GraphInstance],
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize cleanup table TAMP system."""
        self._render_mode = render_mode
        super().__init__(planning_components, seed=seed, render_mode=render_mode)

    def _create_wrapped_env(
        self, components: PlanningComponents[GraphInstance]
    ) -> gym.Env:
        """Create wrapped environment for training."""
        return ImprovWrapper(
            base_env=self.env,
            perceiver=components.perceiver,
            step_penalty=-1.0,
            achievement_bonus=100.0,
            action_scale=0.005,
        )

    @classmethod
    def create_default(
        cls,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> CleanupTableTAMPSystem:
        """Factory method for creating system with default components."""
        scene_description = CleanupTableSceneDescription(
            num_toys=3,
        )
        sim = CleanupTablePyBulletObjectsEnv(
            scene_description=scene_description,
            render_mode=render_mode,
            use_gui=False,
        )
        pybullet_skills = {
            s(sim, max_motion_planning_time=0.1)  # type:ignore[abstract]
            for s in SKILLS_CLEANUP
        }
        skills: set[Skill[GraphInstance, NDArray[np.float32]]] = cast(
            set[Skill[GraphInstance, NDArray[np.float32]]], pybullet_skills
        )
        perceiver = CleanupTablePyBulletObjectsPerceiver(sim)
        predicates = CleanupTablePredicates()
        system = cls(
            PlanningComponents(
                types=set(TYPES),
                predicate_container=predicates,
                operators=set(OPERATORS_CLEANUP),
                skills=skills,
                perceiver=perceiver,
            ),
            seed=seed,
            render_mode=render_mode,
        )
        return system
