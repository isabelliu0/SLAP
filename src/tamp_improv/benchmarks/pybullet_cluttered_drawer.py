"""ClutteredDrawer environment implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium.spaces import GraphInstance
from numpy.typing import NDArray
from pybullet_blocks.envs.cluttered_drawer_env import (
    ClutteredDrawerPyBulletObjectsEnv,
    ClutteredDrawerSceneDescription,
)
from pybullet_blocks.planning_models.action import OPERATORS_DRAWER, SKILLS_DRAWER
from pybullet_blocks.planning_models.perception import (
    DRAWER_PREDICATES,
    TYPES,
    ClutteredDrawerPyBulletObjectsPerceiver,
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
class ClutteredDrawerPredicates:
    """Container for ClutteredDrawer predicates."""

    def __getitem__(self, key: str) -> Any:
        """Get predicate by name."""
        return next(p for p in DRAWER_PREDICATES if p.name == key)

    def as_set(self) -> set[Predicate]:
        """Convert to set of predicates."""
        return set(DRAWER_PREDICATES)


class BaseClutteredDrawerTAMPSystem(BaseTAMPSystem[GraphInstance, NDArray[np.float32]]):
    """Base TAMP system for cluttered drawer environment."""

    def __init__(
        self,
        planning_components: PlanningComponents[GraphInstance],
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize cluttered drawer TAMP system."""
        self._render_mode = render_mode
        super().__init__(
            planning_components, name="ClutteredDrawerTAMPSystem", seed=seed
        )

    def _create_env(self) -> gym.Env:
        """Create base environment."""
        scene_description = ClutteredDrawerSceneDescription(
            num_drawer_objects=4,
        )
        return ClutteredDrawerPyBulletObjectsEnv(
            scene_description=scene_description,
            render_mode=self._render_mode,
            use_gui=False,
        )

    def _get_domain_name(self) -> str:
        """Get domain name."""
        return "cluttered-drawer-domain"

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
    ) -> BaseClutteredDrawerTAMPSystem:
        """Factory method for creating system with default components."""
        scene_description = ClutteredDrawerSceneDescription(
            num_drawer_objects=4,
        )
        sim = ClutteredDrawerPyBulletObjectsEnv(
            scene_description=scene_description,
            render_mode=render_mode,
            use_gui=False,
        )
        pybullet_skills = {
            s(sim, max_motion_planning_time=0.1)  # type:ignore[abstract]
            for s in SKILLS_DRAWER
        }
        skills: set[Skill[GraphInstance, NDArray[np.float32]]] = cast(
            set[Skill[GraphInstance, NDArray[np.float32]]], pybullet_skills
        )
        perceiver = ClutteredDrawerPyBulletObjectsPerceiver(sim)
        predicates = ClutteredDrawerPredicates()
        system = cls(
            PlanningComponents(
                types=set(TYPES),
                predicate_container=predicates,
                operators=set(OPERATORS_DRAWER),
                skills=skills,
                perceiver=perceiver,
            ),
            seed=seed,
            render_mode=render_mode,
        )
        return system


class ClutteredDrawerTAMPSystem(
    ImprovisationalTAMPSystem[GraphInstance, NDArray[np.float32]],
    BaseClutteredDrawerTAMPSystem,
):
    """TAMP system for cluttered drawer environment with improvisational policy
    learning enabled."""

    def __init__(
        self,
        planning_components: PlanningComponents[GraphInstance],
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize cluttered drawer TAMP system."""
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
    ) -> ClutteredDrawerTAMPSystem:
        """Factory method for creating system with default components."""
        scene_description = ClutteredDrawerSceneDescription(
            num_drawer_objects=4,
        )
        sim = ClutteredDrawerPyBulletObjectsEnv(
            scene_description=scene_description,
            render_mode=render_mode,
            use_gui=False,
        )
        pybullet_skills = {
            s(sim, max_motion_planning_time=0.1)  # type:ignore[abstract]
            for s in SKILLS_DRAWER
        }
        skills: set[Skill[GraphInstance, NDArray[np.float32]]] = cast(
            set[Skill[GraphInstance, NDArray[np.float32]]], pybullet_skills
        )
        perceiver = ClutteredDrawerPyBulletObjectsPerceiver(sim)
        predicates = ClutteredDrawerPredicates()
        system = cls(
            PlanningComponents(
                types=set(TYPES),
                predicate_container=predicates,
                operators=set(OPERATORS_DRAWER),
                skills=skills,
                perceiver=perceiver,
            ),
            seed=seed,
            render_mode=render_mode,
        )
        return system
