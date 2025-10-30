"""Obstacle2D environment graph-based implementation."""

from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium.spaces import GraphInstance
from numpy.typing import NDArray
from relational_structs import (
    GroundAtom,
    LiftedOperator,
    Object,
    PDDLDomain,
    Variable,
)
from task_then_motion_planning.structs import Perceiver

from tamp_improv.benchmarks.base import (
    BaseTAMPSystem,
    ImprovisationalTAMPSystem,
    PlanningComponents,
)
from tamp_improv.benchmarks.obstacle2d import (
    BaseObstacle2DSkill,
    Obstacle2DPredicates,
    Obstacle2DTypes,
)
from tamp_improv.benchmarks.obstacle2d_env import (
    GraphObstacle2DEnv,
    is_block_in_target_area,
)
from tamp_improv.benchmarks.wrappers import ImprovWrapper


class GraphPickUpSkill(BaseObstacle2DSkill):
    """Skill for picking up blocks from non-target surfaces."""

    def _get_operator_name(self) -> str:
        return "PickUp"

    def _get_action_given_objects(
        self,
        objects: Sequence[Object],
        obs: GraphInstance,  # type: ignore[override]
    ) -> NDArray[np.float32]:
        robot_pos, block_positions, _ = self._extract_positions_from_graph(obs)
        robot_width, robot_height = 0.2, 0.2
        block_width, block_height = 0.2, 0.2

        # Get which block to pick up
        block_obj = objects[1]
        block_id = int(block_obj.name.replace("block", ""))
        assert block_id in block_positions
        block_pos = block_positions[block_id]

        # Find the other blocks
        other_block_positions = []
        for other_id, other_pos in block_positions.items():
            if other_id != block_id:
                other_block_positions.append(other_pos)

        # If too close to the other block, move away first
        for other_block_pos in other_block_positions:
            if (
                np.isclose(robot_pos[1], other_block_pos[1], atol=1e-3)
                and abs(robot_pos[0] - other_block_pos[0])
                < (robot_width + block_width) / 2
                and not np.isclose(robot_pos[0], other_block_pos[0], atol=1e-3)
            ):
                dx = np.clip(robot_pos[0] - other_block_pos[0], -0.1, 0.1)
                return np.array([dx, 0.0, -1.0])

        # Target position above block
        target_y = block_pos[1] + block_height / 2 + robot_height / 2

        # Move towards y-level of target position first
        if not np.isclose(robot_pos[1], target_y, atol=1e-3):
            dy = np.clip(target_y - robot_pos[1], -0.1, 0.1)
            return np.array([0.0, dy, -1.0])

        # Move towards x-level of target position next
        if not np.isclose(robot_pos[0], block_pos[0], atol=1e-3):
            dx = np.clip(block_pos[0] - robot_pos[0], -0.1, 0.1)
            return np.array([dx, 0.0, -1.0])

        # If aligned and not holding block, pick it up
        return np.array([0.0, 0.0, 1.0])


class GraphPickUpFromTargetSkill(GraphPickUpSkill):
    """Skill for picking up blocks from target area."""

    def _get_operator_name(self) -> str:
        return "PickUpFromTarget"


class GraphPutDownSkill(BaseObstacle2DSkill):
    """Skill for putting down blocks on non-target surfaces."""

    def _get_operator_name(self) -> str:
        return "PutDown"

    def _get_action_given_objects(
        self,
        objects: Sequence[Object],
        obs: GraphInstance,  # type: ignore[override]
    ) -> NDArray[np.float32]:
        robot_pos, _, gripper_status = self._extract_positions_from_graph(obs)
        robot_height = 0.2
        block_height = 0.2

        # Get which surface to place on and its position
        surface_obj = objects[2]
        if surface_obj.name == "target_area":
            target_x, target_y = 0.5, 0.0
        else:
            target_x, target_y = 0.0, 0.0

        # Check if block is already on the target surface
        if np.isclose(robot_pos[0], target_x, atol=1e-3) and np.isclose(
            robot_pos[1], target_y, atol=1e-3
        ):
            if gripper_status > 0.0:
                return np.array([0.0, 0.0, -1.0])
            return np.array([0.0, 0.0, 0.0])

        # Target position above drop location
        target_y = target_y + block_height / 2 + robot_height / 2

        # Move towards y-level of target position first
        if not np.isclose(robot_pos[1], target_y, atol=1e-3):
            dy = np.clip(target_y - robot_pos[1], -0.1, 0.1)
            return np.array([0.0, dy, gripper_status])

        # Move towards x-level of target position next
        if not np.isclose(robot_pos[0], target_x, atol=1e-3):
            dx = np.clip(target_x - robot_pos[0], -0.1, 0.1)
            return np.array([dx, 0.0, gripper_status])

        # If aligned and holding block, release it
        if gripper_status > 0.0:
            return np.array([0.0, 0.0, -1.0])

        return np.array([0.0, 0.0, 0.0])


class GraphPutDownOnTargetSkill(GraphPutDownSkill):
    """Skill for putting down blocks in target area."""

    def _get_operator_name(self) -> str:
        return "PutDownOnTarget"


class GraphObstacle2DPerceiver(Perceiver[GraphInstance]):
    """Perceiver for blocks2d environment."""

    def __init__(self, types: Obstacle2DTypes, n_blocks: int = 2) -> None:
        """Initialize with required types."""
        self._robot = Object("robot", types.robot)
        self._blocks = [Object(f"block{i+1}", types.block) for i in range(n_blocks)]
        self._table = Object("table", types.surface)
        self._target_area = Object("target_area", types.surface)
        self._predicates: Obstacle2DPredicates | None = None
        self._types = types
        self.n_blocks = n_blocks

    def initialize(self, predicates: Obstacle2DPredicates) -> None:
        """Initialize predicates after environment creation."""
        self._predicates = predicates

    @property
    def predicates(self) -> Obstacle2DPredicates:
        """Get predicates, ensuring they're initialized."""
        if self._predicates is None:
            raise RuntimeError("Predicates not initialized. Call initialize() first.")
        return self._predicates

    def reset(
        self,
        obs: GraphInstance,
        _info: dict[str, Any],
    ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        """Reset perceiver with observation and info."""
        objects = {
            self._robot,
            self._table,
            self._target_area,
        }
        objects.update(self._blocks)
        atoms = self._get_atoms(obs)
        goal = {
            self.predicates["On"]([self._blocks[0], self._target_area]),
            self.predicates["GripperEmpty"]([self._robot]),
        }
        return objects, atoms, goal

    def step(self, obs: GraphInstance) -> set[GroundAtom]:
        """Step perceiver with observation."""
        return self._get_atoms(obs)

    def _get_atoms(self, obs: GraphInstance) -> set[GroundAtom]:
        atoms = set()
        robot_pos = None
        robot_gripper = None
        block_positions = {}

        for node in obs.nodes:
            node_type = int(node[0])
            if node_type == 0:  # Robot
                robot_pos = node[1:3]
                robot_gripper = float(node[5])
            elif node_type == 1:  # Block
                block_id = int(node[5])
                block_positions[block_id] = node[1:3]
        assert robot_pos is not None
        assert robot_gripper is not None

        target_x = 0.5
        target_y = 0.0
        target_width = 0.2
        target_height = 0.2
        block_width = 0.2
        block_height = 0.2

        atoms.add(self.predicates["IsTarget"]([self._target_area]))
        atoms.add(self.predicates["NotIsTarget"]([self._table]))

        held_block_id = -1
        if robot_gripper > 0.5:
            for i in range(self.n_blocks):
                block_id = i + 1
                if block_id in block_positions and np.allclose(
                    block_positions[block_id], robot_pos, atol=1e-3
                ):
                    atoms.add(
                        self.predicates["Holding"]([self._robot, self._blocks[i]])
                    )
                    held_block_id = block_id
                    break
        if held_block_id == -1:
            atoms.add(self.predicates["GripperEmpty"]([self._robot]))

        overlapping_blocks = []
        for i in range(self.n_blocks):
            block_id = i + 1
            assert block_id in block_positions
            if is_block_in_target_area(
                block_positions[block_id][0],
                block_positions[block_id][1],
                block_width,
                block_height,
                target_x,
                target_y,
                target_width,
                target_height,
            ):
                overlapping_blocks.append(i)
                atoms.add(self.predicates["On"]([self._blocks[i], self._target_area]))
                atoms.add(
                    self.predicates["Overlap"]([self._blocks[i], self._target_area])
                )
            elif block_id != held_block_id and self._is_target_area_blocked(
                block_positions[block_id][0],
                block_width,
                target_x,
                target_width,
            ):
                overlapping_blocks.append(i)
                atoms.add(
                    self.predicates["Overlap"]([self._blocks[i], self._target_area])
                )
            elif block_id != held_block_id:
                atoms.add(self.predicates["On"]([self._blocks[i], self._table]))

        # Check if target area is clear
        if not overlapping_blocks:
            atoms.add(self.predicates["Clear"]([self._target_area]))

        # Table is always "clear" since we can place things on it
        atoms.add(self.predicates["Clear"]([self._table]))

        return atoms

    def _is_target_area_blocked(
        self,
        block_x: float,
        block_width: float,
        target_x: float,
        target_width: float,
    ) -> bool:
        """Check if block 2 blocks the target area.

        Block 2 is considered blocking if it overlaps with the target
        area enough that another block cannot fit in the remaining
        space.
        """
        target_left = target_x - target_width / 2
        target_right = target_x + target_width / 2
        block_left = block_x - block_width / 2
        block_right = block_x + block_width / 2

        # If no horizontal overlap, not blocking
        if block_right <= target_left or block_left >= target_right:
            return False

        # Calculate remaining free width
        overlap_width = min(block_right, target_right) - max(block_left, target_left)
        free_width = target_width - overlap_width + 1e-6

        # Block needs at least its width to fit
        return free_width < block_width


class BaseGraphObstacle2DTAMPSystem(BaseTAMPSystem[GraphInstance, NDArray[np.float32]]):
    """Base TAMP system for 2D obstacle graph-based environment."""

    def __init__(
        self,
        planning_components: PlanningComponents[GraphInstance],
        n_blocks: int = 2,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize graph-based Obstacle2D TAMP system."""
        self._render_mode = render_mode
        self.n_blocks = n_blocks
        super().__init__(
            planning_components, name="GraphObstacle2DTAMPSystem", seed=seed
        )

    def _create_env(self) -> gym.Env:
        """Create base environment."""
        return GraphObstacle2DEnv(n_blocks=self.n_blocks, render_mode=self._render_mode)

    def _get_domain_name(self) -> str:
        """Get domain name."""
        return "graph-obstacle2d-domain"

    def get_domain(self) -> PDDLDomain:
        """Get domain."""
        return PDDLDomain(
            self._get_domain_name(),
            self.components.operators,
            self.components.predicate_container.as_set(),
            self.components.types,
        )

    @classmethod
    def _create_planning_components(
        cls, n_blocks: int = 2
    ) -> PlanningComponents[GraphInstance]:
        """Create planning components for graph-based Obstacle2D system."""
        types_container = Obstacle2DTypes()
        types_set = types_container.as_set()

        predicates = Obstacle2DPredicates(types_container)

        perceiver = GraphObstacle2DPerceiver(types_container, n_blocks)
        perceiver.initialize(predicates)

        robot = Variable("?robot", types_container.robot)
        block = Variable("?block", types_container.block)
        surface = Variable("?surface", types_container.surface)

        operators = {
            LiftedOperator(
                "PickUp",
                [robot, block, surface],
                preconditions={
                    predicates["GripperEmpty"]([robot]),
                    predicates["On"]([block, surface]),
                    predicates["NotIsTarget"]([surface]),
                },
                add_effects={
                    predicates["Holding"]([robot, block]),
                    predicates["Clear"]([surface]),
                },
                delete_effects={
                    predicates["GripperEmpty"]([robot]),
                    predicates["On"]([block, surface]),
                },
            ),
            LiftedOperator(
                "PickUpFromTarget",
                [robot, block, surface],
                preconditions={
                    predicates["GripperEmpty"]([robot]),
                    predicates["Overlap"]([block, surface]),
                    predicates["IsTarget"]([surface]),
                },
                add_effects={
                    predicates["Holding"]([robot, block]),
                    predicates["Clear"]([surface]),
                },
                delete_effects={
                    predicates["GripperEmpty"]([robot]),
                    predicates["Overlap"]([block, surface]),
                    predicates["On"]([block, surface]),
                },
            ),
            LiftedOperator(
                "PutDown",
                [robot, block, surface],
                preconditions={
                    predicates["Holding"]([robot, block]),
                    predicates["Clear"]([surface]),
                    predicates["NotIsTarget"]([surface]),
                },
                add_effects={
                    predicates["On"]([block, surface]),
                    predicates["GripperEmpty"]([robot]),
                },
                delete_effects={predicates["Holding"]([robot, block])},
            ),
            LiftedOperator(
                "PutDownOnTarget",
                [robot, block, surface],
                preconditions={
                    predicates["Holding"]([robot, block]),
                    predicates["Clear"]([surface]),
                    predicates["IsTarget"]([surface]),
                },
                add_effects={
                    predicates["On"]([block, surface]),
                    predicates["Overlap"]([block, surface]),
                    predicates["GripperEmpty"]([robot]),
                },
                delete_effects={
                    predicates["Holding"]([robot, block]),
                    predicates["Clear"]([surface]),
                },
            ),
        }

        return PlanningComponents(
            types=types_set,
            predicate_container=predicates,
            operators=operators,
            skills=set(),
            perceiver=perceiver,
        )

    @classmethod
    def create_default(
        cls,
        n_blocks: int = 2,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> BaseGraphObstacle2DTAMPSystem:
        """Factory method for creating system with default components."""
        planning_components = cls._create_planning_components(n_blocks=n_blocks)
        system = cls(
            planning_components,
            n_blocks=n_blocks,
            seed=seed,
            render_mode=render_mode,
        )
        skills = {
            GraphPickUpSkill(system.components),  # type: ignore[arg-type]
            GraphPickUpFromTargetSkill(system.components),  # type: ignore[arg-type]
            GraphPutDownSkill(system.components),  # type: ignore[arg-type]
            GraphPutDownOnTargetSkill(system.components),  # type: ignore[arg-type]
        }
        system.components.skills.update(skills)
        return system


class GraphObstacle2DTAMPSystem(
    ImprovisationalTAMPSystem[GraphInstance, NDArray[np.float32]],
    BaseGraphObstacle2DTAMPSystem,
):
    """TAMP system for 2D obstacle graph-based environment with improvisational
    policy learning enabled."""

    def __init__(
        self,
        planning_components: PlanningComponents[GraphInstance],
        n_blocks: int = 2,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize graph-based Obstacle2D TAMP system."""
        self.n_blocks = n_blocks
        self._render_mode = render_mode
        ImprovisationalTAMPSystem.__init__(
            self,
            planning_components,
            seed=seed,
            render_mode=render_mode,
        )
        BaseGraphObstacle2DTAMPSystem.__init__(
            self,
            planning_components,
            n_blocks=n_blocks,
            seed=seed,
            render_mode=render_mode,
        )

    def _create_wrapped_env(
        self, components: PlanningComponents[GraphInstance]
    ) -> gym.Env:
        """Create wrapped environment for training."""
        return ImprovWrapper(
            base_env=self.env,
            perceiver=components.perceiver,
            step_penalty=-0.5,
            achievement_bonus=10.0,
        )

    @classmethod
    def create_default(
        cls,
        n_blocks: int = 2,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> GraphObstacle2DTAMPSystem:
        """Factory method for creating improvisational system with default
        components."""
        planning_components = cls._create_planning_components(n_blocks=n_blocks)
        system = GraphObstacle2DTAMPSystem(
            planning_components,
            n_blocks=n_blocks,
            seed=seed,
            render_mode=render_mode,
        )
        skills = {
            GraphPickUpSkill(system.components),  # type: ignore[arg-type]
            GraphPickUpFromTargetSkill(system.components),  # type: ignore[arg-type]
            GraphPutDownSkill(system.components),  # type: ignore[arg-type]
            GraphPutDownOnTargetSkill(system.components),  # type: ignore[arg-type]
        }
        system.components.skills.update(skills)
        return system
