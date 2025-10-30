"""Core obstacle2d environment."""

from __future__ import annotations

from typing import Any, NamedTuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Graph, GraphInstance
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tomsgeoms2d.structs import Rectangle
from tomsutils.utils import fig2data


class Obstacle2DState(NamedTuple):
    """State of the obstacle2d environment."""

    robot_position: NDArray[np.float32]
    block_1_position: NDArray[np.float32]
    block_2_position: NDArray[np.float32]
    gripper_status: float
    block_3_position: NDArray[np.float32] | None = None

    def to_graph_observation(self) -> GraphInstance:
        """Convert to graph observation."""
        nodes = []

        # Robot node: [type=0, x, y, width, height, gripper_status]
        robot_node = np.zeros(6, dtype=np.float32)
        robot_node[0] = 0
        robot_node[1:3] = self.robot_position
        robot_node[3:5] = [0.2, 0.2]
        robot_node[5] = self.gripper_status
        nodes.append(robot_node)

        # Block nodes: [type=1, x, y, width, height, block_id]
        block1_node = np.zeros(6, dtype=np.float32)
        block1_node[0] = 1
        block1_node[1:3] = self.block_1_position
        block1_node[3:5] = [0.2, 0.2]
        block1_node[5] = 1
        nodes.append(block1_node)

        block2_node = np.zeros(6, dtype=np.float32)
        block2_node[0] = 1
        block2_node[1:3] = self.block_2_position
        block2_node[3:5] = [0.2, 0.2]
        block2_node[5] = 2
        nodes.append(block2_node)

        if self.block_3_position is not None:
            block3_node = np.zeros(6, dtype=np.float32)
            block3_node[0] = 1
            block3_node[1:3] = self.block_3_position
            block3_node[3:5] = [0.2, 0.2]
            block3_node[5] = 3
            nodes.append(block3_node)

        return GraphInstance(nodes=np.stack(nodes), edges=None, edge_links=None)


def is_block_in_target_area(
    block_x: float,
    block_y: float,
    block_width: float,
    block_height: float,
    target_x: float,
    target_y: float,
    target_width: float,
    target_height: float,
) -> bool:
    """Check if block is completely in target area."""
    target_left = target_x - target_width / 2 - 1e-4
    target_right = target_x + target_width / 2 + 1e-4
    target_bottom = target_y - target_height / 2 - 1e-4
    target_top = target_y + target_height / 2 + 1e-4

    block_left = block_x - block_width / 2
    block_right = block_x + block_width / 2
    block_bottom = block_y - block_height / 2
    block_top = block_y + block_height / 2

    return (
        target_left <= block_left
        and block_right <= target_right
        and target_bottom <= block_bottom
        and block_top <= target_top
    )


class GraphObstacle2DEnv(gym.Env):
    """A block environment in 2D with graph observation support and variable
    number of blocks."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, n_blocks: int = 2, render_mode: str | None = None) -> None:
        assert n_blocks in (2, 3), "n_blocks must be either 2 or 3 given env constraint"
        self.n_blocks = n_blocks
        self.observation_space = Graph(
            node_space=Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            edge_space=None,
        )
        self.action_space = Box(
            low=np.array([-0.1, -0.1, -1.0]),
            high=np.array([0.1, 0.1, 1.0]),
            dtype=np.float32,
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._robot_width = 0.2
        self._robot_height = 0.2
        self._block_width = 0.2
        self._block_height = 0.2
        self._target_area = {"x": 0.5, "y": 0.0, "width": 0.2, "height": 0.2}

        self.block_positions: list[NDArray[np.float32]] = []

        # Initialize state
        self.state = self._get_default_state()

    @property
    def robot_position(self) -> NDArray[np.float32]:
        """Get robot position."""
        return self.state.robot_position

    @property
    def block_1_position(self) -> NDArray[np.float32]:
        """Get block 1 position."""
        return self.state.block_1_position

    @property
    def block_2_position(self) -> NDArray[np.float32]:
        """Get block 2 position."""
        return self.state.block_2_position

    @property
    def gripper_status(self) -> float:
        """Get gripper status."""
        return self.state.gripper_status

    def _get_default_state(self) -> Obstacle2DState:
        """Get default initial state with randomized blocks' positions."""
        target_x = self._target_area["x"]
        target_width = self._target_area["width"]
        target_left = target_x - target_width / 2
        target_right = target_x + target_width / 2

        # Position block 1 and 2 randomly but in the hardest positions for the task
        block_2_x = self.np_random.choice([target_left, target_x, target_right])
        state = Obstacle2DState(
            robot_position=np.array([0.5, 1.0], dtype=np.float32),
            block_1_position=np.array([0.8, 0.0], dtype=np.float32),
            block_2_position=np.array([block_2_x, 0.0], dtype=np.float32),
            gripper_status=0.0,
        )
        if self.n_blocks > 2:
            block_3_x = np.random.uniform(0.0, 0.2)
            state = Obstacle2DState(
                robot_position=state.robot_position,
                block_1_position=state.block_1_position,
                block_2_position=state.block_2_position,
                block_3_position=np.array([block_3_x, 0.0], dtype=np.float32),
                gripper_status=0.0,
            )

        self.block_positions = [state.block_1_position, state.block_2_position]
        if self.n_blocks > 2 and state.block_3_position is not None:
            self.block_positions.append(state.block_3_position)

        return state

    def reset_from_state(
        self,
        state: Obstacle2DState | GraphInstance,
        *,
        seed: int | None = None,
    ) -> tuple[GraphInstance, dict[str, Any]]:
        """Reset environment to specific state."""
        super().reset(seed=seed)

        if isinstance(state, GraphInstance):
            # Rebuild state from graph
            robot_pos = None
            block_positions = {}
            gripper_status = 0.0

            for node in state.nodes:
                node_type = int(node[0])
                if node_type == 0:  # Robot
                    robot_pos = node[1:3].copy()
                    gripper_status = float(node[5])
                elif node_type == 1:  # Block
                    block_id = int(node[5])
                    block_positions[block_id] = node[1:3].copy()

            assert robot_pos is not None, "Robot node not found"
            assert 1 in block_positions, "Block 1 not found"
            assert 2 in block_positions, "Block 2 not found"

            if 3 in block_positions and self.n_blocks > 2:
                self.state = Obstacle2DState(
                    robot_position=robot_pos,
                    block_1_position=block_positions[1],
                    block_2_position=block_positions[2],
                    block_3_position=block_positions[3],
                    gripper_status=gripper_status,
                )
                self.block_positions = [
                    block_positions[1],
                    block_positions[2],
                    block_positions[3],
                ]
            else:
                self.state = Obstacle2DState(
                    robot_position=robot_pos,
                    block_1_position=block_positions[1],
                    block_2_position=block_positions[2],
                    gripper_status=gripper_status,
                )
                self.block_positions = [block_positions[1], block_positions[2]]
        else:
            self.state = state
            self.block_positions = [state.block_1_position, state.block_2_position]
            if state.block_3_position is not None:
                self.block_positions.append(state.block_3_position)

        return self._get_obs(), self._get_info()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[GraphInstance, dict[str, Any]]:
        """Reset environment to default state."""
        super().reset(seed=seed)

        if options is None:
            self.state = self._get_default_state()
        else:
            robot_pos = options.get(
                "robot_pos", np.array([0.5, 1.0], dtype=np.float32)
            ).copy()
            block_1_pos = options.get(
                "block_1_pos", np.array([0.0, 0.0], dtype=np.float32)
            ).copy()
            block_2_pos = options.get(
                "block_2_pos", np.array([0.5, 0.0], dtype=np.float32)
            ).copy()
            block_3_pos = None
            if self.n_blocks > 2:
                block_3_pos = options.get(
                    "block_3_pos", np.array([1.0, 0.0], dtype=np.float32)
                ).copy()

            self.block_positions = [block_1_pos, block_2_pos]
            if self.n_blocks > 2 and block_3_pos is not None:
                self.block_positions.append(block_3_pos)

            self.state = Obstacle2DState(
                robot_position=robot_pos,
                block_1_position=block_1_pos,
                block_2_position=block_2_pos,
                block_3_position=block_3_pos,
                gripper_status=0.0,
            )

        return self._get_obs(), self._get_info()

    def _get_obs(self) -> GraphInstance:
        """Get observation from current state."""
        return self.state.to_graph_observation()

    def _get_info(self) -> dict[str, Any]:
        """Get info from current state."""
        return {
            "distance_to_block1": np.linalg.norm(
                self.robot_position - self.block_1_position
            ),
            "distance_to_block2": np.linalg.norm(
                self.robot_position - self.block_2_position
            ),
        }

    def step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[GraphInstance, float, bool, bool, dict[str, Any]]:
        """Take environment step."""
        dx, dy, gripper_action = action

        prev_state = self.state

        new_robot_position = np.array(
            [
                np.clip(self.robot_position[0] + dx, 0.0, 1.0),
                np.clip(self.robot_position[1] + dy, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        new_block_positions = [pos.copy() for pos in self.block_positions]
        new_gripper_status = float(gripper_action)

        # Handle pushing for all blocks
        pushed_blocks = set()
        # First handle direct robot pushing
        for i, block_pos in enumerate(self.block_positions):
            if self._is_adjacent(self.robot_position, block_pos):
                relative_pos = self.robot_position[0] - block_pos[0]
                if relative_pos * dx < 0.0:  # Push
                    new_block_positions[i][0] = np.clip(
                        block_pos[0] + dx, 0.0, 1.0
                    ).astype(np.float32)
                    pushed_blocks.add(i)
        # Then handle pushing (block-to-block)
        push_chain_continues = True
        while push_chain_continues:
            push_chain_continues = False
            for i in pushed_blocks.copy():
                pushing_block_pos = self.block_positions[i]
                for j, block_pos in enumerate(self.block_positions):
                    if j not in pushed_blocks and self._is_adjacent(
                        pushing_block_pos, block_pos
                    ):
                        relative_pos = pushing_block_pos[0] - block_pos[0]
                        if relative_pos * dx < 0.0:
                            new_block_positions[j][0] = np.clip(
                                block_pos[0] + dx, 0.0, 1.0
                            ).astype(np.float32)
                            pushed_blocks.add(j)
                            push_chain_continues = True

        # Check if already holding a block
        held_block_idx = -1
        for i, block_pos in enumerate(self.block_positions):
            if np.allclose(block_pos, self.robot_position, atol=1e-3):
                held_block_idx = i
                break

        picking_up_block_idx = -1
        # Handle picking up and placing down a block
        if held_block_idx >= 0:  # Already holding a block
            if new_gripper_status < -0.5:  # Releasing grip
                new_block_positions[held_block_idx][1] = 0.0
            else:  # Continuing to hold
                new_block_positions[held_block_idx] = new_robot_position.copy()
        elif new_gripper_status > 0.5:  # Attempting to pick up
            for i, block_pos in enumerate(self.block_positions):
                dist = np.linalg.norm(new_robot_position - block_pos)
                if dist <= ((self._robot_width + self._block_width) / 2) + 1e-3:
                    new_block_positions[i] = new_robot_position.copy()
                    picking_up_block_idx = i
                    break

        if self.n_blocks > 2:
            new_state = Obstacle2DState(
                robot_position=new_robot_position,
                block_1_position=new_block_positions[0],
                block_2_position=new_block_positions[1],
                block_3_position=new_block_positions[2],
                gripper_status=new_gripper_status,
            )
        else:
            new_state = Obstacle2DState(
                robot_position=new_robot_position,
                block_1_position=new_block_positions[0],
                block_2_position=new_block_positions[1],
                gripper_status=new_gripper_status,
            )
        self.state = new_state
        self.block_positions = new_block_positions

        # Check for collisions - revert to previous state if collision
        if self._check_collisions(held_block_idx, picking_up_block_idx):
            self.state = prev_state
            self.block_positions = [
                prev_state.block_1_position,
                prev_state.block_2_position,
            ]
            if self.n_blocks > 2 and prev_state.block_3_position is not None:
                self.block_positions.append(prev_state.block_3_position)
            obs = self._get_obs()
            info = self._get_info()
            return obs, -0.1, False, False, info

        obs = self._get_obs()
        info = self._get_info()

        goal_reached = is_block_in_target_area(
            self.block_1_position[0],
            self.block_1_position[1],
            self._block_width,
            self._block_height,
            self._target_area["x"],
            self._target_area["y"],
            self._target_area["width"],
            self._target_area["height"],
        )

        reward = 1.0 if goal_reached else 0.0
        terminated = goal_reached

        return obs, reward, terminated, False, info

    def extract_relevant_object_features(self, obs, relevant_object_names):
        """Extract features from relevant objects in the observation."""
        if not hasattr(obs, "nodes"):
            return obs  # Not a graph observation

        nodes = obs.nodes
        robot_node = None
        block_nodes = {}
        for node in nodes:
            node_type = int(node[0])
            if node_type == 0:  # Robot
                robot_node = node[1:]
            elif node_type == 1:  # Block
                block_id = int(node[5])
                block_name = f"block{block_id}"
                if block_name in relevant_object_names:
                    block_nodes[block_name] = node[1:5]

        features = []
        assert robot_node is not None
        features.extend(robot_node)
        for block_name in sorted(relevant_object_names):
            if block_name in block_nodes:
                features.extend(block_nodes[block_name])

        return np.array(features, dtype=np.float32)

    def _check_collisions(self, held_block_idx, picking_up_block_idx) -> bool:
        """Check for collisions between objects."""
        # Check for collisions between blocks
        for i in range(len(self.block_positions)):
            for j in range(i + 1, len(self.block_positions)):
                if self._check_collision_between(
                    self.block_positions[i], self.block_positions[j]
                ):
                    return True

        # Check collision between robot and blocks (except held block)
        for i, block_pos in enumerate(self.block_positions):
            if (
                i != held_block_idx
                and i != picking_up_block_idx
                and self._check_collision_between(self.robot_position, block_pos)
            ):
                return True

        return False

    def _check_collision_between(
        self,
        pos1: NDArray[np.float32],
        pos2: NDArray[np.float32],
    ) -> bool:
        """Check collision between two positions."""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])

        width_sum = self._block_width - 1e-3
        height_sum = self._block_height - 1e-3

        if np.array_equal(pos1, self.robot_position) or np.array_equal(
            pos2, self.robot_position
        ):
            width_sum = (self._robot_width + self._block_width) / 2 - 1e-3
            height_sum = (self._robot_height + self._block_height) / 2 - 1e-3

        return dx < width_sum and dy < height_sum

    def _is_adjacent(
        self,
        robot_position: NDArray[np.float32],
        block_position: NDArray[np.float32],
    ) -> bool:
        vertical_aligned = (
            np.abs(robot_position[1] - block_position[1])
            < (self._robot_height + self._block_height) / 4
        )
        horizontal_adjacent = np.isclose(
            np.abs(robot_position[0] - block_position[0]),
            (self._robot_width + self._block_width) / 2,
            atol=2e-2,
        )
        return vertical_aligned and horizontal_adjacent

    def render(self) -> NDArray[np.uint8]:  # type: ignore
        """Render the environment."""
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set_xlim(
            (
                0.0 - max(self._robot_width / 2, self._block_width / 2),
                1.0 + max(self._robot_width / 2, self._block_width / 2),
            )
        )
        ax.set_ylim(
            (
                0.0 - max(self._robot_height / 2, self._block_height / 2),
                1.0 + max(self._robot_height / 2, self._block_height / 2),
            )
        )

        target_rect = Rectangle.from_center(
            self._target_area["x"],
            self._target_area["y"],
            self._target_area["width"],
            self._target_area["height"],
            0.0,
        )
        target_rect.plot(ax, facecolor="green", edgecolor="red")

        robot_rect = Rectangle.from_center(
            self.robot_position[0],
            self.robot_position[1],
            self._robot_width,
            self._robot_height,
            0.0,
        )
        robot_rect.plot(ax, facecolor="silver", edgecolor="black")

        for i, block_position in enumerate(self.block_positions):
            block_rect = Rectangle.from_center(
                block_position[0],
                block_position[1],
                self._block_width,
                self._block_height,
                0.0,
            )
            block_rect.plot(ax, facecolor="blue", edgecolor="black")

            if i == 0:
                ax.text(
                    block_position[0],
                    block_position[1],
                    "T",
                    fontsize=20,
                    ha="center",
                    va="center",
                    color="black",
                )

        img = fig2data(fig)
        plt.close(fig)
        return img

    def clone(self) -> GraphObstacle2DEnv:
        """Clone the environment."""
        clone_env = GraphObstacle2DEnv(
            n_blocks=self.n_blocks, render_mode=self.render_mode
        )
        clone_env.reset_from_state(self.state)
        return clone_env


class Obstacle2DEnv(gym.Env):
    """A block environment in 2D.

    Observations are 15D:
    - 4D for the x, y position (center), the width, and the height of
    the robot
    - 2D for the x, y position (center) of block 1 (the target block)
    - 2D for the x, y position (center) of block 2 (the other block)
    - 2D for the width and the height of the blocks
    - 1D for the gripper "activation"
    - 4D for the x, y position (center), the width, and the height of
    the target area

    Actions are 3D:
    - 2D for dx, dy for the robot
    - 1D for activating / deactivating the gripper

    The environment has boundaries x=0 to x=1 and y=0 to y=1.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None) -> None:
        self.observation_space = Box(low=0, high=1, shape=(15,), dtype=np.float32)
        self.action_space = Box(
            low=np.array([-0.1, -0.1, -1.0]),
            high=np.array([0.1, 0.1, 1.0]),
            dtype=np.float32,
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._robot_width = 0.2
        self._robot_height = 0.2
        self._block_width = 0.2
        self._block_height = 0.2
        self._target_area = {"x": 0.5, "y": 0.0, "width": 0.2, "height": 0.2}

        self.state = self._get_default_state()

    @property
    def robot_position(self) -> NDArray[np.float32]:
        """Get robot position."""
        return self.state.robot_position

    @property
    def block_1_position(self) -> NDArray[np.float32]:
        """Get block 1 position."""
        return self.state.block_1_position

    @property
    def block_2_position(self) -> NDArray[np.float32]:
        """Get block 2 position."""
        return self.state.block_2_position

    @property
    def gripper_status(self) -> float:
        """Get gripper status."""
        return self.state.gripper_status

    def _get_default_state(self) -> Obstacle2DState:
        """Get default initial state with randomized blocks' positions."""
        target_x = self._target_area["x"]
        target_width = self._target_area["width"]
        target_left = target_x - target_width / 2
        target_right = target_x + target_width / 2

        # Position block 1 and 2 in random but hardest positions for the task
        block_2_x = self.np_random.choice([target_left, target_x, target_right])
        return Obstacle2DState(
            robot_position=np.array([0.5, 1.0], dtype=np.float32),
            block_1_position=np.array([0.8, 0.0], dtype=np.float32),
            block_2_position=np.array([block_2_x, 0.0], dtype=np.float32),
            gripper_status=0.0,
        )

    def reset_from_state(
        self,
        state: Obstacle2DState | NDArray[np.float32],
        *,
        seed: int | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset environment to specific state."""
        super().reset(seed=seed)

        if isinstance(state, np.ndarray):
            # Convert array to state
            self.state = Obstacle2DState(
                robot_position=state[0:2].copy(),
                block_1_position=state[4:6].copy(),
                block_2_position=state[6:8].copy(),
                gripper_status=float(state[10]),
            )
        else:
            self.state = state

        return self._get_obs(), self._get_info()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset environment to default state."""
        super().reset(seed=seed)

        if options is None:
            self.state = self._get_default_state()
        else:
            self.state = Obstacle2DState(
                robot_position=options.get(
                    "robot_pos", np.array([0.5, 1.0], dtype=np.float32)
                ).copy(),
                block_1_position=options.get(
                    "block_1_pos", np.array([0.0, 0.0], dtype=np.float32)
                ).copy(),
                block_2_position=options.get(
                    "block_2_pos", np.array([0.5, 0.0], dtype=np.float32)
                ).copy(),
                gripper_status=0.0,
            )

        return self._get_obs(), self._get_info()

    def _get_obs(self) -> NDArray[np.float32]:
        """Get observation from current state."""
        return np.array(
            [
                self.robot_position[0],
                self.robot_position[1],
                self._robot_width,
                self._robot_height,
                self.block_1_position[0],
                self.block_1_position[1],
                self.block_2_position[0],
                self.block_2_position[1],
                self._block_width,
                self._block_height,
                self.gripper_status,
                self._target_area["x"],
                self._target_area["y"],
                self._target_area["width"],
                self._target_area["height"],
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        """Get info from current state."""
        return {
            "distance_to_block1": np.linalg.norm(
                self.robot_position - self.block_1_position
            ),
            "distance_to_block2": np.linalg.norm(
                self.robot_position - self.block_2_position
            ),
        }

    def step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Take environment step."""
        dx, dy, gripper_action = action

        prev_state = self.state

        new_robot_position = np.array(
            [
                np.clip(self.robot_position[0] + dx, 0.0, 1.0),
                np.clip(self.robot_position[1] + dy, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

        new_block_1_position = self.block_1_position.copy()
        new_block_2_position = self.block_2_position.copy()
        new_gripper_status = float(gripper_action)

        # Handle block 2 pushing
        if self._is_adjacent(self.robot_position, self.block_2_position):
            relative_pos = self.robot_position[0] - self.block_2_position[0]
            if relative_pos * dx < 0.0:  # Push
                new_block_2_position[0] = np.clip(
                    self.block_2_position[0] + dx, 0.0, 1.0
                ).astype(np.float32)

        # Check if already holding a block
        block1_held = np.allclose(self.block_1_position, self.robot_position, atol=1e-3)
        block2_held = np.allclose(self.block_2_position, self.robot_position, atol=1e-3)
        block_held = block1_held or block2_held

        # Handle picking up and placing down a block
        picking_up_block1 = False
        picking_up_block2 = False
        if block_held:  # Already holding a block
            if new_gripper_status < -0.5:  # Releasing grip
                if block1_held:
                    new_block_1_position = np.array(
                        [self.block_1_position[0], 0.0], dtype=np.float32
                    )
                elif block2_held:
                    new_block_2_position = np.array(
                        [self.block_2_position[0], 0.0], dtype=np.float32
                    )
            else:  # Continuing to hold
                if block1_held:
                    new_block_1_position = new_robot_position.copy()
                elif block2_held:
                    new_block_2_position = new_robot_position.copy()
        elif new_gripper_status > 0.5:  # Attempting to pick up
            # Calculate distances to blocks
            dist_to_block1 = np.linalg.norm(new_robot_position - self.block_1_position)
            dist_to_block2 = np.linalg.norm(new_robot_position - self.block_2_position)
            # Pick up a block that's close enough
            threshold = ((self._robot_width + self._block_width) / 2) + 1e-3
            if dist_to_block1 <= threshold:
                new_block_1_position = new_robot_position.copy()
                picking_up_block1 = True
            elif dist_to_block2 <= threshold:
                new_block_2_position = new_robot_position.copy()
                picking_up_block2 = True

        self.state = Obstacle2DState(
            robot_position=new_robot_position,
            block_1_position=new_block_1_position,
            block_2_position=new_block_2_position,
            gripper_status=new_gripper_status,
        )

        # Check for collisions - revert to previous state if collision
        if self._check_collisions(
            block1_held, block2_held, picking_up_block1, picking_up_block2
        ):
            self.state = prev_state
            obs = self._get_obs()
            info = self._get_info()
            return obs, -0.1, False, False, info

        obs = self._get_obs()
        info = self._get_info()

        goal_reached = is_block_in_target_area(
            self.block_1_position[0],
            self.block_1_position[1],
            self._block_width,
            self._block_height,
            self._target_area["x"],
            self._target_area["y"],
            self._target_area["width"],
            self._target_area["height"],
        )

        reward = 1.0 if goal_reached else 0.0
        terminated = goal_reached

        return obs, reward, terminated, False, info

    def _check_collisions(
        self,
        block1_held: bool,
        block2_held: bool,
        picking_up_block1: bool,
        picking_up_block2: bool,
    ) -> bool:
        """Check for collisions between objects."""
        if (
            self._check_collision_between(self.robot_position, self.block_1_position)
            and not picking_up_block1
            and not block1_held
        ):
            return True
        if (
            self._check_collision_between(self.robot_position, self.block_2_position)
            and not picking_up_block2
            and not block2_held
        ):
            return True
        if self._check_collision_between(self.block_1_position, self.block_2_position):
            return True
        return False

    def _check_collision_between(
        self,
        pos1: NDArray[np.float32],
        pos2: NDArray[np.float32],
    ) -> bool:
        """Check collision between two positions."""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])

        width_sum = self._block_width - 1e-3
        height_sum = self._block_height - 1e-3

        if np.array_equal(pos1, self.robot_position) or np.array_equal(
            pos2, self.robot_position
        ):
            width_sum = (self._robot_width + self._block_width) / 2 - 1e-3
            height_sum = (self._robot_height + self._block_height) / 2 - 1e-3

        return dx < width_sum and dy < height_sum

    def _is_adjacent(
        self,
        robot_position: NDArray[np.float32],
        block_position: NDArray[np.float32],
    ) -> bool:
        vertical_aligned = (
            np.abs(robot_position[1] - block_position[1])
            < (self._robot_height + self._block_height) / 4
        )
        horizontal_adjacent = np.isclose(
            np.abs(robot_position[0] - block_position[0]),
            (self._robot_width + self._block_width) / 2,
            atol=2e-2,
        )
        return vertical_aligned and horizontal_adjacent

    def render(self) -> NDArray[np.uint8]:  # type: ignore
        """Render the environment."""
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set_xlim(
            (
                0.0 - max(self._robot_width / 2, self._block_width / 2),
                1.0 + max(self._robot_width / 2, self._block_width / 2),
            )
        )
        ax.set_ylim(
            (
                0.0 - max(self._robot_height / 2, self._block_height / 2),
                1.0 + max(self._robot_height / 2, self._block_height / 2),
            )
        )

        target_rect = Rectangle.from_center(
            self._target_area["x"],
            self._target_area["y"],
            self._target_area["width"],
            self._target_area["height"],
            0.0,
        )
        target_rect.plot(ax, facecolor="green", edgecolor="red")

        robot_rect = Rectangle.from_center(
            self.robot_position[0],
            self.robot_position[1],
            self._robot_width,
            self._robot_height,
            0.0,
        )
        robot_rect.plot(ax, facecolor="silver", edgecolor="black")

        for i, block_position in enumerate(
            [self.block_1_position, self.block_2_position]
        ):
            block_rect = Rectangle.from_center(
                block_position[0],
                block_position[1],
                self._block_width,
                self._block_height,
                0.0,
            )
            block_rect.plot(ax, facecolor="blue", edgecolor="black")

            if i == 0:
                ax.text(
                    block_position[0],
                    block_position[1],
                    "T",
                    fontsize=20,
                    ha="center",
                    va="center",
                    color="black",
                )

        img = fig2data(fig)
        plt.close(fig)
        return img

    def clone(self) -> Obstacle2DEnv:
        """Clone the environment."""
        clone_env = Obstacle2DEnv(self.render_mode)
        clone_env.reset_from_state(self.state)
        return clone_env
