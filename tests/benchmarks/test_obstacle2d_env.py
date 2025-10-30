"""Tests for core obstacle2d environment."""

import numpy as np
from gymnasium.wrappers import TimeLimit

from tamp_improv.benchmarks.obstacle2d_env import GraphObstacle2DEnv, Obstacle2DEnv


def test_graph_obstacle2d_env():
    """Test basic functionality of Obstacle2D environment."""
    env = GraphObstacle2DEnv(n_blocks=3, render_mode="rgb_array")
    env = TimeLimit(env, max_episode_steps=100)

    # # Uncomment to generate videos.
    # from gymnasium.wrappers import RecordVideo

    # env = RecordVideo(env, "videos/graph-obstacle2d-test")

    obs, info = env.reset()

    env.action_space.seed(123)

    # Hard-coded sequence of actions
    actions = [
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([-0.1, 0.0, 0.0]),  # Move left
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([-0.1, 0.0, 0.0]),  # Move left
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([-0.1, 0.0, 0.0]),  # Move left
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([0.1, 0.0, 0.0]),  # Move right
        np.array([0.1, 0.0, 0.0]),  # Move right
        np.array([0.1, 0.0, 0.0]),  # Move right
        np.array([0.1, 0.0, 0.0]),  # Move right
        np.array([0.1, 0.0, 0.0]),  # Move right
        np.array([0.1, 0.0, 0.0]),  # Move right
        np.array([0.1, 0.0, 0.0]),  # Move right
        np.array([0.1, 0.0, 0.0]),  # Move right
    ]

    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Obs: {obs}, Reward: {reward}, Info: {info}")

        if terminated or truncated:
            print("Episode finished")
            break

    env.close()


def test_obstacle2d_env():
    """Test basic functionality of Obstacle2D environment."""
    env = Obstacle2DEnv(render_mode="rgb_array")
    env = TimeLimit(env, max_episode_steps=100)

    # # Uncomment to generate videos.
    # from gymnasium.wrappers import RecordVideo

    # env = RecordVideo(env, "videos/obstacle2d-test")

    obs, info = env.reset()

    env.action_space.seed(123)

    # Hard-coded sequence of actions to reach the goal
    actions = [
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([-0.1, 0.0, 0.0]),  # Move left
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([-0.1, 0.0, 0.0]),  # Move left
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([-0.1, 0.0, 0.0]),  # Move left
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([-0.1, 0.0, 0.0]),  # Move left
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([-0.1, 0.0, 0.0]),  # Move left
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([0.0, -0.1, 0.0]),  # Move down
        np.array([0.0, 0.0, 1.0]),  # Activate gripper and pick up block
        np.array([0.1, 0.0, 1.0]),  # Move right
        np.array([0.1, 0.0, 1.0]),  # Move right
        np.array([0.1, 0.0, 1.0]),  # Move right
        np.array([0.1, 0.0, 1.0]),  # Move right
        np.array([0.1, 0.0, 1.0]),  # Move right
        np.array([0.0, 0.0, -1.0]),  # Drop block
    ]

    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Obs: {obs}, Reward: {reward}, Info: {info}")

        if terminated or truncated:
            print("Episode finished")
            break

    env.close()
