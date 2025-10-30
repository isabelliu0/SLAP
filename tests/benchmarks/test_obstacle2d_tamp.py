"""Tests for Obstacle2D environment with TAMP."""

import pytest
from gymnasium.wrappers import TimeLimit
from task_then_motion_planning.planning import TaskThenMotionPlanner

from tamp_improv.benchmarks.obstacle2d import BaseObstacle2DTAMPSystem
from tamp_improv.benchmarks.obstacle2d_graph import BaseGraphObstacle2DTAMPSystem


@pytest.mark.parametrize(
    "system_cls", [BaseObstacle2DTAMPSystem, BaseGraphObstacle2DTAMPSystem]
)
def test_obstacle2d_tamp_system(system_cls):
    """Test Obstacle2D environment with TAMP planner."""
    tamp_system = system_cls.create_default(render_mode="rgb_array", seed=42)

    env = TimeLimit(tamp_system.env, max_episode_steps=50)

    # # Uncomment to generate videos.
    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/obstacle2d-planning-test")

    planner = TaskThenMotionPlanner(
        types=tamp_system.types,
        predicates=tamp_system.predicates,
        perceiver=tamp_system.perceiver,
        operators=tamp_system.operators,
        skills=tamp_system.skills,
        planner_id="pyperplan",
    )

    obs, info = env.reset()
    objects, atoms, goal = tamp_system.perceiver.reset(obs, info)
    print("Objects:", objects)
    print("Initial atoms:", atoms)
    print("Goal:", goal)

    try:
        planner.reset(obs, info)
    except Exception as e:
        print("Error during planner reset:", str(e))
        print(
            "Current problem:",
            planner._current_problem,  # pylint: disable=protected-access
        )
        print("Current domain:", planner._domain)  # pylint: disable=protected-access
        raise

    total_reward = 0
    for step in range(100):
        action = planner.step(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        print(f"Step {step + 1}: Action: {action}, Obs: {obs}, Reward: {reward}")

        if terminated or truncated:
            print(f"Episode finished after {step + 1} steps")
            print(f"Total reward: {total_reward}")
            break
    else:
        print("Episode didn't finish within 100 steps")

    env.close()
