"""Test pybullet obstacle tower environment."""

from task_then_motion_planning.planning import TaskThenMotionPlanner

from tamp_improv.benchmarks.pybullet_obstacle_tower_graph import (
    GraphObstacleTowerTAMPSystem,
)


def test_pybullet():
    """Test base TAMP functionality of pybullet environment."""
    system = GraphObstacleTowerTAMPSystem.create_default(
        seed=124, render_mode="rgb_array", num_obstacle_blocks=3
    )
    # from gymnasium.wrappers import RecordVideo
    # system.env = RecordVideo(system.env, "videos/obstacle-tower-ttmp-test")

    planner = TaskThenMotionPlanner(
        system.types,
        system.predicates,
        system.perceiver,
        system.operators,
        system.skills,
        planner_id="pyperplan",
    )

    obs, info = system.env.reset(seed=124)
    planner.reset(obs, info)

    for step in range(10000):
        action = planner.step(obs)
        obs, reward, done, _, info = system.env.step(action)
        if done:
            print(f"Goal reached in {step + 1} steps!")
            assert reward > 0
            break
    else:
        assert False, "Goal not reached"

    system.env.close()
