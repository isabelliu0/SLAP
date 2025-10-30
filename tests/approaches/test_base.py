"""Test base random approach."""

import pytest

from tamp_improv.approaches.random import RandomApproach
from tamp_improv.benchmarks.obstacle2d import Obstacle2DTAMPSystem
from tamp_improv.benchmarks.pybullet_obstacle_tower import ObstacleTowerTAMPSystem


def run_episode(system, approach, max_steps):
    """Run single episode with approach."""
    obs, info = system.reset()
    step_result = approach.reset(obs, info)

    obs, reward, terminated, truncated, info = system.env.step(step_result.action)
    if terminated or truncated:
        return 1

    for step in range(1, max_steps):
        step_result = approach.step(obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = system.env.step(step_result.action)
        if terminated or truncated:
            return step + 1
    return max_steps


@pytest.mark.parametrize(
    "system_cls,max_steps",
    [(Obstacle2DTAMPSystem, 100), (ObstacleTowerTAMPSystem, 200)],
)
def test_random_approach(system_cls, max_steps):
    """Test random approach on different environments."""
    system = system_cls.create_default(seed=42)
    approach = RandomApproach(system, seed=42)

    steps = run_episode(system, approach, max_steps)
    assert steps <= max_steps
