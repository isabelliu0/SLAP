"""Pure planning baselines for PyBullet environments."""

import argparse

from task_then_motion_planning.planning import TaskThenMotionPlanner

from tamp_improv.benchmarks.pybullet_cleanup_table import (
    BaseCleanupTableTAMPSystem,
)
from tamp_improv.benchmarks.pybullet_cluttered_drawer import (
    BaseClutteredDrawerTAMPSystem,
)
from tamp_improv.benchmarks.pybullet_obstacle_tower_graph import (
    BaseGraphObstacleTowerTAMPSystem,
)


def run_obstacle_tower_planning(
    seed: int = 124,
    render_mode: str | None = None,
    max_steps: int = 500,
) -> None:
    """Run pure planning baseline on GraphObstacleTower environment."""
    print("\n" + "=" * 80)
    print("Running Pure Planning on GraphObstacleTower Environment")
    print("=" * 80)

    system = BaseGraphObstacleTowerTAMPSystem.create_default(
        seed=seed,
        render_mode=render_mode,
        num_obstacle_blocks=3,
    )

    planner = TaskThenMotionPlanner(
        system.types,
        system.predicates,
        system.perceiver,
        system.operators,
        system.skills,
        planner_id="pyperplan",
    )

    obs, info = system.env.reset(seed=seed)
    planner.reset(obs, info)

    for step in range(max_steps):
        action = planner.step(obs)
        obs, reward, done, _, info = system.env.step(action)
        if done:
            print(f"\nGoal reached in {step + 1} steps!")
            print(f"Final reward: {reward}")
            if float(reward) > 0:
                print("SUCCESS: Task completed successfully!")
            break
    else:
        print(f"\nFAILED: Goal not reached within {max_steps} steps")

    system.env.close()  # type: ignore[no-untyped-call]


def run_cluttered_drawer_planning(
    seed: int = 123,
    render_mode: str | None = None,
    max_steps: int = 10000,
) -> None:
    """Run pure planning baseline on ClutteredDrawer environment."""
    print("\n" + "=" * 80)
    print("Running Pure Planning on ClutteredDrawer Environment")
    print("=" * 80)

    system = BaseClutteredDrawerTAMPSystem.create_default(
        seed=seed,
        render_mode=render_mode,
    )

    planner = TaskThenMotionPlanner(
        system.types,
        system.predicates,
        system.perceiver,
        system.operators,
        system.skills,
        planner_id="pyperplan",
    )

    obs, info = system.env.reset(seed=seed)
    planner.reset(obs, info)

    for step in range(max_steps):
        action = planner.step(obs)
        obs, reward, done, _, info = system.env.step(action)
        if done:
            print(f"\nGoal reached in {step + 1} steps!")
            print(f"Final reward: {reward}")
            if float(reward) > 0:
                print("SUCCESS: Task completed successfully!")
            break
    else:
        print(f"\nFAILED: Goal not reached within {max_steps} steps")

    system.env.close()  # type: ignore[no-untyped-call]


def run_cleanup_table_planning(
    seed: int = 123,
    render_mode: str | None = None,
    max_steps: int = 10000,
    max_replans: int = 5,
    max_steps_per_plan: int = 500,
) -> None:
    """Run pure planning baseline on CleanupTable environment with
    replanning."""
    print("\n" + "=" * 80)
    print("Running Pure Planning on CleanupTable Environment (with replanning)")
    print("=" * 80)

    system = BaseCleanupTableTAMPSystem.create_default(
        seed=seed,
        render_mode=render_mode,
    )

    planner = TaskThenMotionPlanner(
        system.types,
        system.predicates,
        system.perceiver,
        system.operators,
        system.skills,
        planner_id="pyperplan",
    )

    obs, info = system.env.reset(seed=seed)

    total_steps = 0
    for replan_attempt in range(max_replans):
        print(f"\nPlanning attempt {replan_attempt + 1}/{max_replans}")

        planner.reset(obs, info)
        steps_in_current_plan = 0

        for _ in range(max_steps_per_plan):
            steps_in_current_plan += 1
            total_steps += 1

            try:
                action = planner.step(obs)
            except Exception as e:
                print(f"Planner failed with exception: {e}. Replanning...")
                break

            obs, reward, done, _, info = system.env.step(action)
            if done:
                print(f"\nGoal reached in {total_steps} total steps!")
                print(
                    f"Steps in final plan: {steps_in_current_plan} "
                    f"(attempt {replan_attempt + 1})"
                )
                print(f"Final reward: {reward}")
                if float(reward) > 0:
                    print("SUCCESS: Task completed successfully!")
                system.env.close()  # type: ignore[no-untyped-call]
                return

            if total_steps >= max_steps:
                system.env.close()  # type: ignore[no-untyped-call]
                return

    print(f"\nFAILED: Goal not reached after {max_replans} planning attempts")
    system.env.close()  # type: ignore[no-untyped-call]


def main() -> None:
    """Main function to run pure planning baselines."""
    parser = argparse.ArgumentParser(
        description="Run pure planning baselines on PyBullet environments"
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=[
            "obstacle_tower",
            "cluttered_drawer",
            "cleanup_table",
        ],
        help="Environment to run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering (rgb_array mode)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Maximum steps per episode",
    )

    args = parser.parse_args()
    render_mode = "rgb_array" if args.render else None

    if args.env == "obstacle_tower":
        run_obstacle_tower_planning(
            seed=args.seed,
            render_mode=render_mode,
            max_steps=args.max_steps,
        )
    elif args.env == "cluttered_drawer":
        run_cluttered_drawer_planning(
            seed=args.seed,
            render_mode=render_mode,
            max_steps=args.max_steps,
        )
    elif args.env == "cleanup_table":
        run_cleanup_table_planning(
            seed=args.seed,
            render_mode=render_mode,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()  # type: ignore[no-untyped-call]
