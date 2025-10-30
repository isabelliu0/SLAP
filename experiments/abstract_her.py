"""Training script for Abstract HER shortcut learning scheme."""

from pathlib import Path

from tamp_improv.approaches.improvisational.policies.goal_rl import (
    GoalConditionedRLConfig,
    GoalConditionedRLPolicy,
)
from tamp_improv.approaches.improvisational.training import (
    TrainingConfig,
    train_and_evaluate_goal_conditioned,
)
from tamp_improv.benchmarks.obstacle2d import Obstacle2DTAMPSystem
from tamp_improv.benchmarks.pybullet_cleanup_table import CleanupTableTAMPSystem
from tamp_improv.benchmarks.pybullet_cluttered_drawer import ClutteredDrawerTAMPSystem
from tamp_improv.benchmarks.pybullet_obstacle_tower_graph import (
    GraphObstacleTowerTAMPSystem,
)


def train_abstract_her_obstacle2d(
    seed: int = 42,
    algorithm: str = "SAC",
    render: bool = False,
    episodes_per_scenario: int = 250,
    save_dir: str = "trained_policies/abstract_her",
):
    """Train Abstract HER on Obstacle2D."""
    print("\n=== Training Abstract HER on Obstacle2D ===")

    config = TrainingConfig(
        seed=seed,
        num_episodes=5,
        max_steps=50,
        collect_episodes=5,
        episodes_per_scenario=episodes_per_scenario,
        render=render,
        record_training=False,
        training_record_interval=100,
        training_data_dir="training_data/abstract_her",
        save_dir=save_dir,
        max_atom_size=14,
        success_threshold=0.01,
        success_reward=10.0,
        step_penalty=-0.5,
    )

    system = Obstacle2DTAMPSystem.create_default(
        seed=config.seed, render_mode="rgb_array" if config.render else None
    )

    def policy_factory(seed):
        return GoalConditionedRLPolicy(
            seed=seed,
            config=GoalConditionedRLConfig(
                algorithm=algorithm,
                batch_size=64,
                buffer_size=1000,
                n_sampled_goal=4,
            ),
        )

    metrics = train_and_evaluate_goal_conditioned(
        system,
        policy_factory,
        config,
        policy_name=f"AbstractHER_{algorithm}",
        use_atom_as_obs=False,
        use_random_rollouts=True,
        num_rollouts_per_node=1000,
        max_steps_per_rollout=100,
        shortcut_success_threshold=1,
    )

    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")

    results_file = (
        Path(save_dir) / f"Obstacle2DTAMPSystem_AbstractHER_{algorithm}" / "results.txt"
    )
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("Environment: Obstacle2D\n")
        f.write(f"seed: {seed}\n")
        f.write(f"algorithm: {algorithm}\n")
        f.write(f"success_rate: {metrics.success_rate:.4f}\n")
        f.write(f"avg_episode_length: {metrics.avg_episode_length:.2f}\n")

    return metrics


def train_abstract_her_pybullet(
    system_cls,
    seed: int = 42,
    algorithm: str = "SAC",
    render: bool = False,
    episodes_per_scenario: int = 50,
    save_dir: str = "trained_policies/abstract_her",
    max_atom_size: int = 42,
):
    """Train Abstract HER on PyBullet environments."""
    print(f"\n=== Training Abstract HER on {system_cls.__name__} ===")

    config = TrainingConfig(
        seed=seed,
        num_episodes=1,
        max_steps=500,
        collect_episodes=1,
        episodes_per_scenario=episodes_per_scenario,
        render=render,
        record_training=False,
        training_record_interval=100,
        training_data_dir="training_data/abstract_her",
        save_dir=save_dir,
        max_atom_size=max_atom_size,
        success_threshold=0.01,
        success_reward=10.0,
        step_penalty=-0.5,
    )

    system = system_cls.create_default(
        seed=config.seed, render_mode="rgb_array" if config.render else None
    )

    def policy_factory(seed):
        return GoalConditionedRLPolicy(
            seed=seed,
            config=GoalConditionedRLConfig(
                algorithm=algorithm,
                batch_size=64,
                buffer_size=1000,
                n_sampled_goal=4,
            ),
        )

    metrics = train_and_evaluate_goal_conditioned(
        system,
        policy_factory,
        config,
        policy_name=f"AbstractHER_{algorithm}",
        use_atom_as_obs=False,
        use_random_rollouts=True,
        num_rollouts_per_node=100,
        max_steps_per_rollout=300,
        shortcut_success_threshold=5,
    )

    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")

    results_file = (
        Path(save_dir)
        / f"{system_cls.__name__}_AbstractHER_{algorithm}"
        / "results.txt"
    )
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(f"Environment: {system_cls.__name__}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"algorithm: {algorithm}\n")
        f.write(f"success_rate: {metrics.success_rate:.4f}\n")
        f.write(f"avg_episode_length: {metrics.avg_episode_length:.2f}\n")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Abstract HER shortcut learning scheme"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="obstacle2d",
        choices=["obstacle2d", "obstacle_tower", "cluttered_drawer", "cleanup_table"],
        help="Environment to train on",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="SAC",
        choices=["SAC"],
        help="RL algorithm to use",
    )
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="trained_policies/abstract_her",
        help="Directory to save trained policies",
    )

    args = parser.parse_args()

    if args.env == "obstacle2d":
        train_abstract_her_obstacle2d(
            seed=args.seed,
            algorithm=args.algorithm,
            render=args.render,
            save_dir=args.save_dir,
        )
    elif args.env == "obstacle_tower":
        train_abstract_her_pybullet(
            system_cls=GraphObstacleTowerTAMPSystem,
            seed=args.seed,
            algorithm=args.algorithm,
            render=args.render,
            save_dir=args.save_dir,
            max_atom_size=42,
        )
    elif args.env == "cluttered_drawer":
        train_abstract_her_pybullet(
            system_cls=ClutteredDrawerTAMPSystem,
            seed=args.seed,
            algorithm=args.algorithm,
            render=args.render,
            save_dir=args.save_dir,
            max_atom_size=72,
        )
    elif args.env == "cleanup_table":
        train_abstract_her_pybullet(
            system_cls=CleanupTableTAMPSystem,
            seed=args.seed,
            algorithm=args.algorithm,
            render=args.render,
            save_dir=args.save_dir,
            max_atom_size=60,
        )
    else:
        raise ValueError(f"Unknown environment: {args.env}")
