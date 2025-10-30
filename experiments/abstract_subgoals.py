"""Training script for Abstract Subgoals shortcut learning scheme."""

from pathlib import Path

import torch

from tamp_improv.approaches.improvisational.policies.rl import RLConfig, RLPolicy
from tamp_improv.approaches.improvisational.training import (
    TrainingConfig,
    train_and_evaluate,
)
from tamp_improv.benchmarks.obstacle2d import Obstacle2DTAMPSystem
from tamp_improv.benchmarks.pybullet_cleanup_table import CleanupTableTAMPSystem
from tamp_improv.benchmarks.pybullet_cluttered_drawer import ClutteredDrawerTAMPSystem
from tamp_improv.benchmarks.pybullet_obstacle_tower_graph import (
    GraphObstacleTowerTAMPSystem,
)


def train_abstract_subgoals_obstacle2d(
    seed: int = 42,
    render: bool = False,
    collect_episodes: int = 3,
    episodes_per_scenario: int = 1000,
    save_dir: str = "trained_policies/abstract_subgoals",
):
    """Train Abstract Subgoals on Obstacle2D."""
    print("\n=== Training Abstract Subgoals on Obstacle2D ===")

    config = TrainingConfig(
        seed=seed,
        num_episodes=5,
        max_steps=50,
        max_training_steps_per_shortcut=50,
        collect_episodes=collect_episodes,
        episodes_per_scenario=episodes_per_scenario,
        force_collect=False,
        render=render,
        record_training=False,
        training_record_interval=125,
        training_data_dir="training_data/abstract_subgoals",
        save_dir=save_dir,
        batch_size=32,
        max_atom_size=14,
    )

    rl_config = RLConfig(
        learning_rate=3e-4,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.05,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    system = Obstacle2DTAMPSystem.create_default(
        seed=config.seed,
        render_mode="rgb_array" if config.render else None,
    )

    def policy_factory(seed: int) -> RLPolicy:
        return RLPolicy(seed=seed, config=rl_config)

    metrics = train_and_evaluate(
        system,
        policy_factory,
        config,
        policy_name="AbstractSubgoals",
        use_context_wrapper=True,
        use_random_rollouts=True,
        num_rollouts_per_node=1000,
        max_steps_per_rollout=100,
        shortcut_success_threshold=1,
    )

    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")

    results_file = (
        Path(save_dir) / "Obstacle2DTAMPSystem_AbstractSubgoals" / "results.txt"
    )
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("Environment: Obstacle2D\n")
        f.write(f"seed: {seed}\n")
        f.write(f"success_rate: {metrics.success_rate:.4f}\n")
        f.write(f"avg_episode_length: {metrics.avg_episode_length:.2f}\n")

    return metrics


def train_abstract_subgoals_pybullet(
    system_cls,
    seed: int = 42,
    render: bool = False,
    episodes_per_scenario: int = 3000,
    save_dir: str = "trained_policies/abstract_subgoals",
    action_scale: float = 0.015,
    max_atom_size: int = 42,
):
    """Train Abstract Subgoals on PyBullet environments."""
    print(f"\n=== Training Abstract Subgoals on {system_cls.__name__} ===")

    config = TrainingConfig(
        seed=seed,
        num_episodes=1,
        max_steps=500,
        max_training_steps_per_shortcut=100,
        collect_episodes=1,
        episodes_per_scenario=episodes_per_scenario,
        force_collect=False,
        render=render,
        record_training=False,
        training_record_interval=100,
        training_data_dir="training_data/abstract_subgoals",
        save_dir=save_dir,
        batch_size=16,
        max_atom_size=max_atom_size,
        action_scale=action_scale,
    )

    rl_config = RLConfig(
        learning_rate=3e-4,
        batch_size=16,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    system = system_cls.create_default(
        seed=config.seed, render_mode="rgb_array" if config.render else None
    )

    def policy_factory(seed: int) -> RLPolicy:
        return RLPolicy(seed=seed, config=rl_config)

    metrics = train_and_evaluate(
        system,
        policy_factory,
        config,
        policy_name="AbstractSubgoals",
        use_context_wrapper=True,
        use_random_rollouts=True,
        num_rollouts_per_node=100,
        max_steps_per_rollout=300,
        shortcut_success_threshold=5,
    )

    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")

    results_file = (
        Path(save_dir) / f"{system_cls.__name__}_AbstractSubgoals" / "results.txt"
    )
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(f"Environment: {system_cls.__name__}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"success_rate: {metrics.success_rate:.4f}\n")
        f.write(f"avg_episode_length: {metrics.avg_episode_length:.2f}\n")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Abstract Subgoals shortcut learning scheme"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="obstacle2d",
        choices=["obstacle2d", "obstacle_tower", "cluttered_drawer", "cleanup_table"],
        help="Environment to train on",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="trained_policies/abstract_subgoals",
        help="Directory to save trained policies",
    )

    args = parser.parse_args()

    if args.env == "obstacle2d":
        train_abstract_subgoals_obstacle2d(
            seed=args.seed,
            render=args.render,
            save_dir=args.save_dir,
        )
    elif args.env == "obstacle_tower":
        train_abstract_subgoals_pybullet(
            system_cls=GraphObstacleTowerTAMPSystem,
            seed=args.seed,
            render=args.render,
            save_dir=args.save_dir,
            action_scale=0.015,
            max_atom_size=42,
        )
    elif args.env == "cluttered_drawer":
        train_abstract_subgoals_pybullet(
            system_cls=ClutteredDrawerTAMPSystem,
            seed=args.seed,
            render=args.render,
            save_dir=args.save_dir,
            action_scale=0.015,
            max_atom_size=72,
        )
    elif args.env == "cleanup_table":
        train_abstract_subgoals_pybullet(
            system_cls=CleanupTableTAMPSystem,
            seed=args.seed,
            render=args.render,
            save_dir=args.save_dir,
            action_scale=0.005,
            max_atom_size=60,
        )
