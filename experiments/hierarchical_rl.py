"""Training script for Hierarchical RL baseline."""

from pathlib import Path

import torch

from tamp_improv.approaches.improvisational.policies.rl import RLConfig, RLPolicy
from tamp_improv.approaches.improvisational.training import (
    TrainingConfig,
    train_and_evaluate_rl_baseline,
)
from tamp_improv.benchmarks.obstacle2d import Obstacle2DTAMPSystem
from tamp_improv.benchmarks.pybullet_cleanup_table import CleanupTableTAMPSystem
from tamp_improv.benchmarks.pybullet_cluttered_drawer import ClutteredDrawerTAMPSystem
from tamp_improv.benchmarks.pybullet_obstacle_tower_graph import (
    GraphObstacleTowerTAMPSystem,
)


def train_hierarchical_rl_obstacle2d(
    seed: int = 42,
    render: bool = False,
    save_dir: str = "trained_policies/hierarchical_rl",
    single_step_skills: bool = False,
):
    """Train Hierarchical RL baseline on Obstacle2D."""
    print("\n=== Training Hierarchical RL on Obstacle2D ===")

    config = TrainingConfig(
        seed=seed,
        num_episodes=5,
        max_steps=100,
        render=render,
        record_training=False,
        training_record_interval=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
        step_penalty=-0.1,
        success_reward=10.0,
        action_scale=1.0,
    )

    rl_config = RLConfig(
        learning_rate=3e-4,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.05,
        device=config.device,
    )

    system = Obstacle2DTAMPSystem.create_default(
        seed=config.seed,
        render_mode="rgb_array" if config.render else None,
    )

    def policy_factory(seed: int) -> RLPolicy:
        return RLPolicy(seed=seed, config=rl_config)

    metrics = train_and_evaluate_rl_baseline(
        system,
        policy_factory,
        config,
        policy_name="HierarchicalRL",
        baseline_type="hierarchical",
        single_step_skills=single_step_skills,
        max_skill_steps=20,
        skill_failure_penalty=-1.0,
    )

    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")

    results_file = (
        Path(save_dir) / "Obstacle2DTAMPSystem_HierarchicalRL" / "results.txt"
    )
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("Environment: Obstacle2D\n")
        f.write(f"seed: {seed}\n")
        f.write(f"single_step_skills: {single_step_skills}\n")
        f.write(f"success_rate: {metrics.success_rate:.4f}\n")
        f.write(f"avg_episode_length: {metrics.avg_episode_length:.2f}\n")

    return metrics


def train_hierarchical_rl_pybullet(
    system_cls,
    seed: int = 42,
    render: bool = False,
    save_dir: str = "trained_policies/hierarchical_rl",
    action_scale: float = 0.015,
    single_step_skills: bool = False,
):
    """Train Hierarchical RL baseline on PyBullet environments."""
    print(f"\n=== Training Hierarchical RL on {system_cls.__name__} ===")

    config = TrainingConfig(
        seed=seed,
        num_episodes=1,
        max_steps=500,
        render=render,
        record_training=False,
        training_record_interval=200,
        device="cuda" if torch.cuda.is_available() else "cpu",
        step_penalty=-1.0,
        success_reward=100.0,
        action_scale=action_scale,
    )

    rl_config = RLConfig(
        learning_rate=3e-4,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.05,
        device=config.device,
    )

    system = system_cls.create_default(
        seed=config.seed,
        render_mode="rgb_array" if config.render else None,
    )

    def policy_factory(seed: int) -> RLPolicy:
        return RLPolicy(seed=seed, config=rl_config)

    metrics = train_and_evaluate_rl_baseline(
        system,
        policy_factory,
        config,
        policy_name="HierarchicalRL",
        baseline_type="hierarchical",
        single_step_skills=single_step_skills,
        max_skill_steps=200,
        skill_failure_penalty=-1.0,
    )

    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")

    results_file = (
        Path(save_dir) / f"{system_cls.__name__}_HierarchicalRL" / "results.txt"
    )
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(f"Environment: {system_cls.__name__}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"single_step_skills: {single_step_skills}\n")
        f.write(f"success_rate: {metrics.success_rate:.4f}\n")
        f.write(f"avg_episode_length: {metrics.avg_episode_length:.2f}\n")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Hierarchical RL baseline")
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
        default="trained_policies/hierarchical_rl",
        help="Directory to save trained policies",
    )
    parser.add_argument(
        "--single-step-skills",
        action="store_true",
        help="Use single-step skills instead of multi-step",
    )

    args = parser.parse_args()

    if args.env == "obstacle2d":
        train_hierarchical_rl_obstacle2d(
            seed=args.seed,
            render=args.render,
            save_dir=args.save_dir,
            single_step_skills=args.single_step_skills,
        )
    elif args.env == "obstacle_tower":
        train_hierarchical_rl_pybullet(
            system_cls=GraphObstacleTowerTAMPSystem,
            seed=args.seed,
            render=args.render,
            save_dir=args.save_dir,
            action_scale=0.015,
            single_step_skills=args.single_step_skills,
        )
    elif args.env == "cluttered_drawer":
        train_hierarchical_rl_pybullet(
            system_cls=ClutteredDrawerTAMPSystem,
            seed=args.seed,
            render=args.render,
            save_dir=args.save_dir,
            action_scale=0.005,
            single_step_skills=args.single_step_skills,
        )
    elif args.env == "cleanup_table":
        train_hierarchical_rl_pybullet(
            system_cls=CleanupTableTAMPSystem,
            seed=args.seed,
            render=args.render,
            save_dir=args.save_dir,
            action_scale=0.005,
            single_step_skills=args.single_step_skills,
        )
