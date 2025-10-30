"""Training script for SAC+HER baseline."""

from pathlib import Path

import torch

from tamp_improv.approaches.improvisational.policies.sac_her import (
    SACHERConfig,
    SACHERPolicy,
)
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


def train_sac_her_obstacle2d(
    seed: int = 42,
    render: bool = False,
    save_dir: str = "trained_policies/sac_her",
):
    """Train SAC+HER baseline on Obstacle2D."""
    print("\n=== Training SAC+HER Baseline on Obstacle2D ===")

    config = TrainingConfig(
        seed=seed,
        num_episodes=5,
        max_steps=100,
        render=render,
        record_training=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        step_penalty=-0.1,
        success_reward=10.0,
    )

    sac_her_config = SACHERConfig(
        learning_rate=3e-4,
        batch_size=32,
        device=config.device,
    )

    system = Obstacle2DTAMPSystem.create_default(
        seed=config.seed,
        render_mode="rgb_array" if config.render else None,
    )

    def policy_factory(seed: int) -> SACHERPolicy:
        return SACHERPolicy(seed=seed, config=sac_her_config)

    metrics = train_and_evaluate_rl_baseline(
        system,
        policy_factory,
        config,
        policy_name="SAC_HER",
        baseline_type="sac_her",
    )

    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")

    results_file = Path(save_dir) / "Obstacle2DTAMPSystem_SAC_HER" / "results.txt"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("Environment: Obstacle2D\n")
        f.write(f"seed: {seed}\n")
        f.write(f"success_rate: {metrics.success_rate:.4f}\n")
        f.write(f"avg_episode_length: {metrics.avg_episode_length:.2f}\n")

    return metrics


def train_sac_her_pybullet(
    system_cls,
    seed: int = 42,
    render: bool = False,
    save_dir: str = "trained_policies/sac_her",
    action_scale: float = 0.015,
    max_atom_size: int = 42,
):
    """Train SAC+HER baseline on PyBullet environments."""
    print(f"\n=== Training SAC+HER Baseline on {system_cls.__name__} ===")

    config = TrainingConfig(
        seed=seed,
        num_episodes=1,
        max_steps=500,
        render=render,
        record_training=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        step_penalty=-1.0,
        success_reward=100.0,
        action_scale=action_scale,
    )

    sac_her_config = SACHERConfig(
        learning_rate=3e-4,
        batch_size=32,
        device=config.device,
    )

    system = system_cls.create_default(
        seed=config.seed,
        render_mode="rgb_array" if config.render else None,
    )

    def policy_factory(seed: int) -> SACHERPolicy:
        return SACHERPolicy(seed=seed, config=sac_her_config)

    metrics = train_and_evaluate_rl_baseline(
        system,
        policy_factory,
        config,
        policy_name="SAC_HER",
        baseline_type="sac_her",
        max_atom_size=max_atom_size,
    )

    print("\n=== Results ===")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")

    results_file = Path(save_dir) / f"{system_cls.__name__}_SAC_HER" / "results.txt"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(f"Environment: {system_cls.__name__}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"success_rate: {metrics.success_rate:.4f}\n")
        f.write(f"avg_episode_length: {metrics.avg_episode_length:.2f}\n")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SAC+HER baseline")
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
        default="trained_policies/sac_her",
        help="Directory to save trained policies",
    )

    args = parser.parse_args()

    if args.env == "obstacle2d":
        train_sac_her_obstacle2d(
            seed=args.seed,
            render=args.render,
            save_dir=args.save_dir,
        )
    elif args.env == "obstacle_tower":
        train_sac_her_pybullet(
            system_cls=GraphObstacleTowerTAMPSystem,
            seed=args.seed,
            render=args.render,
            save_dir=args.save_dir,
            action_scale=0.015,
            max_atom_size=42,
        )
    elif args.env == "cluttered_drawer":
        train_sac_her_pybullet(
            system_cls=ClutteredDrawerTAMPSystem,
            seed=args.seed,
            render=args.render,
            save_dir=args.save_dir,
            action_scale=0.015,
            max_atom_size=72,
        )
    elif args.env == "cleanup_table":
        train_sac_her_pybullet(
            system_cls=CleanupTableTAMPSystem,
            seed=args.seed,
            render=args.render,
            save_dir=args.save_dir,
            action_scale=0.005,
            max_atom_size=60,
        )
