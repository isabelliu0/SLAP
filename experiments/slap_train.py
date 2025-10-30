"""Training script for Multi-RL method using Hydra."""

from pathlib import Path
from typing import Any, Type

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy
from tamp_improv.approaches.improvisational.policies.rl import RLConfig
from tamp_improv.approaches.improvisational.training import (
    TrainingConfig,
    train_and_evaluate,
)
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem
from tamp_improv.benchmarks.obstacle2d_graph import GraphObstacle2DTAMPSystem
from tamp_improv.benchmarks.pybullet_cleanup_table import CleanupTableTAMPSystem
from tamp_improv.benchmarks.pybullet_cluttered_drawer import ClutteredDrawerTAMPSystem
from tamp_improv.benchmarks.pybullet_obstacle_tower_graph import (
    GraphObstacleTowerTAMPSystem,
)

SYSTEM_CLASSES: dict[str, Type[ImprovisationalTAMPSystem[Any, Any]]] = {
    "GraphObstacle2DTAMPSystem": GraphObstacle2DTAMPSystem,
    "GraphObstacleTowerTAMPSystem": GraphObstacleTowerTAMPSystem,
    "ClutteredDrawerTAMPSystem": ClutteredDrawerTAMPSystem,
    "CleanupTableTAMPSystem": CleanupTableTAMPSystem,
}


@hydra.main(version_base=None, config_path="configs", config_name="obstacle_tower")
def main(cfg: DictConfig) -> float:
    """Main training function."""
    print("=" * 80)
    print(f"Training Multi-RL on {cfg.env_name}")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    system_cls = SYSTEM_CLASSES[cfg.env_name]
    system_kwargs = {
        "seed": cfg.seed,
        "render_mode": cfg.render_mode if cfg.render_mode != "null" else None,
    }
    if hasattr(cfg, "num_obstacle_blocks"):
        system_kwargs["num_obstacle_blocks"] = cfg.num_obstacle_blocks
    system = system_cls.create_default(**system_kwargs)  # type: ignore[attr-defined]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rl_config = RLConfig(
        learning_rate=cfg.learning_rate,
        batch_size=cfg.rl_batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        ent_coef=cfg.ent_coef,
        deterministic=cfg.deterministic,
        device=device,
    )
    print(f"\nUsing device: {device}")

    training_config = TrainingConfig(
        seed=cfg.seed,
        num_episodes=cfg.num_episodes,
        max_steps=cfg.max_steps,
        max_training_steps_per_shortcut=cfg.max_training_steps_per_shortcut,
        collect_episodes=cfg.collect_episodes,
        episodes_per_scenario=cfg.episodes_per_scenario,
        force_collect=cfg.force_collect,
        render=cfg.render,
        record_training=False,
        training_record_interval=cfg.training_record_interval,
        training_data_dir=cfg.training_data_dir,
        save_dir=cfg.save_dir,
        batch_size=cfg.batch_size,
        action_scale=cfg.action_scale,
    )

    metrics = train_and_evaluate(
        system,
        lambda seed: MultiRLPolicy(seed=seed, config=rl_config),
        training_config,
        policy_name="MultiRL",
        use_context_wrapper=False,
        use_random_rollouts=cfg.use_random_rollouts,
        num_rollouts_per_node=cfg.num_rollouts_per_node,
        max_steps_per_rollout=cfg.max_steps_per_rollout,
        shortcut_success_threshold=cfg.shortcut_success_threshold,
    )

    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print("=" * 80)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    results_file = output_dir / "results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(f"env_name: {cfg.env_name}\n")
        f.write(f"seed: {cfg.seed}\n")
        f.write(f"success_rate: {metrics.success_rate}\n")
        f.write(f"avg_episode_length: {metrics.avg_episode_length}\n")

    return metrics.success_rate


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
