"""Evaluation script for trained Multi-RL policies using Hydra."""

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
    """Main evaluation function."""
    if not hasattr(cfg, "policy_path"):
        raise ValueError("policy_path must be provided")

    policy_path = Path(cfg.policy_path)
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy not found at {policy_path}")

    print("=" * 80)
    print(f"Evaluating Multi-RL on {cfg.env_name}")
    print(f"Policy: {policy_path}")
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

    training_config = TrainingConfig(
        seed=cfg.seed,
        num_episodes=cfg.num_episodes,
        max_steps=cfg.max_steps,
        render=cfg.render,
        collect_episodes=0,
        episodes_per_scenario=0,
        force_collect=False,
        record_training=False,
        training_data_dir=cfg.training_data_dir,
        save_dir=cfg.save_dir,
        fast_eval=True,
    )

    def policy_factory(seed: int) -> MultiRLPolicy[Any, Any]:
        policy: MultiRLPolicy[Any, Any] = MultiRLPolicy(seed=seed, config=rl_config)
        policy.load(str(policy_path))
        return policy

    metrics = train_and_evaluate(
        system,
        policy_factory,
        training_config,
        policy_name="MultiRL_Loaded",
    )

    print("\n" + "=" * 80)
    print("Evaluation Results:")
    print("=" * 80)
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Average Episode Length: {metrics.avg_episode_length:.2f}")
    print("=" * 80)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    results_file = output_dir / "eval_results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(f"env_name: {cfg.env_name}\n")
        f.write(f"policy_path: {policy_path}\n")
        f.write(f"seed: {cfg.seed}\n")
        f.write(f"success_rate: {metrics.success_rate}\n")
        f.write(f"avg_episode_length: {metrics.avg_episode_length}\n")

    return metrics.success_rate


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
