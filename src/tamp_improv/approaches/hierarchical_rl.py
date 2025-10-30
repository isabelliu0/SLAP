"""Hierarchical RL approach that combines low-level actions with TAMP
skills."""

from typing import Any, TypeVar

from tamp_improv.approaches.base import ApproachStepResult, BaseApproach
from tamp_improv.approaches.improvisational.policies.base import Policy
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem
from tamp_improv.benchmarks.hierarchical_wrapper import HierarchicalRLWrapper

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class HierarchicalRLApproach(BaseApproach[ObsType, ActType]):
    """Hierarchical RL approach that uses both low-level actions and TAMP
    skills."""

    def __init__(
        self,
        system: ImprovisationalTAMPSystem[ObsType, ActType],
        policy: Policy[ObsType, ActType],
        seed: int,
        hierarchical_env: HierarchicalRLWrapper,
    ) -> None:
        """Initialize approach."""
        super().__init__(system, seed)
        self.policy = policy
        self.hierarchical_env = hierarchical_env

    def reset(self, obs: ObsType, info: dict[str, Any]) -> ApproachStepResult[ActType]:
        """Reset approach with initial observation."""
        wrapped_obs: ObsType
        wrapped_obs, wrapped_info = self.hierarchical_env.reset(seed=self._seed)
        return self.step(wrapped_obs, 0.0, False, False, wrapped_info)

    def step(
        self,
        obs: ObsType,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ApproachStepResult[ActType]:
        """Step approach with new observation."""
        action = self.policy.get_action(obs)
        return ApproachStepResult(action=action)
