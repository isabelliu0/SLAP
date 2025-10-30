"""Pure RL baseline approach without using TAMP structure."""

from typing import Any, TypeVar

from tamp_improv.approaches.base import ApproachStepResult, BaseApproach
from tamp_improv.approaches.improvisational.policies.base import Policy
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem
from tamp_improv.benchmarks.sac_her_wrapper import SACHERWrapper

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class PureRLApproach(BaseApproach[ObsType, ActType]):
    """Pure RL approach that learns the entire task end-to-end."""

    def __init__(
        self,
        system: ImprovisationalTAMPSystem[ObsType, ActType],
        policy: Policy[ObsType, ActType],
        seed: int,
    ) -> None:
        """Initialize approach."""
        super().__init__(system, seed)
        self.policy = policy

    def reset(self, obs: ObsType, info: dict[str, Any]) -> ApproachStepResult[ActType]:
        """Reset approach with initial observation."""
        return self.step(obs, 0.0, False, False, info)

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


class SACHERApproach(BaseApproach[ObsType, ActType]):
    """SAC+HER approach that learns the entire task end-to-end with HER."""

    def __init__(
        self,
        system: ImprovisationalTAMPSystem[ObsType, ActType],
        policy: Policy[ObsType, ActType],
        seed: int,
    ) -> None:
        """Initialize approach."""
        super().__init__(system, seed)
        self.policy = policy

    def reset(self, obs: ObsType, info: dict[str, Any]) -> ApproachStepResult[ActType]:
        """Reset approach with initial observation."""
        return self.step(obs, 0.0, False, False, info)

    def step(
        self,
        obs: ObsType,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ApproachStepResult[ActType]:
        """Step approach with new observation."""
        if isinstance(self.system.wrapped_env, SACHERWrapper):
            if not isinstance(obs, dict):
                atoms = self.system.wrapped_env.perceiver.step(obs)
                current_atom_vector = self.system.wrapped_env.create_atom_vector(atoms)
                goal_atom_vector = self.system.wrapped_env.goal_atom_vector
                formatted_obs = {
                    "observation": self.system.wrapped_env.flatten_obs(obs),
                    "achieved_goal": current_atom_vector,
                    "desired_goal": goal_atom_vector,
                }
                action = self.policy.get_action(formatted_obs)  # type: ignore[arg-type]
            else:
                action = self.policy.get_action(obs)  # type: ignore[arg-type]
        else:
            action = self.policy.get_action(obs)
        return ApproachStepResult(action=action)
