"""Base policy interface."""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

import gymnasium as gym
from relational_structs import GroundAtom

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class TrainingData:
    """Container for policy training data."""

    states: list[Any]
    current_atoms: list[set[GroundAtom]]
    goal_atoms: list[set[GroundAtom]]
    config: dict[str, Any]

    def __len__(self) -> int:
        return len(self.states)

    def save(self, path: Path) -> None:
        """Save training data to disk."""
        path.mkdir(parents=True, exist_ok=True)
        states_path = path / "states.pkl"
        with open(states_path, "wb") as f:
            pickle.dump(self.states, f)

        data_paths = {
            "current_atoms": self.current_atoms,
            "goal_atoms": self.goal_atoms,
        }
        for name, obj in data_paths.items():
            file_path = path / f"{name}.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(obj, f)

        serializable_config = dict(self.config)

        if "atom_to_index" in serializable_config:
            atom_to_index = {
                str(k): v for k, v in serializable_config["atom_to_index"].items()
            }
            serializable_config["atom_to_index"] = atom_to_index

        config_path = path / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(serializable_config, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> TrainingData:
        """Load training data from disk."""
        states_path = path / "states.pkl"
        with open(states_path, "rb") as f:
            states = pickle.load(f)

        data = {}
        data_names = [
            "current_atoms",
            "goal_atoms",
        ]
        for name in data_names:
            file_path = path / f"{name}.pkl"
            if file_path.exists():
                with open(file_path, "rb") as f:
                    data[name] = pickle.load(f)
            else:
                print(f"Warning: {file_path} not found, using empty list.")
                data[name] = []

        config_path = path / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if "atom_to_index" in config:
            config["atom_to_index"] = {
                k: int(v) if isinstance(v, str) else v
                for k, v in config["atom_to_index"].items()
            }

        return cls(
            states=states,
            current_atoms=data["current_atoms"],
            goal_atoms=data["goal_atoms"],
            config=config,
        )


@dataclass
class GoalConditionedTrainingData(TrainingData, Generic[ObsType]):
    """Training data for goal-conditioned learning."""

    node_states: dict[int, ObsType] = field(default_factory=dict)
    valid_shortcuts: list[tuple[int, int]] = field(default_factory=list)
    node_atoms: dict[int, set[GroundAtom]] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Save training data including node states."""
        super().save(path)
        if self.node_states:
            with open(path / "node_states.pkl", "wb") as f:
                pickle.dump(self.node_states, f)
        if self.valid_shortcuts:
            with open(path / "valid_shortcuts.pkl", "wb") as f:
                pickle.dump(self.valid_shortcuts, f)
        if self.node_atoms:
            with open(path / "node_atoms.pkl", "wb") as f:
                pickle.dump(self.node_atoms, f)

    @classmethod
    def load(cls, path: Path) -> GoalConditionedTrainingData:
        """Load training data including node states."""
        train_data = super().load(path)
        node_states: dict[int, ObsType] = {}
        if (path / "node_states.pkl").exists():
            with open(path / "node_states.pkl", "rb") as f:
                node_states = pickle.load(f)
        valid_shortcuts: list[tuple[int, int]] = []
        if (path / "valid_shortcuts.pkl").exists():
            with open(path / "valid_shortcuts.pkl", "rb") as f:
                valid_shortcuts = pickle.load(f)
        node_atoms: dict[int, set[GroundAtom]] = {}
        if (path / "node_atoms.pkl").exists():
            with open(path / "node_atoms.pkl", "rb") as f:
                node_atoms = pickle.load(f)
        return cls(
            states=train_data.states,
            current_atoms=train_data.current_atoms,
            goal_atoms=train_data.goal_atoms,
            config=train_data.config,
            node_states=node_states,
            valid_shortcuts=valid_shortcuts,
            node_atoms=node_atoms,
        )


@dataclass
class PolicyContext(Generic[ObsType, ActType]):
    """Context information passed from approach to policy."""

    goal_atoms: set[GroundAtom]
    current_atoms: set[GroundAtom]
    info: dict[str, Any] = field(default_factory=dict)


class Policy(Generic[ObsType, ActType], ABC):
    """Base class for policies."""

    node_states: dict[int, ObsType]

    def __init__(self, seed: int) -> None:
        """Initialize policy with environment."""
        self._seed = seed

    @abstractmethod
    def initialize(self, env: gym.Env) -> None:
        """Initialize policy with environment."""

    @abstractmethod
    def can_initiate(self) -> bool:
        """Check whether the policy can be executed given the current
        context."""

    @abstractmethod
    def get_action(self, obs: ObsType) -> ActType:
        """Get action from policy."""

    def configure_context(self, context: PolicyContext[ObsType, ActType]) -> None:
        """Configure policy with context information."""

    def train(self, env: gym.Env, train_data: TrainingData | None) -> None:
        """Train the policy if needed.

        Default implementation just initializes the policy and updates
        context. Policies that need training should override this.
        """
        self.initialize(env)
        if hasattr(env, "configure_training"):
            env.configure_training(train_data)

    @abstractmethod
    def save(self, path: str) -> None:
        """Save policy to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load policy from disk."""
