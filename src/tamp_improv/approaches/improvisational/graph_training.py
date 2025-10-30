"""Graph-based training data collection."""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, TypeVar

import gymnasium as gym
import numpy as np
from relational_structs import GroundAtom
from tomsutils.utils import sample_seed_from_rng

from tamp_improv.approaches.improvisational.base import (
    ImprovisationalTAMPApproach,
    ShortcutSignature,
)
from tamp_improv.approaches.improvisational.graph import (
    PlanningGraph,
    PlanningGraphEdge,
    PlanningGraphNode,
)
from tamp_improv.approaches.improvisational.policies.base import (
    GoalConditionedTrainingData,
    TrainingData,
)
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem

ObsType = TypeVar("ObsType")


@dataclass
class ShortcutCandidate:
    """Represents a potential shortcut in the planning graph."""

    source_node: PlanningGraphNode
    target_node: PlanningGraphNode
    source_atoms: set[GroundAtom]
    target_atoms: set[GroundAtom]
    source_state: Any


def collect_states_for_all_nodes(
    system, planning_graph: PlanningGraph, max_attempts: int = 10
) -> dict[int, ObsType]:
    """Collect observed states for all nodes in the planning graph."""
    observed_states: dict[int, ObsType] = {}

    initial_node = None
    if planning_graph.nodes:
        initial_node = planning_graph.nodes[0]
    assert initial_node is not None

    obs, info = system.reset()
    observed_states[initial_node.id] = obs

    remaining_nodes = [n for n in planning_graph.nodes if n.id != initial_node.id]

    for target_node in remaining_nodes:
        path = find_path_to_node(planning_graph, initial_node, target_node)
        if not path:
            print(f"No path found to node {target_node.id}, skipping")
            continue

        for _ in range(max_attempts):
            obs, info = system.reset()
            _ = system.perceiver.reset(obs, info)

            success = True
            for i, edge in enumerate(path):
                print(f"  Step {i+1}/{len(path)}: {edge.source.id} -> {edge.target.id}")

                if not edge.operator:
                    print("  No operator for this edge, skipping")
                    success = False
                    break

                skill = None
                for s in system.skills:
                    if s.can_execute(edge.operator):
                        skill = s
                        break
                if not skill:
                    print(
                        f"  No skill found for operator {edge.operator.name}, skipping"
                    )
                    success = False
                    break

                skill.reset(edge.operator)

                max_steps = 50
                for step in range(max_steps):
                    action = skill.get_action(obs)
                    obs, _, term, trunc, info = system.env.step(action)
                    atoms = system.perceiver.step(obs)
                    if set(edge.target.atoms) == atoms:
                        break
                    if term or trunc:
                        success = False
                        break
                    if step == max_steps - 1:
                        success = False

                if not success:
                    break

            if success:
                observed_states[target_node.id] = obs
                break

    return observed_states


def collect_node_states_for_shortcuts(
    system, planning_graph, max_attempts: int = 3
) -> tuple[dict[int, ObsType], list[tuple[int, int]]]:
    """Collect node states for valid shortcuts in the planning graph."""
    node_states: dict[int, ObsType] = collect_states_for_all_nodes(
        system, planning_graph, max_attempts
    )

    valid_shortcuts = []
    node_ids = sorted(list(node_states.keys()))
    for i, source_id in enumerate(node_ids):
        source_node = next((n for n in planning_graph.nodes if n.id == source_id), None)
        if not source_node:
            continue

        for target_id in node_ids[i + 1 :]:
            target_node = next(
                (n for n in planning_graph.nodes if n.id == target_id), None
            )
            if not target_node:
                continue
            has_direct_edge = False
            for edge in planning_graph.node_to_outgoing_edges.get(source_node, []):
                if edge.target == target_node and not edge.is_shortcut:
                    has_direct_edge = True
                    break
            if has_direct_edge:
                continue
            # Only include shortcuts where states are available
            if source_id in node_states and target_id in node_states:
                valid_shortcuts.append((source_id, target_id))

    print(f"Collected states for {len(node_states)} nodes")
    print(f"Identified {len(valid_shortcuts)} valid shortcuts")
    return node_states, valid_shortcuts


def find_path_to_node(
    planning_graph: PlanningGraph,
    start_node: PlanningGraphNode,
    target_node: PlanningGraphNode,
) -> list[PlanningGraphEdge]:
    """Find a path from start_node to target_node in the planning graph."""
    queue: deque[tuple[PlanningGraphNode, list[PlanningGraphEdge]]]
    queue = deque([(start_node, [])])
    visited = {start_node}
    while queue:
        current, path = queue.popleft()
        if current == target_node:
            return path
        for edge in planning_graph.node_to_outgoing_edges.get(current, []):
            if edge.is_shortcut:
                continue
            next_node = edge.target
            if next_node not in visited:
                visited.add(next_node)
                queue.append((next_node, path + [edge]))
    return []


def collect_graph_based_training_data(
    system: ImprovisationalTAMPSystem,
    approach: ImprovisationalTAMPApproach,
    config: dict[str, Any],
    max_shortcuts_per_graph: int = 150,
    use_random_rollouts: bool = False,
    num_rollouts_per_node: int = 50,
    max_steps_per_rollout: int = 50,
    shortcut_success_threshold: int = 1,
    action_scale: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[TrainingData, dict[int, list[ObsType]]]:
    """Collect training data by exploring the planning graph."""
    approach.training_mode = True
    training_states = []
    current_atoms_list = []
    goal_atoms_list = []
    shortcut_info = []
    collect_episodes = config.get("collect_episodes", 10)
    seed = config.get("seed", 42)
    rng = np.random.default_rng(seed)

    for episode in range(collect_episodes):
        print(f"\n=== Building planning graph for episode {episode + 1} ===")
        episode_seed = sample_seed_from_rng(rng)
        obs, info = system.reset()
        _ = approach.reset(obs, info)
        assert (
            hasattr(approach, "planning_graph") and approach.planning_graph is not None
        )
        planning_graph = approach.planning_graph

        if (
            hasattr(approach, "observed_states")
            and approach.observed_states is not None
        ):
            print("Using observed states already collected from approach")
            observed_states = approach.observed_states
        else:
            print(
                "Warning: No observed states found in approach. Falling back to another round of collection."  # pylint: disable=line-too-long
            )
            observed_states = collect_states_for_all_nodes(
                system, planning_graph, max_attempts=3
            )
            observed_states = {
                k: [v] for k, v in observed_states.items()
            }  # multi-state format

        if use_random_rollouts:
            shortcut_candidates = identify_promising_shortcuts_with_rollouts(
                system,
                planning_graph,
                observed_states,
                num_rollouts_per_node,
                max_steps_per_rollout,
                shortcut_success_threshold,
                action_scale=action_scale,
                random_seed=episode_seed,
            )
        else:
            shortcut_candidates = identify_shortcut_candidates(
                planning_graph,
                observed_states,
            )

        selected_candidates = select_random_shortcuts(
            shortcut_candidates,
            max_shortcuts_per_graph,
            rng,
        )
        print(f"Selected {len(selected_candidates)} shortcuts for training")

        for candidate in selected_candidates:
            source_id = candidate.source_node.id
            if source_id in observed_states and observed_states[source_id] is not None:
                for source_state in observed_states[source_id]:
                    training_states.append(source_state)
                    current_atoms_list.append(candidate.source_atoms)
                    goal_atoms_list.append(candidate.target_atoms)
                    # Store shortcut info (duplicate if needed)
                    shortcut_info.append(
                        {
                            "source_node_id": candidate.source_node.id,
                            "target_node_id": candidate.target_node.id,
                            "source_atoms_count": len(candidate.source_atoms),
                            "target_atoms_count": len(candidate.target_atoms),
                        }
                    )
                signature = ShortcutSignature.from_context(
                    candidate.source_atoms,
                    candidate.target_atoms,
                )
                if signature not in approach.trained_signatures:
                    approach.trained_signatures.append(signature)
            else:
                print(f"Warning: No states found for source node {source_id}")

    print(f"Collected {len(training_states)} examples from {collect_episodes} episodes")
    approach.training_mode = False

    return (
        TrainingData(
            states=training_states,
            current_atoms=current_atoms_list,
            goal_atoms=goal_atoms_list,
            config={
                **config,
                "shortcut_info": shortcut_info,
                "use_random_rollouts": use_random_rollouts,
                "num_rollouts_per_node": num_rollouts_per_node,
                "max_steps_per_rollout": max_steps_per_rollout,
                "shortcut_success_threshold": shortcut_success_threshold,
            },
        ),
        observed_states,
    )


def collect_goal_conditioned_training_data(
    system: ImprovisationalTAMPSystem,
    approach: ImprovisationalTAMPApproach,
    config: dict[str, Any],
    use_random_rollouts: bool = False,
    num_rollouts_per_node: int = 50,
    max_steps_per_rollout: int = 50,
    shortcut_success_threshold: int = 1,
    rng: np.random.Generator | None = None,
) -> GoalConditionedTrainingData:
    """Collect training data for goal-conditioned learning."""
    node_states: dict[int, Any] = {}
    valid_shortcuts: list[tuple[int, int]] = []
    node_atoms: dict[int, set[GroundAtom]] = {}
    train_data, node_states = collect_graph_based_training_data(
        system,
        approach,
        config,
        use_random_rollouts=use_random_rollouts,
        num_rollouts_per_node=num_rollouts_per_node,
        max_steps_per_rollout=max_steps_per_rollout,
        shortcut_success_threshold=shortcut_success_threshold,
        rng=rng,
    )
    shortcut_info = train_data.config.get("shortcut_info", [])
    planning_graph = approach.planning_graph
    assert planning_graph is not None
    for info in shortcut_info:
        source_id = info.get("source_node_id")
        target_id = info.get("target_node_id")
        assert source_id is not None and target_id is not None
        valid_shortcuts.append((source_id, target_id))
    for node in planning_graph.nodes:
        if node.id in node_states:
            node_atoms[node.id] = set(node.atoms)
    goal_train_data = GoalConditionedTrainingData(
        states=train_data.states,
        current_atoms=train_data.current_atoms,
        goal_atoms=train_data.goal_atoms,
        config={
            **train_data.config,
            "node_state_count": len(node_states),
            "valid_shortcut_count": len(valid_shortcuts),
        },
        node_states=node_states,
        valid_shortcuts=valid_shortcuts,
        node_atoms=node_atoms,
    )
    return goal_train_data


def identify_shortcut_candidates(
    planning_graph: PlanningGraph,
    observed_states: dict[int, list[ObsType]],
) -> list[ShortcutCandidate]:
    """Identify potential shortcuts in the planning graph."""
    nodes = list(planning_graph.nodes)
    shortcut_candidates = []

    for source_node in nodes:
        if source_node.id not in observed_states:
            continue

        # Use the first state for the candidate (we'll expand later)
        source_state = observed_states[source_node.id][0]

        for target_node in nodes:
            if source_node == target_node:
                continue
            if target_node.id <= source_node.id:
                continue

            has_direct_edge = False
            for edge in planning_graph.node_to_outgoing_edges.get(source_node, []):
                if edge.target == target_node and not edge.is_shortcut:
                    has_direct_edge = True
                    break

            if has_direct_edge:
                continue

            has_shortcut_edge = False
            for edge in planning_graph.node_to_outgoing_edges.get(source_node, []):
                if edge.target == target_node and edge.is_shortcut:
                    has_shortcut_edge = True
                    break

            if has_shortcut_edge:
                continue

            shortcut_candidates.append(
                ShortcutCandidate(
                    source_node=source_node,
                    target_node=target_node,
                    source_atoms=set(source_node.atoms),
                    target_atoms=set(target_node.atoms),
                    source_state=source_state,
                )
            )

    return shortcut_candidates


def identify_promising_shortcuts_with_rollouts(
    system,
    planning_graph,
    observed_states: dict[int, list[ObsType]],
    num_rollouts_per_node: int = 50,
    max_steps_per_rollout: int = 30,
    shortcut_success_threshold: int = 1,
    action_scale: float = 1.0,
    random_seed: int = 42,
) -> list[ShortcutCandidate]:
    """Identify promising shortcuts by performing random rollouts from each
    node."""
    shortcut_success_counts: defaultdict[tuple[int, int], int] = defaultdict(
        int
    )  # (source_node_id, target_node_id) -> count
    promising_candidates = []

    # Get the base environment for running rollouts
    raw_env = system.env
    sampling_space = gym.spaces.Box(
        low=raw_env.action_space.low * action_scale,
        high=raw_env.action_space.high * action_scale,
        dtype=raw_env.action_space.dtype,
    )
    sampling_space.seed(random_seed)

    for source_node_id, source_states in observed_states.items():
        source_node = next(
            (n for n in planning_graph.nodes if n.id == source_node_id), None
        )
        assert source_node is not None
        source_atoms = set(source_node.atoms)
        rollouts_per_state = max(1, num_rollouts_per_node // len(source_states))
        print(
            f"\nPerforming {rollouts_per_state} rollouts for each of {len(source_states)} state(s) from node {source_node_id}"  # pylint: disable=line-too-long
        )
        reached_nodes: defaultdict[int, int] = defaultdict(int)

        for _, source_state in enumerate(source_states):
            for rollout_idx in range(rollouts_per_state):
                if rollout_idx > 0 and rollout_idx % 100 == 0:
                    print(f"Completed {rollout_idx}/{rollouts_per_state} rollouts")

                raw_env.reset_from_state(source_state)
                curr_atoms = source_atoms.copy()

                reached_in_this_rollout: set[int] = set()
                for _ in range(max_steps_per_rollout):
                    action = sampling_space.sample()
                    obs, _, terminated, truncated, _ = raw_env.step(action)
                    curr_atoms = system.perceiver.step(obs)

                    for target_node in planning_graph.nodes:
                        if target_node.id <= source_node_id:
                            continue

                        has_direct_edge = False
                        for edge in planning_graph.node_to_outgoing_edges.get(
                            source_node, []
                        ):
                            if edge.target == target_node and not edge.is_shortcut:
                                has_direct_edge = True
                                break
                        if has_direct_edge:
                            continue

                        # NOTE: no need to stop this rollout when we reach a node
                        # since we want to explore all reachable nodes
                        if (
                            set(target_node.atoms) == curr_atoms
                            and target_node.id not in reached_in_this_rollout
                        ):
                            reached_nodes[target_node.id] += 1
                            shortcut_success_counts[
                                (source_node_id, target_node.id)
                            ] += 1
                            reached_in_this_rollout.add(target_node.id)

                    if terminated or truncated:
                        break

        if reached_nodes:
            print(f"  Nodes whose atoms are reached from node {source_node_id}:")
            for target_id, count in sorted(reached_nodes.items(), key=lambda x: -x[1]):
                print(f"    → Node {target_id}: {count}/{num_rollouts_per_node} times")
        else:
            print(f"  No nodes whose atoms are reached from node {source_node_id}")

    print("\nShortcuts reaching success threshold:")
    for (source_id, target_id), count in sorted(
        shortcut_success_counts.items(), key=lambda x: -x[1]
    ):
        if count >= shortcut_success_threshold:
            source_node = next(
                (n for n in planning_graph.nodes if n.id == source_id), None
            )
            target_node = next(
                (n for n in planning_graph.nodes if n.id == target_id), None
            )

            assert source_node is not None and target_node is not None
            print(f"  Node {source_id} → Node {target_id}: {count} successes")

            source_state = observed_states[source_id][0]
            candidate = ShortcutCandidate(
                source_node=source_node,
                target_node=target_node,
                source_atoms=set(source_node.atoms),
                target_atoms=set(target_node.atoms),
                source_state=source_state,
            )
            promising_candidates.append(candidate)

    print(f"\nFound {len(promising_candidates)} promising shortcut candidates")
    return promising_candidates


def select_random_shortcuts(
    candidates: list[ShortcutCandidate],
    max_shortcuts: int,
    rng: np.random.Generator | None = None,
) -> list[ShortcutCandidate]:
    """Select a random subset of shortcut candidates."""
    rng = rng or np.random.default_rng()
    if len(candidates) <= max_shortcuts:
        return candidates
    indices = np.arange(len(candidates))
    selected_indices = rng.choice(indices, size=max_shortcuts, replace=False)
    return [candidates[i] for i in selected_indices]
