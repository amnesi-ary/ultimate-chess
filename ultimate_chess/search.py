from dataclasses import dataclass, field
from math import sqrt
from typing import Protocol

import numpy as np

from .game import OPEN, State, legal_actions, outcome, step


class Evaluator(Protocol):
    def __call__(self, states: list[State]) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(slots=True)
class Node:
    state: State
    prior: float = 0.0
    children: dict[int, 'Node'] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def expand(self, priors: np.ndarray) -> None:
        self.children = {
            action: Node(step(self.state, action), float(priors[action]))
            for action in legal_actions(self.state)
        }

    def select(self) -> 'Node':
        exploration = 1.5 * sqrt(self.visits + 1)
        # A child's mean is for the player who moves there, our opponent.
        return max(self.children.values(), key=lambda child: (
            -child.mean + exploration * child.prior / (child.visits + 1)
        ))


def search(states: list[State], evaluator: Evaluator, simulations: int,
           noise_rng: np.random.Generator | None = None) -> np.ndarray:
    """Run independent trees, batching one pending evaluation per tree."""
    if not states or any(state.result != OPEN for state in states) or simulations < 1:
        raise ValueError("Search requires active states and a positive simulation count")
    roots = [Node(state) for state in states]
    priors, _ = evaluator(states)
    for root, row in zip(roots, priors, strict=True):
        root.expand(row)
        if noise_rng is not None:
            noise = noise_rng.dirichlet(np.full(len(root.children), 0.3))
            for child, sample in zip(root.children.values(), noise, strict=True):
                child.prior = 0.75 * child.prior + 0.25 * float(sample)

    for _ in range(simulations):
        pending = []
        for root in roots:
            path = [root]
            while path[-1].children:
                path.append(path[-1].select())
            leaf = path[-1]
            if leaf.state.result == OPEN:
                pending.append(path)
            else:
                backup(path, outcome(leaf.state.result, leaf.state.player))
        if pending:
            priors, scores = evaluator([path[-1].state for path in pending])
            for path, row, score in zip(pending, priors, scores, strict=True):
                path[-1].expand(row)
                backup(path, float(score))

    policies = np.zeros((len(roots), 81), dtype=np.float32)
    for row, root in zip(policies, roots, strict=True):
        for action, child in root.children.items():
            row[action] = child.visits / simulations
    return policies


def backup(path: list[Node], value: float) -> None:
    for node in reversed(path):
        node.visits += 1
        node.value_sum += value
        value = -value
