import argparse
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from .checkpoint import CHECKPOINT_VERSION, DEFAULT_CHECKPOINT, read_checkpoint
from .game import DRAW, O, OPEN, X, State, initial_state, outcome, step
from .network import ChessNet, NetworkEvaluator, encode
from .search import Evaluator, search


# Edit these values before running python -m ultimate_chess.training.
CHECKPOINT = DEFAULT_CHECKPOINT
ITERATIONS = 10  # Additional iterations per invocation.
GAMES_PER_ITERATION = 64
PARALLEL_GAMES = 16
SEARCH_SIMULATIONS = 64
BATCH_SIZE = 128
UPDATES_PER_ITERATION = 32
REPLAY_SIZE = 100_000
LEARNING_RATE = 1e-3
SAMPLE_MOVES = 12
SEED = 42


def _symmetries():
    grid = np.arange(9).reshape(3, 3)
    # Each flat permutation takes an index in the transformed grid to the original.
    return tuple(np.rot90(base, turns).copy().ravel()
                 for base in (grid, np.fliplr(grid)) for turns in range(4))


SYMMETRIES = _symmetries()


def transform(state: State, policy: np.ndarray, permutation: np.ndarray):
    inverse = permutation.argsort()
    cells = tuple(bytes(state.cells[board][cell] for cell in permutation) for board in permutation)
    boards = bytes(state.boards[board] for board in permutation)
    target = None if state.forced_board is None else int(inverse[state.forced_board])
    policy = policy.reshape(9, 9)[np.ix_(permutation, permutation)].ravel()
    return State(cells, boards, state.player, target, state.result), policy


@dataclass(slots=True)
class Game:
    state: State = field(default_factory=initial_state)
    history: list[tuple[State, np.ndarray]] = field(default_factory=list)


Sample = tuple[State, np.ndarray, int]


def self_play(game_count: int, parallel: int, simulations: int,
              evaluator: Evaluator, rng: np.random.Generator):
    active = [Game() for _ in range(min(game_count, parallel))]
    remaining = game_count - len(active)
    while active:
        policies = search([game.state for game in active], evaluator, simulations, rng)
        next_games = []
        for game, policy in zip(active, policies, strict=True):
            state = game.state
            if len(game.history) < SAMPLE_MOVES:
                probabilities = policy.astype(np.float64)
                action = rng.choice(81, p=probabilities / probabilities.sum())
            else:
                action = policy.argmax()
            game.history.append((state, policy.copy()))
            game.state = step(state, action)
            if game.state.result == OPEN:
                next_games.append(game)
            else:
                yield game.state.result, [
                    (position, target, 1 - outcome(game.state.result, position.player))
                    for position, target in game.history
                ]
                if remaining:
                    next_games.append(Game())
                    remaining -= 1
        active = next_games


class Replay:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.samples: list[Sample] = []
        self.cursor = 0

    def extend(self, samples: list[Sample]) -> None:
        for sample in samples:
            if len(self.samples) < self.capacity:
                self.samples.append(sample)
            else:
                self.samples[self.cursor] = sample
            self.cursor = (self.cursor + 1) % self.capacity

    def pack(self) -> dict:
        states = bytearray()
        for state, _, _ in self.samples:
            states.extend(b"".join(state.cells) + state.boards)
            target = 9 if state.forced_board is None else state.forced_board
            states.extend((state.player, target))
        return {
            "capacity": self.capacity,
            "cursor": self.cursor,
            "states": torch.from_numpy(np.frombuffer(states, dtype=np.uint8).reshape(-1, 92)),
            "policies": torch.from_numpy(np.array([sample[1] for sample in self.samples],
                                                 dtype=np.float32).reshape(-1, 81)),
            "labels": torch.tensor([sample[2] for sample in self.samples], dtype=torch.uint8),
        }

    @classmethod
    def unpack(cls, data: dict, capacity: int) -> 'Replay':
        replay = cls(capacity)
        states, policies, labels = (data[key].numpy() for key in ("states", "policies", "labels"))
        if capacity == data["capacity"]:
            order = range(len(states))
        else:
            # Preserve chronological order when shrinking or growing a full ring.
            start = data["cursor"] if len(states) == data["capacity"] else 0
            order = ((start + index) % len(states)
                     for index in range(max(0, len(states) - capacity), len(states)))
        for index in order:
            row = states[index].tobytes()
            player, target = row[90:92]
            cells = tuple(row[start:start + 9] for start in range(0, 81, 9))
            state = State(cells, row[81:90], player, None if target == 9 else target)
            replay.samples.append((state, policies[index].copy(), int(labels[index])))
        replay.cursor = data["cursor"] if capacity == data["capacity"] else len(replay.samples) % capacity
        return replay


def update(model, optimizer, replay, batch_size, rng, device) -> tuple[float, float]:
    indices = rng.integers(len(replay.samples), size=batch_size)
    augmentations = rng.integers(len(SYMMETRIES), size=batch_size)
    states, policies, labels = [], [], []
    for index, augmentation in zip(indices, augmentations, strict=True):
        state, policy, label = replay.samples[index]
        state, policy = transform(state, policy, SYMMETRIES[augmentation])
        states.append(state)
        policies.append(policy)
        labels.append(label)
    local, metadata, mask = (torch.from_numpy(array).to(device) for array in encode(states))
    targets = torch.from_numpy(np.stack(policies)).to(device)
    labels = torch.tensor(labels, device=device)
    model.train()
    policy, value = model(local, metadata)
    log_probs = policy.masked_fill(~mask, float("-inf")).log_softmax(dim=1)
    policy_loss = -(targets * log_probs.masked_fill(~mask, 0.0)).sum(dim=1).mean()
    value_loss = F.cross_entropy(value, labels)
    optimizer.zero_grad(set_to_none=True)
    (policy_loss + value_loss).backward()
    optimizer.step()
    return policy_loss.item(), value_loss.item()


def save_checkpoint(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(data, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train() -> None:
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'}", flush=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = ChessNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    replay = Replay(REPLAY_SIZE)
    completed = 0
    if CHECKPOINT.exists():
        checkpoint = read_checkpoint(CHECKPOINT)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE
        replay = Replay.unpack(checkpoint["replay"], REPLAY_SIZE)
        rng.bit_generator.state = checkpoint["rng"]
        completed = checkpoint["iteration"]
        print(f"Resuming after iteration {completed} from {CHECKPOINT}", flush=True)

    evaluator = NetworkEvaluator(model, device)
    for iteration in range(completed + 1, completed + ITERATIONS + 1):
        started = perf_counter()
        counts = {X: 0, O: 0, DRAW: 0}
        for result, samples in self_play(GAMES_PER_ITERATION, PARALLEL_GAMES, SEARCH_SIMULATIONS,
                                         evaluator, rng):
            counts[result] += 1
            replay.extend(samples)
        losses = [update(model, optimizer, replay, BATCH_SIZE, rng, device)
                  for _ in range(UPDATES_PER_ITERATION)]
        policy_loss, value_loss = np.mean(losses, axis=0)
        save_checkpoint(CHECKPOINT, {
            "version": CHECKPOINT_VERSION, "iteration": iteration, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "replay": replay.pack(),
            "rng": rng.bit_generator.state,
        })
        print(f"Iteration {iteration} | X {counts[X]} O {counts[O]} Draw {counts[DRAW]} | "
              f"replay {len(replay.samples)} | policy {policy_loss:.3f} value {value_loss:.3f} | "
              f"{perf_counter() - started:.1f}s | {CHECKPOINT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model by self-play")
    parser.parse_args()

    try:
        train()
    except (ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "\nInterrupted. Completed training iterations remain saved.\n")


if __name__ == "__main__":
    main()
