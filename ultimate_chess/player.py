from pathlib import Path
from pickle import UnpicklingError

import torch

from .checkpoint import read_checkpoint
from .game import State
from .network import ChessNet, NetworkEvaluator
from .search import search


class ModelPlayer:
    def __init__(self, evaluator: NetworkEvaluator):
        self.evaluator = evaluator

    @classmethod
    def load(cls, path: Path, device: torch.device) -> 'ModelPlayer':
        try:
            checkpoint = read_checkpoint(path)
            model = ChessNet()
            model.load_state_dict(checkpoint["model"])
        except (KeyError, RuntimeError, UnpicklingError, EOFError, TypeError) as error:
            raise ValueError(f"Invalid model checkpoint: {path}") from error
        return cls(NetworkEvaluator(model.to(device), device))

    def choose(self, state: State, simulations: int) -> int:
        return int(search([state], self.evaluator, simulations)[0].argmax())
