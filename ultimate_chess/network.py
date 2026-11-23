import numpy as np
import torch
from torch import nn

from .game import DRAW, OPEN, State, legal_mask


def encode(states: list[State]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build local planes, board metadata, and flat legal-action masks."""
    cells = np.frombuffer(b"".join(board for state in states for board in state.cells),
                          dtype=np.uint8).reshape(-1, 9, 9)
    boards = np.frombuffer(b"".join(state.boards for state in states),
                           dtype=np.uint8).reshape(-1, 9)
    players = np.array([state.player for state in states], dtype=np.uint8)
    masks = np.asarray([legal_mask(state) for state in states], dtype=bool)
    legal = masks.reshape(-1, 9, 9)
    local = np.stack((cells == players[:, None, None],
                      cells == (3 - players)[:, None, None], legal), axis=2)
    metadata = np.stack((boards == OPEN, boards == players[:, None],
                         boards == (3 - players)[:, None], boards == DRAW,
                         legal.any(axis=2)), axis=2)
    return local.reshape(-1, 9, 3, 3, 3).astype(np.float32), metadata.astype(np.float32), masks


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return torch.relu(x + self.layers(x))


class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), 
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            ResidualBlock(32), 
            ResidualBlock(32),
        )
        self.project = nn.Sequential(
            nn.Linear(32 * 9 + 5, 64), 
            nn.ReLU()
        )
        self.global_board = nn.Sequential(
            ResidualBlock(64), 
            ResidualBlock(64)
        )
        self.policy = nn.Sequential(
            nn.Linear(32 + 64 + 64, 64), 
            nn.ReLU(), 
            nn.Linear(64, 1)
        )
        self.value = nn.Sequential(
            nn.Linear(64 * 9, 128), 
            nn.ReLU(), 
            nn.Linear(128, 3)
        )

    def forward(self, local, metadata):
        batch_size = local.shape[0]
        features = self.local(local.reshape(-1, 3, 3, 3)).reshape(batch_size, 9, 32, 9)
        board_vectors = self.project(torch.cat((features.flatten(2), metadata), dim=-1))
        context = self.global_board(board_vectors.transpose(1, 2).reshape(batch_size, 64, 3, 3))
        by_board = context.flatten(2).transpose(1, 2)
        source = by_board[:, :, None, :].expand(-1, -1, 9, -1)
        target = by_board[:, None, :, :].expand(-1, 9, -1, -1)
        moves = torch.cat((features.transpose(2, 3), source, target), dim=-1)
        return self.policy(moves).reshape(batch_size, 81), self.value(context.flatten(1))


class NetworkEvaluator:
    def __init__(self, model: ChessNet, device: torch.device):
        self.model = model
        self.device = device

    @torch.inference_mode()
    def __call__(self, states: list[State]) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        local, metadata, mask = (torch.from_numpy(array).to(self.device) for array in encode(states))
        policy, value = self.model(local, metadata)
        priors = policy.masked_fill(~mask, float("-inf")).softmax(dim=1)
        probabilities = value.softmax(dim=1)
        return priors.cpu().numpy(), (probabilities[:, 0] - probabilities[:, 2]).cpu().numpy()
