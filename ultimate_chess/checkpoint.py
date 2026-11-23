from pathlib import Path


import torch

CHECKPOINT_VERSION = 1
DEFAULT_CHECKPOINT = Path("runs/latest.pt")


def read_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version")
    return checkpoint
