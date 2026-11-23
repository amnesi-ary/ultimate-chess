import argparse
from pathlib import Path

import torch

from .checkpoint import DEFAULT_CHECKPOINT
from .game import DRAW, O, OPEN, X, State, initial_state, step
from .player import ModelPlayer

MARKS = {OPEN: ".", X: "X", O: "O", DRAW: "="}
SEARCH_SIMULATIONS = 64


def render(state: State) -> str:
    rows = []
    for outer in range(0, 9, 3):
        rows.append(" | ".join(f" {board + 1}[{MARKS[state.boards[board]]}] "
                               for board in range(outer, outer + 3)))
        for inner in range(0, 9, 3):
            rows.append(" | ".join(" ".join(MARKS[mark] for mark in state.cells[board][inner:inner + 3])
                                   for board in range(outer, outer + 3)))
        if outer < 6:
            rows.append("------+-------+------")
    return "\n".join(rows)



def play(ai: ModelPlayer, human: int) -> None:
    state = initial_state()
    while state.result == OPEN:
        print(render(state))
        if state.player != human:
            action = ai.choose(state, SEARCH_SIMULATIONS)
            board, cell = divmod(action, 9)
            print(f"AI {MARKS[state.player]} played board {board + 1}, cell {cell + 1}.")
            state = step(state, action)
            continue
        target = "any open board" if state.forced_board is None else f"board {state.forced_board + 1}"
        try:
            text = input(f"Your move ({target}), enter board (1-9) cell (1-9) or q: ").strip()
        except EOFError:
            return
        if text.lower() == "q":
            return
        try:
            board, cell = map(int, text.split())
            if not (1 <= board <= 9 and 1 <= cell <= 9):
                raise ValueError
            state = step(state, (board - 1) * 9 + cell - 1)
        except ValueError:
            print("Illegal move. Check the allowed board and choose an empty cell.")
    print(render(state))
    print("Draw." if state.result == DRAW else f"{MARKS[state.result]} wins.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play nine-board tic-tac-toe against a trained model")
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_CHECKPOINT,
                        help=f"model checkpoint (default: {DEFAULT_CHECKPOINT})")
    args = parser.parse_args()
    try:
        torch.set_num_threads(2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ai = ModelPlayer.load(args.model, device)
        while True:
            side = input("Choose X (first), O (second), or Q (quit) [X]: ").strip().upper()
            if side == "Q":
                return
            if side in ("", "X", "O"):
                play(ai, O if side == "O" else X)
                return
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
