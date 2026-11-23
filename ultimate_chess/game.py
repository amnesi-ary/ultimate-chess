from dataclasses import dataclass
from operator import index

OPEN, X, O, DRAW = range(4)
LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)
LINES_AT = tuple(tuple(line for line in LINES if pos in line) for pos in range(9))


@dataclass(frozen=True, slots=True)
class State:
    cells: tuple[bytes, ...]
    boards: bytes
    player: int
    forced_board: int | None
    result: int = OPEN


def initial_state() -> State:
    return State((bytes(9),) * 9, bytes(9), X, 4)


def legal_actions(state: State) -> list[int]:
    if state.result != OPEN:
        return []
    boards = range(9) if state.forced_board is None else (state.forced_board,)
    return [
        board * 9 + cell
        for board in boards if state.boards[board] == OPEN
        for cell, mark in enumerate(state.cells[board]) if mark == OPEN
    ]


def legal_mask(state: State) -> tuple[bool, ...]:
    mask = [False] * 81
    for action in legal_actions(state):
        mask[action] = True
    return tuple(mask)


def _can_win(boards: bytes, player: int) -> bool:
    return any(all(boards[board] in (OPEN, player) for board in line) for line in LINES)


def step(state: State, action: int) -> State:
    action = index(action)
    if not 0 <= action < 81:
        raise ValueError(f"Action out of range: {action}")
    board, cell = divmod(action, 9)
    if (state.result != OPEN or state.boards[board] != OPEN
            or state.forced_board not in (None, board) or state.cells[board][cell]):
        raise ValueError(f"Illegal action: {action}")

    player = state.player
    local = state.cells[board]
    local = local[:cell] + bytes((player,)) + local[cell + 1:]
    cells = state.cells[:board] + (local,) + state.cells[board + 1:]
    boards = state.boards
    result = OPEN
    local_result = OPEN
    if any(all(local[pos] == player for pos in line) for line in LINES_AT[cell]):
        local_result = player
    elif OPEN not in local:
        local_result = DRAW

    if local_result != OPEN:
        boards = boards[:board] + bytes((local_result,)) + boards[board + 1:]
        if local_result == player and any(
            all(boards[pos] == player for pos in line) for line in LINES_AT[board]
        ):
            result = player
        elif not (_can_win(boards, X) or _can_win(boards, O)):
            result = DRAW

    target = cell if result == OPEN and boards[cell] == OPEN else None
    return State(cells, boards, 3 - player, target, result)


def outcome(result: int, player: int) -> int:
    if result == OPEN:
        raise ValueError("The game is still in progress")
    if result == DRAW:
        return 0
    return 1 if result == player else -1
