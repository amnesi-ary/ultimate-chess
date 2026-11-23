from dataclasses import replace
from random import Random
import unittest

from ultimate_chess.game import DRAW, O, OPEN, X, State, initial_state, legal_actions, legal_mask, step


DRAW_BOARD = bytes((X, O, X, X, O, O, O, X, X))
NEAR_DRAW = DRAW_BOARD[:-1] + bytes((OPEN,))
WIN_X = bytes((X, X, X, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN))
NEAR_WIN = bytes((X, X, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN))


def position(boards, active, forced):
    templates = {OPEN: bytes(9), X: WIN_X, DRAW: DRAW_BOARD}
    cells = tuple(active.get(board, templates[status]) for board, status in enumerate(boards))
    return State(cells, bytes(boards), X, forced)


class GameTests(unittest.TestCase):
    def test_first_move_forces_center_then_routes(self):
        start = initial_state()
        self.assertEqual(legal_actions(start), list(range(36, 45)))
        with self.assertRaises(ValueError):
            step(start, 0)
        after = step(start, 38)
        self.assertEqual((after.forced_board, after.player), (2, O))
        self.assertEqual(legal_actions(after), list(range(18, 27)))
        self.assertEqual(start.cells[4][2], OPEN)
        self.assertEqual(after.cells[4][2], X)
        with self.assertRaises(ValueError):
            step(replace(after, forced_board=None), 38)

    def test_closed_target_allows_any_open_board(self):
        for result in (X, DRAW):
            with self.subTest(result=result):
                start = position((result,) + (OPEN,) * 8, {}, 4)
                after = step(start, 36)
                self.assertIsNone(after.forced_board)
                self.assertEqual({action // 9 for action in legal_actions(after)}, set(range(1, 9)))
                with self.assertRaises(ValueError):
                    step(after, 3)

    def test_move_that_wins_its_own_target_releases_next_player(self):
        after = step(position((OPEN,) * 9, {2: NEAR_WIN}, 2), 20)
        self.assertEqual(after.boards[2], X)
        self.assertIsNone(after.forced_board)

    def test_last_cell_win_takes_precedence_over_draw(self):
        local = bytes((X, O, O, O, X, X, X, O, OPEN))
        after = step(position((OPEN,) * 9, {4: local}, 4), 44)
        self.assertEqual(after.boards[4], X)

    def test_small_board_stays_open_when_all_lines_blocked_but_not_full(self):
        local = NEAR_DRAW[:7] + bytes((OPEN, OPEN))
        after = step(position((OPEN,) * 9, {4: local}, 4), 43)
        self.assertEqual(after.cells[4], NEAR_DRAW)
        self.assertEqual(after.boards[4], OPEN)
        self.assertIn(44, legal_actions(replace(after, forced_board=None)))

    def test_big_win_closes_game(self):
        start = position((X, X) + (OPEN,) * 7, {2: NEAR_WIN}, 2)
        after = step(start, 20)
        self.assertEqual(after.result, X)
        self.assertFalse(any(legal_mask(after)))
        with self.assertRaises(ValueError):
            step(after, 36)

    def test_early_draw_and_one_side_can_still_win(self):
        early = (OPEN, OPEN, OPEN, DRAW, DRAW, DRAW, DRAW, DRAW, OPEN)
        alive = (X, OPEN, X, DRAW, OPEN, DRAW, DRAW, DRAW, OPEN)
        for boards, closing, expected in ((early, 0, DRAW), (alive, 4, OPEN)):
            with self.subTest(expected=expected):
                after = step(position(boards, {closing: NEAR_DRAW}, closing), closing * 9 + 8)
                self.assertEqual(after.boards[closing], DRAW)
                self.assertEqual(after.result, expected)
                self.assertEqual(after.cells[8], bytes(9))
                self.assertEqual(bool(legal_actions(after)), expected == OPEN)

    def test_random_games_terminate(self):
        rng = Random(42)
        for _ in range(100):
            state = initial_state()
            for _ in range(81):
                choices = legal_actions(state)
                if state.result != OPEN:
                    break
                self.assertEqual(choices, [i for i, legal in enumerate(legal_mask(state)) if legal])
                state = step(state, rng.choice(choices))
            self.assertNotEqual(state.result, OPEN)
            self.assertFalse(legal_actions(state))


if __name__ == "__main__":
    unittest.main()
