from random import Random
import unittest

import numpy as np
import torch

from ultimate_chess.game import OPEN, X, State, initial_state, legal_actions, legal_mask, step
from ultimate_chess.network import ChessNet, NetworkEvaluator
from ultimate_chess.search import search
from ultimate_chess.training import Replay, SYMMETRIES, transform


def neutral(states):
    priors = np.zeros((len(states), 81), dtype=np.float32)
    for row, state in zip(priors, states, strict=True):
        actions = legal_actions(state)
        row[actions] = 1 / len(actions)
    return priors, np.zeros(len(states), dtype=np.float32)


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_immediate_win_and_batched_network_masks(self):
        win = bytes((X, X, X, 0, 0, 0, 0, 0, 0))
        threat = bytes((X, X, 0, 0, 0, 0, 0, 0, 0))
        state = State((win, win, threat) + (bytes(9),) * 6,
                      bytes((X, X) + (OPEN,) * 7), X, 2)
        self.assertEqual(step(state, 20).result, X)
        policies = search([initial_state(), state], neutral, 24)
        self.assertEqual(policies[1].argmax(), 20)
        self.assertTrue(np.allclose(policies.sum(axis=1), 1))

        evaluator = NetworkEvaluator(ChessNet(), torch.device("cpu"))
        priors, values = evaluator([initial_state(), state])
        masks = np.asarray([legal_mask(initial_state()), legal_mask(state)])
        self.assertTrue((priors[~masks] == 0).all())
        self.assertTrue(np.allclose(priors.sum(axis=1), 1))
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(values.shape, (2,))

    def test_all_symmetries_commute_with_moves_and_relabel_policy(self):
        rng = Random(4)
        state = initial_state()
        self.assertEqual(len({tuple(permutation) for permutation in SYMMETRIES}), 8)
        while state.result == OPEN:
            action = rng.choice(legal_actions(state))
            policy = np.zeros(81, dtype=np.float32)
            policy[action] = 1
            after = step(state, action)
            for permutation in SYMMETRIES:
                transformed, target = transform(state, policy, permutation)
                expected, _ = transform(after, policy, permutation)
                self.assertEqual(step(transformed, target.argmax()), expected)
            state = after

    def test_replay_roundtrip_and_resize_keep_newest(self):
        replay = Replay(3)
        state = initial_state()
        policy = neutral([state])[0][0]
        samples = []
        for offset in range(4):
            next_state = step(state, 36 + offset)
            samples.append((next_state, neutral([next_state])[0][0], offset % 3))
        replay.extend(samples)
        for capacity in (2, 3, 4):
            restored = Replay.unpack(replay.pack(), capacity)
            start = restored.cursor if len(restored.samples) == capacity else 0
            ordered = restored.samples[start:] + restored.samples[:start]
            expected = samples[-min(3, capacity):]
            self.assertEqual([sample[0] for sample in ordered], [sample[0] for sample in expected])
            self.assertTrue(all(np.array_equal(a[1], b[1]) and a[2] == b[2]
                                for a, b in zip(ordered, expected, strict=True)))
        self.assertTrue(np.isclose(policy.sum(), 1))
