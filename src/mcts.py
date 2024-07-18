import random
import copy
import math
import numpy as np
import torch
from torch import Tensor
from model import PVNet
from bitboard import Board, Stone, ACTION_COUNT, flip


def count_to_score(b: int, w: int) -> float:
    return (b - w) / (b + w)


class MCT:
    def __init__(
        self,
        net: PVNet,
        alpha: float,
        c_puct: float = 1.0,
        eps: float = 0.25,
    ) -> None:

        self.net = net
        self.device = next(net.parameters()).device

        self.alpha = alpha
        self.c_puct = c_puct
        self.eps = eps

        self.P = {}
        self.N = {}
        self.W = {}

        self.transition_cache: dict[int, list[int]] = {}

    def search(self, board: Board, stone: Stone, simul: int):
        s = board.to_key(stone)

        if s not in self.P:
            _ = self.expand(board, stone)

        actions = board.get_actions(stone)
        noise = np.random.dirichlet([self.alpha] * len(actions))

        for a, noise in zip(actions, noise):
            self.P[s][a] = (1 - self.eps) * self.P[s][a] + self.eps * noise

        for _ in range(simul):
            U = [
                self.c_puct
                * self.P[s][a]
                * math.sqrt(sum(self.N[s]))
                / (1 + self.N[s][a])
                for a in range(ACTION_COUNT)
            ]

            Q = [w / n if n != 0 else w for w, n in zip(self.W[s], self.N[s])]

            scores = [u + q for u, q in zip(U, Q)]
            scores = torch.tensor(
                [score if a in actions else -1e8 for a, score in enumerate(scores)]
            )

            action = random.choice(torch.where(scores == scores.max())[0])
            next_board = self.transition_cache[s][action]

            value = -self.evaluate(next_board, flip(stone))

            self.W[s][action] += value
            self.N[s][action] += 1

        mcts_policy = [n / sum(self.N[s]) for n in self.N[s]]

        return mcts_policy

    def expand(self, board: Board, stone: Stone) -> float:
        s = board.to_key(stone)

        with torch.no_grad():
            st = board.to_tensor(stone).to(self.device)
            output = self.net(st.reshape(1, 2, 8, 8))
            policy: Tensor = output[0][0]
            value: Tensor = output[1][0][0]

        self.P[s] = policy.tolist()
        self.N[s] = [0] * ACTION_COUNT
        self.W[s] = [0] * ACTION_COUNT

        actions = board.get_actions(stone)

        self.transition_cache[s] = [
            copy.deepcopy(board).act(stone, a) if (a in actions) else None
            for a in range(ACTION_COUNT)
        ]

        return value

    def evaluate(self, board: Board, stone: Stone) -> float:
        s = board.to_key(stone)

        if board.is_over():
            b, w, e = board.get_count()
            score = count_to_score(b, w)
            if stone == Stone.BLACK:
                return score
            else:
                return -score

        elif s not in self.P:
            value = self.expand(board, stone)
            return value

        else:
            U = [
                self.c_puct
                * self.P[s][a]
                * math.sqrt(sum(self.N[s]))
                / (1 + self.N[s][a])
                for a in range(ACTION_COUNT)
            ]

            Q = [q / n if n != 0 else q for q, n in zip(self.W[s], self.N[s])]

            actions = board.get_actions(stone)
            scores = [u + q for u, q in zip(U, Q)]
            scores = torch.tensor(
                [score if a in actions else -1e8 for a, score in enumerate(scores)]
            )

            action = random.choice(torch.where(scores == scores.max())[0])

            next_board = self.transition_cache[s][action]

            value = -self.evaluate(next_board, flip(stone))

            self.W[s][action] += value
            self.N[s][action] += 1

            return value
