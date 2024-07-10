import torch
import numpy as np
import random
import copy
import ray
from dataclasses import dataclass

from model import PVNet
from bitboard import Board, Stone, ACTION_COUNT, flip
from mcts import MCT


@dataclass
class MCTStep:
    board: Board
    stone: Stone
    policy: np.ndarray
    value: float


@ray.remote(num_cpus=1, num_gpus=0)
def auto_match(state_dict, simul: int, alpha: float = 0.35):
    board = Board()
    net = PVNet()
    net.load_state_dict(state_dict)

    mct = MCT(net, alpha)
    stone = Stone.BLACK
    turn = 0

    record: list[MCTStep] = []

    while True:
        if board.is_over():
            break

        policy = mct.search(board, stone, simul)
        policy = np.array(policy, dtype=np.float32)

        if turn < 10:
            action = np.random.choice(range(ACTION_COUNT), p=policy)
        else:
            action = random.choice(np.where(policy == policy.max())[0])

        record.append(MCTStep(copy.deepcopy(board), stone, policy, 0))
        board.act(stone, int(action))

        stone = flip(stone)
        turn += 1

    b, w, e = board.get_count()
    for i in range(len(record)):
        if b == w:
            record[i].value = 0.0
        elif b > w and record[i].stone == Stone.BLACK:
            record[i].value = 1.0
        elif b < w and record[i].stone == Stone.WHITE:
            record[i].value = 1.0
        else:
            record[i].value = -1.0

    return record


if __name__ == "__main__":
    net = PVNet()

    rec = auto_match(net.state_dict(), 40, alpha=0.2)
    print(rec)
