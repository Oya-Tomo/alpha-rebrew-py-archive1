import random
import torch
import numpy as np
from model import PVNet
from mcts import MCT
from bitboard import Board, Stone, pos_to_idx, flip


def input_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Invalid input. Please enter an integer.")


def test_match(stone: Stone):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PVNet().to(device)

    net.load_state_dict(torch.load("checkpoint/model_129.pt")["model"])

    mct = MCT(net, 0.01)
    board = Board()

    turn = Stone.BLACK

    while True:
        print(board)

        if board.is_over():
            break

        actions = board.get_actions(turn)

        if actions == [64]:
            print("No valid moves. Passing.")
            turn = flip(turn)
            continue

        if turn == stone:
            while True:
                x = input_int("Enter x: ")
                y = input_int("Enter y: ")

                action = pos_to_idx(x, y)

                if action in actions:
                    break
                else:
                    print("Invalid move. Please try again.")
        else:
            policy = mct.search(board, turn, 2000)
            policy = np.array(policy, dtype=np.float32)

            _, value = net(board.to_tensor(turn).reshape(1, 2, 8, 8).to(device))
            # _ = policy.cpu().detach().numpy().reshape(65)

            while True:
                # action = int(random.choice(np.where(policy == policy.max())[0]))
                action = int(np.argmax(policy))

                if action in actions:
                    break
                else:
                    policy[action] = -1

            print(f"Value: {value.item()}")

        board.act(turn, action)
        turn = flip(turn)

    b, w, e = board.get_count()
    print(f"Result: Black - {b}, White - {w}, Empty - {e}")


if __name__ == "__main__":
    test_match(Stone.BLACK)
