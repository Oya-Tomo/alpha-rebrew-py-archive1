import torch
from model import PVNet
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

    net.load_state_dict(torch.load("checkpoint/model_19.pt")["model"])

    board = Board()

    turn = Stone.BLACK

    while True:
        print(board)

        if board.is_over():
            break

        actions = board.get_actions(turn)

        if actions == [65]:
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
            policy, value = net(board.to_tensor(turn).reshape(1, 2, 8, 8).to(device))
            policy = policy[0]
            while True:
                action = policy.argmax().item()

                if action in actions:
                    break
                else:
                    policy[action] = -1

        board.act(turn, action)
        turn = flip(turn)


if __name__ == "__main__":
    test_match(Stone.BLACK)
