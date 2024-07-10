import copy
import torch
from enum import IntEnum

ACTION_COUNT = 65


class Stone(IntEnum):
    BLACK = 1
    WHITE = -1
    EMPTY = 0


def flip(stone: Stone) -> Stone:
    return Stone(stone * -1)


def bit_shift(target: int, shift: int) -> int:
    if shift >= 0:
        return target << shift
    else:
        return target >> -shift


def pos_to_idx(x: int, y: int) -> int:
    return y * 8 + x


def idx_to_pos(idx: int) -> tuple[int, int]:
    return idx % 8, idx // 8


class Board:
    V_TRIM_MASK = 0x00FFFFFFFFFFFF00  # vertical side
    H_TRIM_MASK = 0x7E7E7E7E7E7E7E7E  # horizontal side
    A_TRIM_MASK = 0x007E7E7E7E7E7E00  # all side

    shifts = [9, 8, 7, 1, -1, -7, -8, -9]

    def __init__(self) -> None:
        self.black_board = 0
        self.white_board = 0

        self.black_board |= 1 << pos_to_idx(4, 3)
        self.black_board |= 1 << pos_to_idx(3, 4)
        self.white_board |= 1 << pos_to_idx(3, 3)
        self.white_board |= 1 << pos_to_idx(4, 4)

        self.black_actions = 0
        self.white_actions = 0

        self._update_action_board()

    def __str__(self) -> str:
        char = {Stone.BLACK: "○", Stone.WHITE: "●", Stone.EMPTY: "*"}
        s = ""
        for y in range(8):
            for x in range(8):
                idx = pos_to_idx(x, y)
                if self.black_board & (1 << idx):
                    s += char[Stone.BLACK]
                elif self.white_board & (1 << idx):
                    s += char[Stone.WHITE]
                else:
                    s += char[Stone.EMPTY]
                s += " "
            s += "\n"
        return s

    def get_actions(self, stone: Stone) -> int:
        actor_actions = (
            self.black_actions if stone == Stone.BLACK else self.white_actions
        )
        actions = []
        for i in range(64):
            if actor_actions & (1 << i):
                actions.append(i)
        if len(actions) == 0:
            actions.append(64)  # passd
        return actions

    def act(self, stone: Stone, action: int) -> "Board":
        if action == 64:
            return self

        if stone == Stone.BLACK:
            self.black_board, self.white_board = self._put(
                action, self.black_board, self.white_board
            )
        else:
            self.white_board, self.black_board = self._put(
                action, self.white_board, self.black_board
            )

        self._update_action_board()
        return self

    def is_over(self) -> bool:
        return self.black_actions == 0 and self.white_actions == 0

    def get_count(self):
        b = self.black_board.bit_count()
        w = self.white_board.bit_count()
        e = 64 - b - w
        return b, w, e

    @classmethod
    def _put(cls, pos: int, actor_board: int, oppnt_board: int) -> tuple[int, int]:
        action_board = 1 << pos
        flip_board = 0
        for shift in cls.shifts:
            flip_line = 0
            for i in range(1, 8):
                mask = bit_shift(action_board, shift * i) & 0xFFFFFFFFFFFFFFFF
                continue_flag = mask & cls._get_shift_mask(shift)
                if mask == 0:
                    break

                if oppnt_board & mask:
                    flip_line |= mask
                elif actor_board & mask:
                    flip_board |= flip_line
                    break
                else:
                    break

                if continue_flag == 0:
                    break
        actor_board ^= flip_board | action_board
        oppnt_board ^= flip_board

        return actor_board, oppnt_board

    def _update_action_board(self):
        self.black_actions = self._get_legal_board(self.black_board, self.white_board)
        self.white_actions = self._get_legal_board(self.white_board, self.black_board)

    @classmethod
    def _get_shift_mask(cls, shift: int) -> int:
        if abs(shift) == 1:
            return cls.H_TRIM_MASK
        elif abs(shift) == 8:
            return cls.V_TRIM_MASK
        else:
            return cls.A_TRIM_MASK

    @classmethod
    def _get_legal_board(cls, actor_board: int, oppnt_board: int) -> int:
        v_trim_board = oppnt_board & cls.V_TRIM_MASK
        h_trim_board = oppnt_board & cls.H_TRIM_MASK
        a_trim_board = oppnt_board & cls.A_TRIM_MASK
        blank_board = ~(actor_board | oppnt_board)

        legal_board = 0

        for shift in cls.shifts:
            if abs(shift) == 1:
                tmp = h_trim_board & bit_shift(actor_board, shift)
                tmp |= h_trim_board & bit_shift(tmp, shift)
                tmp |= h_trim_board & bit_shift(tmp, shift)
                tmp |= h_trim_board & bit_shift(tmp, shift)
                tmp |= h_trim_board & bit_shift(tmp, shift)
                tmp |= h_trim_board & bit_shift(tmp, shift)
                legal_board |= blank_board & bit_shift(tmp, shift)
            elif abs(shift) == 8:
                tmp = v_trim_board & bit_shift(actor_board, shift)
                tmp |= v_trim_board & bit_shift(tmp, shift)
                tmp |= v_trim_board & bit_shift(tmp, shift)
                tmp |= v_trim_board & bit_shift(tmp, shift)
                tmp |= v_trim_board & bit_shift(tmp, shift)
                tmp |= v_trim_board & bit_shift(tmp, shift)
                legal_board |= blank_board & bit_shift(tmp, shift)
            else:
                tmp = a_trim_board & bit_shift(actor_board, shift)
                tmp |= a_trim_board & bit_shift(tmp, shift)
                tmp |= a_trim_board & bit_shift(tmp, shift)
                tmp |= a_trim_board & bit_shift(tmp, shift)
                tmp |= a_trim_board & bit_shift(tmp, shift)
                tmp |= a_trim_board & bit_shift(tmp, shift)
                legal_board |= blank_board & bit_shift(tmp, shift)
        return legal_board

    def to_tensor(self, stone: Stone) -> torch.Tensor:
        t = torch.zeros(size=(2, 8, 8))
        bc = 0 if stone == Stone.BLACK else 1
        wc = 1 if stone == Stone.BLACK else 0
        for i in range(64):
            mask = 1 << i
            x, y = idx_to_pos(i)
            if self.black_board & mask:
                t[bc][y][x] = 1.0
            elif self.white_board & mask:
                t[wc][y][x] = 1.0
        return t

    def to_key(self, stone: Stone) -> int:
        a = self.black_board if stone == Stone.BLACK else self.white_board
        o = self.white_board if stone == Stone.BLACK else self.black_board
        key = (a << 64) + o
        return key


def print_legal_board(board: int):
    for r in range(0, 64, 8):
        for i in range(r, r + 8):
            if board & (1 << i):
                print("\x1b[33m*\x1b[0m ", end="")
            else:
                print("* ", end="")
        print()


if __name__ == "__main__":
    board = Board()
    print(board)
    print(board.get_actions(Stone.BLACK))
    board.act(Stone.BLACK, 26)
    print(board)
    print(board.get_actions(Stone.WHITE))
    board.act(Stone.WHITE, 18)
    print(board)
