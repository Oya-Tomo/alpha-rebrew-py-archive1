import pygame
from bitboard import Board, Stone
from game.var import *


def render_surf(screen: pygame.Surface, surf: pygame.Surface, x: float, y: float):
    surf_width = surf.get_width()
    surf_height = surf.get_height()

    x = screen.get_width() * x - surf_width / 2
    y = screen.get_height() * y - surf_height / 2

    screen.blit(surf, (x, y))


def render_board(screen: pygame.Surface, board: Board) -> None:
    size = 800
    rect_size = size // 8
    board_surf = render_field(size)
    board_surf = render_stones(board, board_surf, rect_size)
    render_surf(screen, board_surf, 0.5, 0.5)


def render_field(size: int) -> pygame.Surface:
    rect_size = size // 8
    board_surf = pygame.Surface((size, size))
    board_surf.fill(GREEN_BLACK)
    for x in range(1, 8):
        lx = x * rect_size
        pygame.draw.line(board_surf, BLACK, (lx, 0), (lx, size), width=3)

    for y in range(1, 8):
        ly = y * rect_size
        pygame.draw.line(board_surf, BLACK, (0, ly), (size, ly), width=3)

    return board_surf


def render_stones(board: Board, surf: pygame.Surface, rect_size: int) -> pygame.Surface:
    radius = rect_size // 2 - 5
    stones = board.get_board()
    for y in range(8):
        for x in range(8):
            s = stones[y][x]
            color: Stone = (
                BLACK
                if s == Stone.BLACK
                else (WHITE if s == Stone.WHITE else Stone.EMPTY)
            )
            if color is Stone.EMPTY:
                continue
            pygame.draw.circle(
                surf,
                GREEN_SHADOW,
                (
                    rect_size * x + rect_size // 2 + 2,
                    rect_size * y + rect_size // 2 + 2,
                ),
                radius=radius,
            )
            pygame.draw.circle(
                surf,
                color,
                (
                    rect_size * x + rect_size // 2,
                    rect_size * y + rect_size // 2,
                ),
                radius=radius,
            )
    return surf


def render_bg_pattern(screen: pygame.Surface) -> None:
    bs = 50
    w = screen.get_width() // bs + 1
    h = screen.get_height() // bs + 1

    for y in range(h):
        for x in range(w):
            c = BLACK if (x + y) % 2 == 0 else BLUE_BLACK
            pygame.draw.rect(screen, c, (x * bs, y * bs, bs, bs))
