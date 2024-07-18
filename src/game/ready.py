import sys
import pygame
from pygame.locals import *
from pygame.rect import Rect
from bitboard import Stone
from game.surface import render_surf, render_bg_pattern
from game.var import *


def render_color_select(screen: pygame.Surface, clock: pygame.time.Clock) -> Stone:
    text1 = FONTS[50].render("Click your piece color.", True, WHITE)
    text2 = FONTS[30].render("Quit: Press ESC", True, WHITE)

    cursor = (-1, -1)

    while True:
        screen.fill(BLACK)
        render_bg_pattern()
        render_surf(screen, text1, 0.5, 0.1)
        render_surf(screen, text2, 0.5, 0.9)

        # Black
        black_piece_box_rect = Rect(0, 0, 300, 400)
        black_piece_box_rect.center = (screen.get_width() // 4, 450)
        pygame.draw.rect(screen, GREEN, black_piece_box_rect, border_radius=10)

        if black_piece_box_rect.collidepoint(cursor):
            black_piece_shadow_rect = Rect(0, 0, 200, 200)
            black_piece_shadow_rect.center = (
                screen.get_width() // 4 + 5,
                450 + 5,
            )
        else:
            black_piece_shadow_rect = Rect(0, 0, 200, 200)
            black_piece_shadow_rect.center = (
                screen.get_width() // 4 + 8,
                450 + 8,
            )
        pygame.draw.circle(screen, GREEN_SHADOW, black_piece_shadow_rect.center, 100)

        black_piece_rect = Rect(0, 0, 200, 200)
        black_piece_rect.center = (screen.get_width() // 4, 450)
        pygame.draw.circle(screen, BLACK, black_piece_rect.center, 100)

        # White
        white_piece_box_rect = Rect(0, 0, 300, 400)
        white_piece_box_rect.center = (screen.get_width() // 4 * 3, 450)
        pygame.draw.rect(screen, GREEN, white_piece_box_rect, border_radius=10)

        if white_piece_box_rect.collidepoint(cursor):
            white_piece_shadow_rect = Rect(0, 0, 200, 200)
            white_piece_shadow_rect.center = (
                screen.get_width() // 4 * 3 + 5,
                450 + 5,
            )
        else:
            white_piece_shadow_rect = Rect(0, 0, 200, 200)
            white_piece_shadow_rect.center = (
                screen.get_width() // 4 * 3 + 8,
                450 + 8,
            )
        pygame.draw.circle(screen, GREEN_SHADOW, white_piece_shadow_rect.center, 100)

        white_piece_rect = Rect(0, 0, 200, 200)
        white_piece_rect.center = (screen.get_width() // 4 * 3, 450)
        pygame.draw.circle(screen, WHITE, white_piece_rect.center, 100)

        clock.tick(30)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()
            elif event.type == MOUSEMOTION:
                cursor = event.pos
            elif event.type == MOUSEBUTTONDOWN:
                if black_piece_box_rect.collidepoint(event.pos):
                    return Stone.BLACK
                elif white_piece_box_rect.collidepoint(event.pos):
                    return Stone.WHITE
