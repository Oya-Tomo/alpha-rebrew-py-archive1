import torch
from torch import nn
import numpy as np
import random
import sys
import pygame
from pygame.locals import *
from bitboard import Board, Stone, pos_to_idx
from model import PVNet
from mcts import MCT
from game.var import *
from game.surface import render_surf, render_bg_pattern, render_board


def render_countdown(
    screen: pygame.Surface, clock: pygame.time.Clock, board: Board
) -> None:
    frame_count = 0
    countdown = 3

    text_ready = FONTS[50].render("Are you ready ?", True, YELLOW)

    while True:
        render_bg_pattern(screen)
        render_board(screen, board)

        frame_count += 1
        count = countdown - frame_count // 30
        if count == 0:
            break

        text_count = FONTS[80].render(str(count), True, YELLOW)
        render_surf(screen, text_ready, 0.5, 0.3)
        render_surf(screen, text_count, 0.5, 0.5)

        pygame.display.update()
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()


def predict(mct: MCT, model: nn.Module, stone: Stone, board: Board) -> int:
    actions = board.get_actions(stone)

    with torch.no_grad():
        policy = mct.search(board, stone, 1000)
        policy = np.array(policy, dtype=np.float32)

        while True:
            action = int(random.choice(np.where(policy == policy.max())[0]))

            if action in actions:
                break
            else:
                policy[action] = -1

        return action


def get_click_sq(
    pos: tuple[int, int], screen_size: tuple[int, int], board_size: tuple[int, int]
) -> tuple[int, int]:
    x, y = pos
    screen_width, screen_height = screen_size
    board_width, board_height = board_size
    sq_size = board_width // 8

    x = (x - (screen_width - board_width) / 2) // sq_size
    y = (y - (screen_height - board_height) / 2) // sq_size

    return x, y


def render_player_turn(
    screen: pygame.Surface, clock: pygame.time.Clock, board: Board, turn: Stone
) -> Board:
    actions = board.get_actions(turn)

    if actions == [64]:
        return board

    while True:
        render_bg_pattern(screen)
        render_board(screen, board)

        pygame.display.update()
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()
            elif event.type == MOUSEBUTTONDOWN:
                pos = event.pos
                action = pos_to_idx(pos[0], pos[1])
                if action in actions:
                    return board.act(turn, action)


def render_ai_turn(
    screen: pygame.Surface,
    board: Board,
    turn: Stone,
    mct: MCT,
    model: nn.Module,
) -> Board:
    text_thinking = FONTS[50].render("Thinking...", True, YELLOW)

    render_bg_pattern(screen)
    render_board(screen, board)
    render_surf(screen, text_thinking, 0.5, 0.5)

    action = predict(mct, model, turn, board)
    if action == [64]:
        return board
    return board.act(turn, action)


def render_pass(screen: pygame.Surface, board: Board) -> None:
    pass
