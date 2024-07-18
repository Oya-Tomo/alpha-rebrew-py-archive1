import sys, os
import random
import torch
from torch import nn
from bitboard import Board, Stone
from model import PVNet
from mcts import MCT
import numpy as np
import pygame
from pygame.locals import *

from game.splash import render_splash
from game.ready import render_color_select


pygame.init()


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN_BLACK = (0, 50, 0)
GREEN = (10, 180, 30)
GREEN_SHADOW = (10, 100, 10)
BLUE_BLACK = (0, 0, 70)
YELLOW = (255, 200, 0)
PURPLE = (150, 0, 255)

FONTS = {
    10: pygame.font.SysFont("Noto Sans CJK JP", 10),
    20: pygame.font.SysFont("Noto Sans CJK JP", 20),
    30: pygame.font.SysFont("Noto Sans CJK JP", 30),
    40: pygame.font.SysFont("Noto Sans CJK JP", 40),
    50: pygame.font.SysFont("Noto Sans CJK JP", 50),
    60: pygame.font.SysFont("Noto Sans CJK JP", 60),
    70: pygame.font.SysFont("Noto Sans CJK JP", 70),
    80: pygame.font.SysFont("Noto Sans CJK JP", 80),
    90: pygame.font.SysFont("Noto Sans CJK JP", 90),
    100: pygame.font.SysFont("Noto Sans CJK JP", 100),
}


class Game:
    def __init__(self) -> None:
        self.screen = pygame.display.set_mode((1000, 850))
        pygame.display.set_caption("Othello AI")
        self.clock = pygame.time.Clock()

        self.save_data = torch.load("checkpoint/model_19.pt")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PVNet().to(self.device)
        self.model.load_state_dict(self.save_data["model"])

        self.mct: MCT = None
        self.board: Board = None

        self.turn = Stone.BLACK
        self.player = Stone.BLACK

        self.scenes: list[function] = [
            self.render_splash,
            self.render_color_select,
        ]

        self.cursor = (-1, -1)

    def run(self) -> None:
        while True:
            for scene in self.scenes:
                scene()

    def render_splash(self) -> None:
        render_splash(self.screen, self.clock)

    def render_color_select(self) -> None:
        stone = render_color_select(self.screen, self.clock)
        self.player = stone
        self.board = Board()
        self.mct = MCT(self.model, 0.01)

    def render_match(self) -> None:
        while True:
            if self.board.is_over():
                break

            actions = self.board.get_actions(self.turn)
            if actions == [64]:
                pass


if __name__ == "__main__":
    game = Game()
    game.run()
