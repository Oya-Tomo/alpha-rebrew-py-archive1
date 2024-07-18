import sys
import pygame
from pygame.locals import *
from game.var import *
from game.surface import render_surf


def render_splash(screen: pygame.Surface, clock: pygame.time.Clock):
    frame_count = 0
    splash_image = pygame.image.load("assets/othello_logo.png").convert_alpha()
    splash_image.set_alpha(0)

    while frame_count < 30 * 5:
        screen.fill(BLACK)
        if frame_count < 50:
            splash_image.set_alpha(frame_count * 5)
        elif frame_count < 120:
            splash_image.set_alpha(255)
        else:
            splash_image.set_alpha((150 - frame_count) * 8)

        render_surf(screen, splash_image, 0.5, 0.5)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        frame_count += 1
        clock.tick(30)
