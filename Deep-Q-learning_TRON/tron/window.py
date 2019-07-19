
import pygame

class Window:

    def __init__(self, game, factor):

        self.factor = factor

        (width, height) = (factor * (game.width + 2), factor * (game.height + 2))

        self.screen = pygame.display.set_mode((width, height))
        self.render_map(game.map())

    def scale_box(self, row, col, width, height):

        return [row * self.factor, col * self.factor, width * self.factor, height * self.factor]

    def render_map(self, game_map):

        self.screen.fill((255, 255, 255))

        pygame.draw.rect(
            self.screen,
            (0, 0, 0),
            self.scale_box(1, 1, game_map.width, game_map.height))

        for row in range(game_map.height):
            for col in range(game_map.width):
                block = game_map[row,col]
                if block:
                    pygame.draw.rect(
                        self.screen,
                        block.color(),
                        self.scale_box(col+1.1, row+1.1, 1, 1))

        pygame.display.flip()
