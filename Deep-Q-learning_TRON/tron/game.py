from time import sleep
from enum import Enum

from tron.map import Map, Tile

class Winner(Enum):
    PLAYER_ONE = 1
    PLAYER_TWO = 2

class PositionPlayer:

    def __init__(self, id, player, position):
        self.id = id
        self.player = player
        self.position = position
        self.alive = True

    def body(self):
 
        if self.id == 1:
            return Tile.PLAYER_ONE_BODY
        elif self.id == 2:
            return Tile.PLAYER_TWO_BODY

    def head(self):

        if self.id == 1:
            return Tile.PLAYER_ONE_HEAD
        elif self.id == 2:
            return Tile.PLAYER_TWO_HEAD


class HistoryElement:

    def __init__(self, mmap, player_one_direction, player_two_direction):
        self.map = mmap
        self.player_one_direction = player_one_direction
        self.player_two_direction = player_two_direction


class Game:

    def __init__(self, width, height, pps):

        self.width = width
        self.height = height
        mmap = Map(width, height, Tile.EMPTY, Tile.WALL)
        self.history = [HistoryElement(mmap, None, None)]
        self.pps = pps
        self.winner = None

        for pp in self.pps:
            self.history[-1].map[pp.position[0], pp.position[1]] = pp.head()

    def map(self):

        return self.history[-1].map.clone()

    def next_frame(self, window = None):

        map_clone = self.map()

        for pp in self.pps:
            map_clone[pp.position[0], pp.position[1]] = pp.body()

        for id, pp in enumerate(self.pps):
            try:
                (pp.position, pp.player.direction) = pp.player.next_position_and_direction(pp.position, id + 1, self.map())
            except:
                print("ERRRRRRRRRRRRRRRRRRRRROR")
                if id == 0:
                    self.winner = 2
                elif id == 1:
                    self.winner = 1
                return False

        self.history[-1].player_one_direction = self.pps[0].player.direction
        self.history[-1].player_two_direction = self.pps[1].player.direction

        if window:
            import pygame
            while True:
                event = pygame.event.poll()

                if event.type == pygame.NOEVENT:
                    break

                for pp in self.pps:
                    try:
                        pp.player.manage_event(event)
                    except:
                        if id == 0:
                            self.winner = 2
                        elif id == 1:
                            self.winner = 1
                        return False

        for (id, pp) in enumerate(self.pps):

            if pp.position[0] < 0 or pp.position[1] < 0 or \
               pp.position[0] >= self.width or pp.position[1] >= self.height:

                pp.alive = False
                map_clone[pp.position[0], pp.position[1]] = pp.head()

            elif map_clone[pp.position[0], pp.position[1]] is not Tile.EMPTY:
                pp.alive = False
                map_clone[pp.position[0], pp.position[1]] = pp.head()

            else:
                map_clone[pp.position[0], pp.position[1]] = pp.head()

        self.history.append(HistoryElement(map_clone, None, None))

        return True

    def main_loop(self, window = None):

        if window:
            window.render_map(self.map())

        while True:
            alive_count = 0
            alive = None

            if window:
                sleep(0.3)
				#sleep(0.5)

            if not self.next_frame(window):
                break

            for pp in self.pps:
                if pp.alive:
                    alive_count += 1
                    alive = pp.id

            if alive_count <= 1:
                if alive_count == 1:
                    if self.pps[0].position[0] != self.pps[1].position[0] or \
                       self.pps[0].position[1] != self.pps[1].position[1]:

                       self.winner = alive
                break

            if window:
                window.render_map(self.map())
