
import numpy as np
from enum import Enum

def is_on_border(i, j, w ,h):
    return i == 0 or i == w - 1 or j == 0 or j == h - 1

class Tile(Enum):

    EMPTY = 0
    WALL = -1
    PLAYER_ONE_BODY = 1
    PLAYER_ONE_HEAD = 2
    PLAYER_TWO_BODY = 3
    PLAYER_TWO_HEAD = 4

    def color(self):

        if self == Tile.EMPTY:
            return (0, 0, 0)
        elif self == Tile.WALL:
            return (255, 255, 255)
        elif self == Tile.PLAYER_ONE_BODY:
            return (0, 17, 128)
        elif self == Tile.PLAYER_ONE_HEAD:
            return (0, 34, 255)
        elif self == Tile.PLAYER_TWO_BODY:
            return (128, 17, 0)
        elif self == Tile.PLAYER_TWO_HEAD:
            return (255, 34, 0)
        else:
            return None

class Map:

    def __init__(self, w, h, empty, wall):

        self.width = w
        self.height = h
        self._data = np.array([[wall if is_on_border(i, j, w + 2, h + 2) else empty for i in range(h + 2)] for j in range(w + 2)])

    def clone(self):

        clone = Map(self.width, self.height, 0, 0)
        clone._data = np.copy(self._data)
        return clone

    def apply(self, converter):

        converted = Map(self.width, self.height, 0, 0)
        converted._data = np.array([[converter(self._data[i][j]) for i in range(self.height + 2)] for j in range(self.width + 2)])
        return converted

    def array(self):

        return self._data

    def clone_array(self):

        clone_map = self.clone()
        return clone_map._data

    def color(self, t, p):

        if t == Tile.EMPTY:
            return 1
        elif t == Tile.WALL:
            return -1
        elif t == Tile.PLAYER_ONE_BODY:
            return -1
        elif t == Tile.PLAYER_ONE_HEAD:
            return 10 if p == 1 else -10
        elif t == Tile.PLAYER_TWO_BODY:
            return -1
        elif t == Tile.PLAYER_TWO_HEAD:
            return 10 if p == 2 else -10
        else:
            return None

    def state_for_player(self, p):

        return self.apply(lambda tile: self.color(tile, p)).array()

    def __getitem__(self, index):
        (i, j) = index
        return self._data[i+1][j+1]

    def __setitem__(self, position, other):
        (i, j) = position
        self._data[i+1][j+1] = other

