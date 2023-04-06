import tkinter as tk
import random

import pandas as pd

from engine.tile import Tile

IMAGES_CSV = "data/test.csv"


class Board:
    """The tkinter window representing a gaming board of 9 tiles.

    On initialization creates a random solvable board. Pressing on a tile next
    to empty tile will swap the tile with empty tile. The goal is to align all
    tiles in order from 1 to 8 with empty tile in the bottom right corner.
    """

    def __init__(self):
        self.window = tk.Tk()
        self.window.wm_title("Game of 8")

        self.tiles = self.initialize_tiles(self.window)
        self._draw_buttons()

        self.window.mainloop()

    def initialize_tiles(self, window: tk.Tk) -> list[list[Tile]]:
        """Initialize the matrix of board tiles.

        Args:
          window: the parent tkinter window.
        
        Returns:
          The 3x3 matrix of initialized tiles.
        """
        tiles = [[None] * 3 for i in range(3)]
        initial_permutation = self._generate_solvable_permutation()

        for i in range(3):
            for j in range(3):
                index = i * 3 + j
                value = initial_permutation[index]

                if value == 0:
                    # 0 encodes an empty tile
                    tiles[i][j] = Tile(window=window,
                                       value=value,
                                       image_file=None)
                    self.empty = (i, j)
                else:
                    filename = self._pick_image(value)
                    tiles[i][j] = Tile(window=window,
                                       value=value,
                                       image_file=filename)
        return tiles

    def _draw_buttons(self) -> None:
        """Display the buttons according to tile matrix and set listeners."""
        for i in range(3):
            for j in range(3):
                self.tiles[i][j].button.grid(row=i, column=j)
                self.tiles[i][j].button.configure(
                    command= lambda x=i, y=j: self._press_tile(x, y))

        if self._is_solved():
            self.window.destroy()

    def _pick_image(self, number: int) -> str:
        """Randomly selects an image of given digit from MNIST test set.
        
        Args:
          number: Integer digit, drawn on the image.

        Returns:
          Filename of the image in data/ directory.
        """
        df = pd.read_csv(IMAGES_CSV)
        df = df.loc[df['labels'] == number]

        random_row = df.sample(n=1)
        file = random_row['image_paths'].values[0]
        return f"data/{file}"

    def _press_tile(self, i: int, j: int) -> None:
        """Process tile button action.

        If the button is next to an empty tile swap the tile with empty tile and
        reload the board. Update the empty tile.

        Args:
          i: selected row.
          j: selected column.
        """
        if self._is_next_to_empty(i, j):
            self.tiles[self.empty[0]][self.empty[1]], self.tiles[i][j] = \
                self.tiles[i][j], self.tiles[self.empty[0]][self.empty[1]]

            self.empty = (i, j)

        self._draw_buttons()

    def _is_solved(self) -> bool:
        """Check if the board is in solved state.

        Return:
          True if the board has form
            1  2  3
            4  5  6
            7  8  
        """
        solved = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        for i in range(3):
            for j in range(3):
                if self.tiles[i][j].value != solved[i][j]:
                    return False
        return True

    def _is_next_to_empty(self, i: int, j: int) -> bool:
        """Check if the selected tile is next to the empty tile."""
        # Check if L1 distance is exactly 1
        distance = abs(i - self.empty[0]) + abs(j - self.empty[1])
        return distance == 1

    def _generate_solvable_permutation(self) -> list[int]:
        """Generate random solvable permutation of board tiles.
        
        The board is solvable if the number of inversions plus the taxicab
        distance from empty tile to bottom right corner is even. The algorithm
        randomly shuffles the initial indices until a solvable permutation is
        reached.

        Returns:
          A list of indices 0-8 where 0 encodes an empty tile. The first three
            values correspond to the first row of the board, the next 3 to the
            second and the final 3 to the third.
        """
        indices = list(range(9))
        random.shuffle(indices)

        while self._calculate_permutation_parity(indices) % 2 != 0:
            random.shuffle(indices)
        return indices

    def _calculate_permutation_parity(self, permutation: list[int]) -> int:
        """Compute permutation parity plus 0 distance to final position parity.

        Args:
          permutation: The lsit of tile values read row after row. 0 encodes an
            empty tile.

        Returns:
          0 or 1 depending on the invariant parity of the permutation.
        """
        parity = 0
        # Count inversions
        for i in range(len(permutation)):
            for j in range(i + 1, len(permutation)):
                if permutation[i] > permutation[j]:
                    parity += 1

        # Tiles with odd index correspond to the odd taxicab distance from the
        # bottom right corner on 3x3 board
        parity += permutation.index(0) % 2

        return parity % 2
