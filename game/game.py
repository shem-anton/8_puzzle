import tkinter as tk
from tkinter.messagebox import showinfo

from engine.board import Board
from assistant.assistant import ModelControl


class Game:
    """Main class of the 8 puzzle game.

    Contains two tkinter frames. One is the game board, the other is the
    assistant. Defines menu with user manual.
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.wm_title("Game of 8")
        self.create_menu()

        self.assistant = ModelControl(self.root)
        self.assistant.frame.grid(row=0, column=1, padx=5, pady=5)

        self.board = Board(window=self.root,
                           solve_callback=self.on_solution,
                           move_callback=self.on_move)
        self.board.frame.grid(row=0, column=0, padx=5, pady=5)

        self.assistant.set_board(self.board)

    def run(self) -> None:
        """Start the game."""
        self.root.mainloop()

    def on_solution(self) -> None:
        """Destroy the window when the puzzle is solved."""
        self.root.destroy()

    def on_move(self) -> None:
        """Notify assistant about the changes board state."""
        self.assistant.update_info()

    def create_menu(self) -> None:
        """Create the window menu with Help section."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        help_menu = tk.Menu(menubar)
        help_menu.add_command(label="About the game", command=self.show_game_help)

        menubar.add_cascade(label="Help", menu=help_menu)

    def show_game_help(self) -> None:
        """Open a pop up window with Help text."""
        text = "Welcome to the 8 puzzle! The board has 8 tiles numbered 1 "\
        "through 8 and one empty tile. The goal of the puzzle is to slide "\
        "tiles to arrive at the ordered position with empty tile in bottom "\
        "right corner. The top row should be 1 2 3, the second 4 5 6 and the "\
        "last row should be 7 8. You can move a neighboring tile into an empty"\
        " position, just press at the tile you want to move."

        showinfo("How to play", text)


if __name__ == "__main__":
    game = Game()
    game.run()
