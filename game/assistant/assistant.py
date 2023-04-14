import tkinter as tk

from assistant.info_frame import InfoFrame
from assistant.model.recognition import DigitRecognitionModel
from engine.board import Board


class ModelControl:

    def __init__(self, window: tk.Tk):
        self.frame = tk.Frame(window)
        self.model = DigitRecognitionModel()
        self.board = None

        self.train_button = tk.Button(self.frame,
                                      text="Train model for 1 epoch",
                                      command=self.train_model)
        self.train_button.grid(row=0, column=0)

        self.info_frame = InfoFrame(self.frame)
        self.info_frame.grid(row=1, column=0)

    def train_model(self) -> None:
        """Train the model for one epoch and update recommendations."""
        self.model.train_epoch()
        self.update_info()

    def set_board(self, board: Board) -> None:
        self.board = board

    def update_info(self) -> None:
        if self.board is None:
            return

        digits = [self.model.recognize(image) for image
                    in self.board.get_tile_images()]
        self.info_frame.set_vision_text(digits)

        if self.board_makes_sense(digits):
            self.info_frame.set_recommendation(self.get_recommendation(digits))
        else:
            self.info_frame.set_recommendation("The board does not make sense.")

    def board_makes_sense(self, digits: list[int]) -> bool:
        """True if all digits appear exactly once.
        
        Args:
          digits: flattened matrix of predicted digits.
        """
        return sorted(digits) == [0, 1, 2, 3, 4, 5, 6, 7, 8]

    def get_recommendation(self, digits: list[int]) -> str:
        if digits[0] != 1:
            return "Push 1 to the upper left corner"
        if digits[1] != 2 or digits[2] != 3:
            if digits[1] != 3:
                return "Push 3 to upper central position"
            if digits[4] != 2:
                return "Push 2 to the central position"
            return "Cycle 2 and 3 to put them in place"
        if digits[3] != 4 or digits[6] != 7:
            if digits[3] != 7:
                return "Push 7 to the central left position"
            if digits[4] != 4:
                return "Push 4 to the central position"
            return "Cycle 4 and 7 to put them in place"
        return "Cycle 5, 6 and 8"
