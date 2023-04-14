import tkinter as tk


class InfoFrame(tk.Frame):

    def __init__(self, root: tk.Frame):
        super().__init__(root)

        vision_text = "I don't see the board"
        self.vision_label = tk.Label(self, text=vision_text)
        self.vision_label.pack(pady=10)

        recommendation_text = ""
        self.recommendation_label = tk.Label(self, text=recommendation_text)
        self.recommendation_label.pack(pady=10)

    def set_vision_text(self, digits: list[int]) -> None:
        """Display the information about how the model perceives board.
        
        Args:
          digits: flattened matrix of predicted digits.
        """
        digits = [str(x) if x!=0 else "□" for x in digits]
        vision_text = "Here is the board I see:"
        for i in range(3):
            line = ""
            for j in range(3):
                line += digits[i * 3 + j]
            vision_text += f"\n{line}"
        self.vision_label.config(text=vision_text)

    def set_recommendation(self, text: str) -> None:
        """Display the recommendation text.
        
        Args:
          text: a line to display.
        """
        self.recommendation_label.config(text=text)
