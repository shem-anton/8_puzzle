import tkinter as tk

from PIL import Image, ImageTk
import PIL.ImageOps

# Size of a square tile in pixels
TILE_SIZE = 200


class Tile:
    """Represent one tile of the game board.

    The class stores tkinter button which is the board tile and the image on
    that button. Image needs to be stored separately, otherwise it will get
    cleared by garbage collector and no image will be displayed.

    Args:
      window: Parent window, which contains the tkinter button.
      value: Integer value, which is depicted on the image. 0 correspons to the
        empty tile.
      image_file: Name of the image with button contents.
    """

    def __init__(self, window: tk.Tk, value: int, image_file: str):
        self.value = value
        if self.value == 0:
            image = Image.new('L', (TILE_SIZE, TILE_SIZE), 255)
            self.image = ImageTk.PhotoImage(image)
            self.button = tk.Button(window,
                                    image=self.image,
                                    borderwidth=0,
                                    state="disabled")
        else:
            self.image = self._load_button_image(image_file)
            self.button = tk.Button(window, image=self.image, borderwidth=0)

    def _load_button_image(self, filename: str) -> tk.PhotoImage:
        """Load image from the disc as tkinter image object.

        MNIST images are white on black. To look better to user images are 
        inverted and resized to fit tile size.
        """
        image = Image.open(filename)
        image = image.resize((TILE_SIZE, TILE_SIZE))

        image = PIL.ImageOps.invert(image)
        return ImageTk.PhotoImage(image)
