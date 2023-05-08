# 8 puzzle
## Practicing the basics of tensorflow framework for recognizing MNIST digits.

The goal of "8 puzzle" is to shuffle the tiles on the board to order them from
1 to 8 with the empty tile in the bottom right corner. The player is only
allowed to move tiles in an empty space. Seems like an easy puzzle, but it is
not trivial! Fortunately an assistant helps you to find the next move. However
an assistant first needs to recognize the boards, because digits are presented 
as images. Here is where a neural network comes into play.

To run the game you need *tensorflow* and *tkinter*. When the dependencies are
installed simply run

    python3 game/game.py

Unfortunately, the neural network very rarely learns to the point where
it can recognize every digit. It gets pretty close, but a few tiles are
typically guessed wrong. Maybe it is because of all incorrect labels in MNIST
dataset ¯\\_(ツ)_/¯
