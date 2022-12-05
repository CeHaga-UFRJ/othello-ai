import random

class RandomPlayer:
    def __init__(self, color):
        self.color = color

    def play(self, board):
        return random.choice(board.valid_moves(self.color))
