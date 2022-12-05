import numpy as np
import math 
from models.move import Move
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense


class CNN_DEMO_player:  

  def create_model(self):
    layers = [64, 64, 128, 128]
    model = Sequential()
    model.add(Conv2D(layers[0], 3, padding='same',
              input_shape=(8, 8, 2)))
    for layer in layers[1:]:
        model.add(Conv2D(layer, 3, padding='same'))
    model.add(Flatten())
    model.add(Dense(units=128))
    model.add(Dense(units=64, activation='sigmoid'))
    return model

  def __init__(self, color):
    self.color = color
    self.model = self.create_model()
    self.model.load_weights('Othello/tabuleiro-othello/models/players/ckpt/weights.4-05-0.27.hdf5')

  def board_to_input(self, board):
    boardInput = np.zeros((8, 8, 2))
    for i in range(8):
        for j in range(8):
            if board.get_square_color(i+1, j+1) == self.color:
                boardInput[i][j][0] = 1
            elif board.get_square_color(i+1, j+1) == board._opponent(self.color):
                boardInput[i][j][1] = 1
    return boardInput

  def play(self, board):
      boardInput = self.board_to_input(board)
      output = self.model.predict(boardInput.reshape(1, 8, 8, 2),verbose = 0)
      valid_moves = board.valid_moves(self.color)

      while True:
        move_index = np.unravel_index(np.argmax(output, axis=None), output.shape)[1]
        move = Move(move_index // 8 + 1, move_index % 8 + 1)
        if move in valid_moves:
            return move
        else:
            output[0][move_index] = -math.inf  