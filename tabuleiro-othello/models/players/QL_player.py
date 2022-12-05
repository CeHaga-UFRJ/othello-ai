import numpy as np
import math
import pickle
from scipy.special import expit
from models.move import Move

# Constants
EPSILON = 0.6

class NN:
    def __init__(self, layer_dims, learning_rate):
        self.learning_rate = learning_rate
        self.layer_dims = layer_dims
        self.layers = []
        for i in range(len(layer_dims)-1):
            self.layers.append(np.random.normal(0, 1, size=(layer_dims[i+1], layer_dims[i]+1)))

    def load(self, filename):
        with open("Othello/tabuleiro-othello/models/players/QL_model/" + filename, "rb") as f:
            self.layers = pickle.load(f)

    def mkVec(self, vector1D, add_bias = True):
        return np.reshape(vector1D, (len(vector1D), 1))

    def getOutput(self, input_vector):
        output = input_vector
        for i in range(len(self.layers)):
            output = self.activation(self.layers[i].dot(np.vstack((output, 1))))

        return output

    def activation(self, x):
        # return expit(x)
        return np.tanh(x)

class RLPlayer:
    def __init__(self, color, net_lr = 0.01):
        self.policy_net = NN([64, 50, 64], net_lr)
        self.policy_net.load("weights-200000-games")

        # This ought to decay
        self.epsilon = EPSILON
        self.color = color

    def play(self, board):
        movex = 0
        movey = 0
        board_matrix = board.board[1:-1]
        board_vision = []

        for li in board_matrix:
            board_vision.append(li[1:-1])

        board_state = np.array(board_vision)

        # Transform all of "this player's" tokens to 1s, the other player's to -1s and empty spaces to 0s
        input_state = np.apply_along_axis(lambda x: 1 if x==self.color else 0 if x==[board.EMPTY] else -1, 1, board_state.reshape((64, 1))).reshape((64,1))
        out = self.policy_net.getOutput(input_state)

        # Sort the possible moves lowest to highest desire
        possible_moves = board.valid_moves(self.color)
        possible_moves.sort(key=lambda move: out[(move.x-1)*8+(move.y-1)])

        # Pick the highest desire move
        move = possible_moves[-1]

        return move