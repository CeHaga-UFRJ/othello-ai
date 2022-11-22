import itertools
import numpy as np
import random

from reversi import *
import nn

class HumanPlayer:
    def getMove(self, board, tile):
        return getPlayerMove(board, tile)

class RandomPlayer:
    def getMove(self, board, tile):
        return getRandomMove(board, tile)

class SimplePlayer:
    def getMove(self, board, tile):
        return getComputerMove(board, tile)

class RLPlayer:
    def __init__(self, q_lr, discount_factor, net_lr = 0.01):
        # We ougth to use softmax in this
        self.policy_net = nn.NN([64, 128, 128, 64, 64], net_lr)

        # This ought to decay
        self.epsilon = 0.6

        # Variables for Q learning
        self.q_lr = q_lr
        self.discount_factor = discount_factor
        self.play_history = []
        self.wins = 0

    def getMove(self, board, tile, log_history = True):
        movex = 0
        movey = 0
        board_state = np.array(board)

        # Transform all of "this player's" tokens to 1s, the other player's to -1s and empty spaces to 0s
        input_state = np.apply_along_axis(lambda x: 1 if x==tile else 0 if x==[" "] else -1, 1, board_state.reshape((64, 1))).reshape((64,1))


        # epsilon greedy to pick random move
        if np.random.random() < self.epsilon:
            movex, movey = getRandomMove(board, tile)

        else:
            out = self.policy_net.getOutput(input_state)

            # Sort the possible moves lowest to highest desire
            possible_moves = getValidMoves(board, tile)
            possible_moves.sort(key=lambda x: out[x[0]*8+x[1]])

            # Pick the highest desire move
            movex, movey = possible_moves[-1]

        if log_history and movex != 9 and movey != 9:
            self.play_history.append((np.copy(input_state), movex*8 + movey))

        return movex, movey

    def updateWeights(self, final_score):
        i = 0
        state, action = self.play_history[i]
        q = self.policy_net.getOutput(state)
        n_play_history = len(self.play_history)

        while i < n_play_history:
            i += 1
            
            if i == n_play_history:
                q[action] = final_score

            else:
                state_, action_ = self.play_history[i]
                q_next = self.policy_net.getOutput(state_)
                q[action] += self.q_lr * (self.discount_factor * q_next.max() - q[action])
                # q[action] += self.discount_factor * np.max(q_next)
            
            self.policy_net.backProp(state, self.policy_net.mkVec(q))

            if i != n_play_history:
                action, q = action_, q_next

    def train(self, n_games, verbose = False):
        for i in range(n_games):
            board = getNewBoard()
            resetBoard(board)
            turn = random.randint(0, 1)
            can_play = True
            
            print("Game: ", i)
            while can_play:
                if turn == 0:
                    if getValidMoves(board, "X"):
                        movex, movey = self.getMove(board, "X")
                        makeMove(board, "X", movex, movey)
                    else:
                        can_play = False if not getValidMoves(board, "O") else True
                else:
                    if getValidMoves(board, "O"):
                        movex, movey = self.getMove(board, "O")
                        makeMove(board, "O", movex, movey)
                    else:
                        can_play = False if not getValidMoves(board, "X") else True

                turn = (turn + 1) % 2

            if verbose:
                drawBoard(board)
                print("X: %s, O: %s" % (getScoreOfBoard(board)["X"], getScoreOfBoard(board)["O"]))

            if getScoreOfBoard(board)["X"] > getScoreOfBoard(board)["O"]:
                self.wins += 1
                self.updateWeights(1)
            else:
                self.updateWeights(-1)

            self.play_history = []
            self.epsilon *= 0.999

        self.policy_net.save("weights-%s-games" % n_games)
        return self.wins

if __name__ == "__main__":
    # Test the AI
    player = RLPlayer(0.1, 0.9)
    player.train(1000)
    print(player.wins)