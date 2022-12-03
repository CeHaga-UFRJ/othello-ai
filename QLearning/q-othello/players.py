import itertools
import numpy as np
import random

from reversi import *
import nn
import math

# Constants
EPSILON = 0.6

class WisePlayer:
    
    depth = 4
    tile = None

    def getOpponent(self, tile):
        if tile == 'X':
            return 'O'
        else:
            return 'X'

    def getTile(self, board, position):
        return board[position[0]][position[1]]

    def getMove(self, board, tile):
        self.tile = tile
        valid_moves = getValidMoves(board, tile)
        score, best_move = self.minmax(board, self.depth, tile, -math.inf, math.inf)
        scores = self.score(board, self.tile)
        # print("Score: ", scores[0])
        # print("Coin score: ", scores[1][0])
        # print("Mobility score: ", scores[1][1])
        # print("Corner score: ", scores[1][2])
        # print("Square weight score: ", scores[1][3])
        # print("Stability score: ", scores[1][4])
        if best_move in valid_moves:
            return best_move
        else:
            return valid_moves[0]

    def minmax(self, board, depth, tile, alpha, beta):
        if depth == 0 or (not getValidMoves(board, tile) and not getValidMoves(board, self.getOpponent(tile))):
            return self.score(board, tile)[0], None

        if not getValidMoves(board, tile):
            copy_board = getBoardCopy(board)
            evaluation, _ = self.minmax(copy_board, depth - 1, self.getOpponent(tile), alpha, beta)
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                return -math.inf, None
            return evaluation, None
        
        if tile == self.tile:
            maxEval = -math.inf
            best_move = None

            for move in getValidMoves(board, tile):
                copy_board = getBoardCopy(board)
                makeMove(copy_board, tile, move[0], move[1])
                evaluation, _ = self.minmax(copy_board, depth - 1, self.getOpponent(tile), alpha, beta)
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break
                if evaluation > maxEval:
                    maxEval = evaluation
                    best_move = move
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break
            return maxEval, best_move
        else:
            minEval = math.inf
            best_move = None

            for move in getValidMoves(board, tile):
                copy_board = getBoardCopy(board)
                makeMove(copy_board, tile, move[0], move[1])
                evaluation, _ = self.minmax(copy_board, depth - 1, self.getOpponent(tile), alpha, beta)
                beta = min(beta, evaluation)
                if evaluation < minEval:
                    minEval = evaluation
                    best_move = move
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break
            return minEval, best_move
    
    def coin_parity(self, board, max_player, min_player):
        score = getScoreOfBoard(board)
        return 100 * (score[max_player] - score[min_player]) / (score[max_player] + score[min_player])
    
    def mobility(self, board, tile):
        max_player_moves = len(getValidMoves(board, tile))
        min_player_moves = len(getValidMoves(board, self.getOpponent(tile)))
        if max_player_moves + min_player_moves == 0:
            return 0
        return 100 * (max_player_moves - min_player_moves) / (max_player_moves + min_player_moves)
        
    def corner_captured(self, board, tile):
        squares = [[0,0], [0,7], [7,0], [7,7]]

        max_score = 0
        min_score = 0

        for square in squares:
            if self.getTile(board, square) == tile:
                max_score += 1
            elif self.getTile(board, square) == self.getOpponent(tile):
                min_score += 1
        if max_score + min_score == 0:
            return 0
          
        return 100 * (max_score - min_score) / (max_score + min_score)

    def stability(self, board, tile):
        corners = [[0,0], [0,7], [7,0], [7,7]]

        max_score = 0
        min_score = 0

        for corner in corners:
            max_score += self.stability_corner(board, corner, tile)
            min_score += self.stability_corner(board, corner, self.getOpponent(tile))
        
        return max_score - min_score

    def stability_corner(self, board, corner, tile):
        horizontal_direction = 1 if corner[0] == 0 else -1
        vertical_direction = 1 if corner[1] == 0 else -1

        stable_positions = []
        # Percorre a horizontal 
        for i in range(0, 8):
            corner_square = corner[0] + i * horizontal_direction, corner[1]
            
            # Se a posição conter uma peça do jogador, então percorre a vertical
            if self.getTile(board, corner_square) == tile:
                for j in range(0, 8):
                    stable_square = corner_square[0], corner_square[1] + j * vertical_direction
                    # Se a posição conter uma peça do jogador, então a posição é estável
                    if self.getTile(board, stable_square) == tile:
                        stable_positions.append(stable_square)
                    else:
                        break
            else:
                break
        #if len(stable_positions) > 1:
        #    print(stable_positions)
        return len(stable_positions)
            

        
    def squere_weight(self, board, tile):
        weight = [
            [200, -100, 100,  50,  50, 100, -100,  200],
            [-100, -200, -50, -50, -50, -50, -200, -100],
            [100,  -50, 100,   0,   0, 100,  -50,  100],
            [50,  -50,   0,   0,   0,   0,  -50,   50],
            [50,  -50,   0,   0,   0,   0,  -50,   50],
            [100,  -50, 100,   0,   0, 100,  -50,  100],
            [-100, -200, -50, -50, -50, -50, -200, -100],
            [200, -100, 100,  50,  50, 100, -100,  200],
        ]

        max_score = 0
        min_score = 0
        for i in range(0, 7):
            for j in range(0, 7):
                if self.getTile(board, [i, j]) == tile:
                    max_score += weight[i][j]
                elif self.getTile(board, [i, j]) == self.getOpponent(tile):
                    min_score += weight[i][j]

        return max_score - min_score

    def countDiscs(self, board):
        discs = 0
        for i in range(0, 7):
            for j in range(0, 7):
                if self.getTile(board, [i, j]) == 'X' or self.getTile(board, [i, j]) == 'O':
                    discs += 1
        return discs

    def score(self, board, tile):
        max_player = 'O'
        min_player = 'X'
        if tile == 'X':
            max_player, min_player = min_player, max_player
        
        discs = self.countDiscs(board)

        coin_weight = 0
        mobility_weight = 0
        corner_weight = 10000
        stability_weight = 10000
        square_weight = 0

        # Early game
        if discs <= 20:
            mobility_weight = 5
            square_weight = 20
        # Mid Game
        elif discs <= 58:
            coin_weight = 10
            mobility_weight = 2
            square_weight = 100
        # Late Game
        else:
            coin_weight = 500
            mobility_weight = 0

        
        coin_score = coin_weight * self.coin_parity(board, max_player, min_player)
        mobility_score = mobility_weight * self.mobility(board, tile)
        corner_score = corner_weight * self.corner_captured(board, tile)
        square_weight_score = square_weight * self.squere_weight(board, tile)
        stability_score = stability_weight * self.stability(board, tile)

        scores = [coin_score, mobility_score, corner_score, square_weight_score, stability_score]
        return coin_score + mobility_score + corner_score + square_weight_score + stability_score, scores 


class HumanPlayer:
    def getMove(self, board, tile):
        return getPlayerMove(board, tile)

class GreedyPlayer:
    def getMove(self, board, tile):
        # randomize the order of the possible moves
        possibleMoves = getValidMoves(board, tile)
        random.shuffle(possibleMoves)

        # Go through all the possible moves and remember the best scoring move
        bestScore = -1
        for x, y in possibleMoves:
            dupeBoard = getBoardCopy(board)
            makeMove(dupeBoard, tile, x, y)
            score = getScoreOfBoard(dupeBoard)[tile]
            if score > bestScore:
                bestMove = [x, y]
                bestScore = score
        return bestMove

class RandomPlayer:
    def getMove(self, board, tile):
        return getRandomMove(board, tile)

class CornerPlayer:
    def getMove(self, board, tile):
        return getComputerMove(board, tile)

class RLPlayer:
    def __init__(self, q_lr, discount_factor, net_lr = 0.01):
        self.policy_net = nn.NN([64, 50, 64], net_lr)

        # This ought to decay
        self.epsilon = EPSILON

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

        if log_history:
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
                # q[action] += self.q_lr * (self.discount_factor * np.max(q_next) - q[action])
                q[action] += self.discount_factor * np.max(q_next)
            
            self.policy_net.backProp(state, self.policy_net.mkVec(q))

            if i != n_play_history:
                action, q = action_, q_next

    def train(self, n_games, verbose = False):
        # Train the network by playing 'n_games' games and updating the weights after each game
        for i in range(n_games):
            board = getNewBoard()
            resetBoard(board)
            turn = random.randint(0, 1)
            can_play = True
            
            # Play the game until it's over or a player can't play
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
                
            # Update weights based on final score
            scorex = getScoreOfBoard(board)["X"]
            scoreo = getScoreOfBoard(board)["O"]
            if scorex > scoreo:
                self.wins += 1
            self.updateWeights((scorex/(scorex+scoreo) - 0.5)*2)

            self.play_history = []

            # Decay epsilon linearly over each game until it reaches 0 (no random moves)
            newEpsilon = EPSILON - EPSILON * (i / n_games)
            self.epsilon = newEpsilon if newEpsilon > 0 else 0

        # Save the weights to a file
        self.policy_net.save("weights-%s-games" % n_games)

        return self.wins

if __name__ == "__main__":
    # Test the AI
    player = RLPlayer(0.1, 0.9)
    player.train(1000)
    print(player.wins)