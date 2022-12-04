import keras
from keras.layers import BatchNormalization, Dense, Activation, Conv2D, Flatten
from keras.layers import Input, Add
from keras.optimizers import Adam
from keras import backend as K
import random
import time
import numpy as np
from OthelloBoard import * 
import copy
from AlphaBeta import AlphaBeta
import math

# How many neurons for each layer
LAYER_SIZE = 256

# Here, REWARD_DECAY is how much we care about the delayed reward compared to
# the immediate reward. REWARD_DECAY = 1 means we care about all reward the
# same, REWARD_DECAY = 0 means we don't care at all about the later rewards.
REWARD_DECAY = 0.99

# Size of the mini-batches used in training
BATCH_SIZE = 64

def reverse(array):
    newarray = copy.deepcopy(array)
    d = {1:-1, 0:0, -1:1}
    for i in range(len(array)):
        for j in range(len(array[0])):
            newarray[i][j] = d[array[i][j]]
    return newarray

def rotate_array(array):
    for i in range(len(array)):
        array[0] = rotate_90(array[0])
        array[1] = rotate_90(array[1])
        array[2] = rotate_90(array[2])

    return array

def process(array):
    new_array = []
    pieces = [0, 1, -1]
    
    for i in range(3):
        board = []
        for j in range(8):
            row = []
            for k in range(8):
                row.append(int(array[j][k] == pieces[i]))
            board.append(row)
        new_array.append(board)

    return new_array

def rotate_90(array):
    # Array is a 8x8 array.
    # ccw rotation

    new_array = []

    for i in range(8):
        row = []
        for j in range(8):
            row.append(array[7-j][i])
        new_array.append(row)

    return new_array

class OthelloPlayer:
    def __init__(self, index, depth, parent = None, learning_rate = 0.00005, epsilon = 2,
                 epsilon_increment = 0.00005, debugging = False):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_increment = epsilon_increment
        self.experience = []
        self.debugging = debugging
        self.parent = parent
        self.index = index
        self.depth = depth

        self.create_model()

    def create_model(self):
        main_input = Input(shape = (3,8,8))

        c1 = Conv2D(LAYER_SIZE, (3,3), activation = 'relu', padding = 'same')(main_input)
        b1 = BatchNormalization()(c1)
        c2 = Conv2D(LAYER_SIZE, (3,3), activation = 'relu', padding = 'same')(b1)
        b2 = BatchNormalization()(c2)
        c3 = Conv2D(LAYER_SIZE, (3,3), activation = 'relu', padding = 'same')(b2)
        b3 = BatchNormalization()(c3)

        a3 = Add()([b3, b1])

        c4 = Conv2D(LAYER_SIZE, (3,3), activation = 'relu', padding = 'same')(a3)
        b4 = BatchNormalization()(c4)
        c5 = Conv2D(LAYER_SIZE, (3,3), activation = 'relu', padding = 'same')(b4)
        b5 = BatchNormalization()(c5)

        a5 = Add()([b5, a3])

        b6 = Conv2D(LAYER_SIZE, (3,3), activation = 'relu', padding = 'same')(a5)
        
        f1 = Flatten()(b6)
        d1 = Dense(LAYER_SIZE, activation = 'relu')(f1)
        d2 = Dense(1, activation = 'tanh')(d1)

        self.model = keras.models.Model(inputs = main_input, outputs = d2)

        self.model.compile(Adam(self.learning_rate), "mse")


    def add_to_history(self, state_array, reward):
        answers = []
        history = self.experience

        current_reward = reward

        processed_array = []

        for i in range(len(state_array)):
            processed_array.append(process(state_array[i]))

        state_array = processed_array
        
        for i in range(len(state_array)):
            current_array = state_array[len(state_array) - i - 1]
            
            history.append([current_array,
                                 current_reward])
            current_array = rotate_array(current_array)
            history.append([current_array,
                                 current_reward])
            current_array = rotate_array(current_array)
            history.append([current_array,
                                 current_reward])
            current_array = rotate_array(current_array)
            history.append([current_array,
                                 current_reward])
            current_reward *= REWARD_DECAY

    def wipe_history(self):
        self.experience = []

    def train_model(self, verbose):
        inputs = []
        answers = []
        history = self.experience
                  
        for i in range(BATCH_SIZE):
            lesson = random.choice(history)
            inputs.append(lesson[0])
            answers.append(lesson[1])

        inputs = np.array(inputs)
        answers = np.array(answers)
        
        self.model.fit(x = inputs, y = answers, verbose = 1)

    # Saves the model's weights.
    def save(self, s):
        self.model.save_weights(s)

    # Loads the weights of a previous model.
    def load(self, s):
        self.model.load_weights(s)
        #self.default_graph.finalize()

    def policy(self, observation, player):
        # Value is an array. The 0th element corresponds to (0,0), the 1st: (0,1)
        # the 8th: (1,0), etc.
        value = []

        if(player == -1):
            observation = reverse(observation)

        possible_moves = findMovesWithoutEnv(observation)

        if(len(possible_moves) == 0):
            # Passes
            return (-1, -1)

        if(self.debugging):
            print(possible_moves)
        
        decision_tree = AlphaBeta(self.parent)
        
        variation = random.random()
        
        if(variation < 1/self.epsilon):
            self.epsilon += self.epsilon_increment
            if(self.debugging):
                print("Random Move for player " + str(env.to_play))
            return random.choice(possible_moves)
        else:
            board = OthelloBoard(8)
            board.board = observation
            value, move = decision_tree.alphabeta(board, self.depth, -math.inf,
                                                  math.inf, 1, self.index)

            if(move == None):
                return (-1,-1)
            return move

class RandomPlayer(OthelloPlayer):
    def __init__(self):
        pass
    
    def add_to_history(self, state_array, reward):
        pass

    def wipe_history(self):
        pass
   
    def train_model(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def policy(self, observation, player):
        if(player == -1):
            observation = reverse(observation)
        
        possibleMoves = findMovesWithoutEnv(observation)

        if(len(possibleMoves) == 0):
            return (-1, -1)
  
        return random.choice(possibleMoves)

class BasicPlayer(RandomPlayer):
    def __init__(self, depth):
        self.weights = [[1000, 50, 100, 100, 100, 100, 50, 1000],
                   [50, -20, -10, -10, -10, -10, -20, 50],
                   [100, -10, 1, 1, 1, 1, -10, 100],
                   [100, -10, 1, 1, 1, 1, -10, 100],
                   [100, -10, 1, 1, 1, 1, -10, 100],
                   [100, -10, 1, 1, 1, 1, -10, 100],
                   [50, -20, -10, -10, -10, -10, -20, 50],
                   [1000, 50, 100, 100, 100, 100, 50, 1000]]

    def calculateScore(self, observation):
        score = 0
        for i in range(len(observation)):
            for j in range(len(observation[0])):
                score += observation[i][j] * self.weights[i][j]
        return score

    def policy(self, observation, player):
        if(player == -1):
            observation = reverse(observation)

        possibleMoves = findMovesWithoutEnv(observation)

        bestScore = -1000
        bestMove = (-1, -1)

        for move in possibleMoves:
            newobs = Board()
            newobs.pieces = copy.deepcopy(observation)
            newobs.execute_move(move, 1)
            tempScore = self.calculateScore(newobs.pieces)
            if(tempScore > bestScore):
                bestScore = tempScore
                bestMove = move
        
        return bestMove

class wisePlayer():
    depth = 4
    player = 1

    def  __init__(self):
        pass

    def valid_moves(self, observation, player):
        new_observation = observation

        if(player == -1):
            new_observation = reverse(observation)

        possibleMoves = findMovesWithoutEnv(new_observation)

        return possibleMoves


    def policy(self, observation, player = 1):       
        moves = self.valid_moves(observation, self.player)

        score, best_move = self.minmax(observation, self.depth, self.player, -math.inf, math.inf)
        print(best_move, moves)
        if best_move in moves:
            return best_move
        else:
            return moves[0] if len(moves) > 0 else [-1, -1]
    
    def minmax(self, observation, depth, player, alpha, beta):
        if depth == 0 or (not self.valid_moves(observation, player * -1) and not self.valid_moves(observation, player * -1)):
            return self.score(observation, player), None

        # Se o jogador atual não possuir movimentos válidos, passa a vez para o oponente
        if not self.valid_moves(observation, player):
            copy_observation = copy.deepcopy(observation)
            evaluation, _ = self.minmax(copy_observation, depth - 1, player * -1 , alpha, beta)
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                return -math.inf, None
            return evaluation, None

        if player == self.player:
            maxEval = -math.inf
            best_move = None
            for move in self.valid_moves(observation, player):
                env = OthelloBoard(8)
                env.board = copy.deepcopy(observation)
                env.to_play = player
                env.MakeMove(move[0], move[1], player)
                copy_observation = copy.deepcopy(env.board)
                evaluation, _ = self.minmax(copy_observation, depth - 1, player * -1, alpha, beta)
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
            for move in self.valid_moves(observation, player):
                env = OthelloBoard(8)
                env.board = copy.deepcopy(observation)
                env.to_play = player
                env.MakeMove(move[0], move[1], player)
                copy_observation = copy.deepcopy(env.board)
                evaluation, _ = self.minmax(copy_observation, depth - 1, player * -1, alpha, beta)
                if evaluation < minEval:
                    minEval = evaluation
                    best_move = move
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break
            return minEval, best_move

    def coin_parity(self, observation, max_player, min_player):
        env = OthelloBoard(8)
        env.board = copy.deepcopy(observation)
        score = env.score()
        return 100 * (score[max_player] - score[min_player]) / (score[max_player] + score[min_player])

    def mobility(self, observation, player):
        max_player_moves = len(self.valid_moves(observation, player))
        min_player_moves = len(self.valid_moves(observation, player * -1))
        if max_player_moves + min_player_moves == 0:
            return 0
        return 100 * (max_player_moves - min_player_moves) / (max_player_moves + min_player_moves)
    
    def corner_captured(self, observation, player):
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]

        max_score = 0
        min_score = 0

        env = OthelloBoard(8)
        env.board = copy.deepcopy(observation)

        for corner in corners:
            if env.get_square_player(corner[0], corner[1]) == player:
                max_score += 1
            elif env.get_square_player(corner[0], corner[1]) == player * -1:
                min_score += 1
        if max_score + min_score == 0:
            return 0
          
        return 100 * (max_score - min_score) / (max_score + min_score)

    def stability(self, observation, player):
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]

        max_score = 0
        min_score = 0

        for corner in corners:
            max_score += self.stability_corner(observation, corner, player)
            min_score += self.stability_corner(observation, corner, player * -1)
        
        return max_score - min_score
    
    def stability_corner(self, observation, corner, player):
        horizontal_direction = 1 if corner[0] == 1 else -1
        vertical_direction = 1 if corner[1] == 1 else -1

        stable_positions = []
        env = OthelloBoard(8)
        env.board = copy.deepcopy(observation)
        
        # Percorre a horizontal 
        for i in range(1, 9):
            corner_square = corner[0] + i * horizontal_direction, corner[1]
            
            # Se a posição conter uma peça do jogador, então percorre a vertical
            if env.get_square_player(corner_square[0], corner_square[1]) == player:
                for j in range(1, 9):
                    stable_square = corner_square[0], corner_square[1] + j * vertical_direction
                    # Se a posição conter uma peça do jogador, então a posição é estável
                    if env.get_square_player(stable_square[0], stable_square[1]) == player:
                        stable_positions.append(stable_square)
                    else:
                        break
            else:
                break
        
        return len(stable_positions)

    def score(self, observation, player):
        max_player = -1
        min_player = 1
        if player == 1:
            max_player, min_player = min_player, max_player
        
        env = OthelloBoard(8)
        env.board = copy.deepcopy(observation)
        score = env.score()

        discs = score[-1] + score[1]

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

        
        coin_score = coin_weight * self.coin_parity(observation, max_player, min_player)
        mobility_score = mobility_weight * self.mobility(observation, player)
        corner_score = corner_weight * self.corner_captured(observation, player)
        square_weight_score = 0# square_weight * self.square_weight(observation, player)
        stability_score = stability_weight * self.stability(observation, player)
        print(coin_score + mobility_score + corner_score + square_weight_score + stability_score)
        scores = [coin_score, mobility_score, corner_score, square_weight_score, stability_score]
        return coin_score + mobility_score + corner_score + square_weight_score + stability_score