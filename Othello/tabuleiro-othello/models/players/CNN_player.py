from models.players.CNN_model.cnn_utils import othello, decisionRule_ai
from models.move import Move
import PIL
from fastai.basic_train import load_learner
from fastai.vision import Image, pil2tensor
import numpy as np

class CNN_player(othello.ai):  

    def __init__(self, color):
        self.color = color
        self.marker = 1
        self.name = "CNN"
        self.learn = load_learner("", "Othello/tabuleiro-othello/models/players/CNN_model/trained_othello_CNN.pkl")
        self.learn.model.float()
        self.greedy = decisionRule_ai(1)
        self.search_mode = 'NN'
        self.depth = 0

    def board_to_input(self, board):
        boardInput = np.zeros((8, 8, 3))
        for i in range(8):
            for j in range(8):
                if board.get_square_color(i+1, j+1) == self.color:
                    boardInput[i][j][0] = 1
                elif board.get_square_color(i+1, j+1) == board._opponent(self.color):
                    boardInput[i][j][1] = 1
        return boardInput

    def play(self, board):
        board_input = self.board_to_input(board)
        player1_mat = board_input[:, :, 0]
        player2_mat = board_input[:, :, 1]
        empty_mat = np.zeros((8, 8))
        mat_3 = np.dstack((player1_mat, player2_mat, empty_mat))
        mat_img = PIL.Image.fromarray((mat_3).astype(np.uint8))
        mat_tensor = pil2tensor(mat_img, np.float32)
        mat_Image = Image(mat_tensor)

        move = self.learn.predict(mat_Image)
        move_string = str(move[0])
        x = int(move_string[0])
        y = int(move_string[2])
        move_output = Move(x, y)

        if (not (move_output in board.valid_moves(self.color))):
            # If not a valid move, then use greedy algo
            
            # This error print happens too often....
            #print("ERROR!")
            move_output = self.greedy.getMove(board, self.color)

        #print("NN move: ", str(move_output[0]), ",", str(move_output[1]))
        return move_output
