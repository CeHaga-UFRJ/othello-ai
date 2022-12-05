import math
class MinMaxPlayer:
    depth = 6

    def __init__(self, color):
        self.color = color

    def play(self, board):
        valid_moves = board.valid_moves(self.color)
        score, best_move = self.minmax(board, self.depth, self.color, -math.inf, math.inf)
        if best_move in valid_moves:
            return best_move
        else:
            return valid_moves[0]

    def minmax(self, board, depth, color, alpha, beta):
        if depth == 0 or (not board.valid_moves(color) and not board.valid_moves(board._opponent(color))):
            return self.score(board, color), None

        # Se o jogador atual não possuir movimentos válidos, passa a vez para o oponente
        if not board.valid_moves(color):
            copy_board = board.get_clone()
            evaluation, _ = self.minmax(copy_board, depth - 1, board._opponent(color), alpha, beta)
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                return -math.inf, None
            return evaluation, None

        if color == self.color:
            maxEval = -math.inf
            best_move = None
            for move in board.valid_moves(color):
                copy_board = board.get_clone()
                copy_board.play(move, color)
                evaluation, _ = self.minmax(copy_board, depth - 1, board._opponent(color), alpha, beta)
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
            for move in board.valid_moves(color):
                copy_board = board.get_clone()
                copy_board.play(move, color)
                evaluation, _ = self.minmax(copy_board, depth - 1, board._opponent(color), alpha, beta)
                if evaluation < minEval:
                    minEval = evaluation
                    best_move = move
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break
            return minEval, best_move



    def score(self, board, color):
        if color == board.WHITE:
            return board.score()[0]
        else:
            return board.score()[1]