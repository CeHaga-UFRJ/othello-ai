import math
# Player com square weights ativado
class WeightPlayer:
    depth = 6

    def __init__(self, color):
        self.color = color

    def play(self, board):
        valid_moves = board.valid_moves(self.color)
        score, best_move = self.minmax(board, self.depth, self.color, -math.inf, math.inf)
        scores = self.score(board, self.color)
        if best_move in valid_moves:
            return best_move
        else:
            return valid_moves[0]

    def minmax(self, board, depth, color, alpha, beta):
        if depth == 0 or (not board.valid_moves(color) and not board.valid_moves(board._opponent(color))):
            return self.score(board, color)[0], None

        # Se o jogador atual não possuir movimentos válidos, passa a vez para o oponente
        if not board.valid_moves(color):
            copy_board = board.get_clone()
            evaluation, _ = self.minmax(copy_board, depth - 1, board._opponent(color), alpha, beta)
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                return 0, None
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


    def coin_parity(self, board, max_player, min_player):
        score = board.score()
        return 100 * (score[max_player] - score[min_player]) / (score[max_player] + score[min_player])
    
    def mobility(self, board, color):
        max_player_moves = len(board.valid_moves(color))
        min_player_moves = len(board.valid_moves(board._opponent(color)))
        if max_player_moves + min_player_moves == 0:
            return 0
        return 100 * (max_player_moves - min_player_moves) / (max_player_moves + min_player_moves)
        
    def corner_captured(self, board, color):
        squares = [(1, 1), (1, 8), (8, 1), (8, 8)]

        max_score = 0
        min_score = 0

        for square in squares:
            if board.get_square_color(square[0], square[1]) == color:
                max_score += 1
            elif board.get_square_color(square[0], square[1]) == board._opponent(color):
                min_score += 1
        if max_score + min_score == 0:
            return 0
          
        return 100 * (max_score - min_score) / (max_score + min_score)

    def stability(self, board, color):
        corners = [(1, 1), (1, 8), (8, 1), (8, 8)]

        max_score = 0
        min_score = 0

        for corner in corners:
            max_score += self.stability_corner(board, corner, color)
            min_score += self.stability_corner(board, corner, board._opponent(color))
        
        return max_score - min_score

    def stability_corner(self, board, corner, color):
        horizontal_direction = 1 if corner[0] == 1 else -1
        vertical_direction = 1 if corner[1] == 1 else -1

        stable_positions = []
        # Percorre a horizontal 
        for i in range(1, 9):
            corner_square = corner[0] + i * horizontal_direction, corner[1]
            
            # Se a posição conter uma peça do jogador, então percorre a vertical
            if board.get_square_color(corner_square[0], corner_square[1]) == color:
                for j in range(1, 9):
                    stable_square = corner_square[0], corner_square[1] + j * vertical_direction
                    # Se a posição conter uma peça do jogador, então a posição é estável
                    if board.get_square_color(stable_square[0], stable_square[1]) == color:
                        stable_positions.append(stable_square)
                    else:
                        break
            else:
                break
        
        return len(stable_positions)
            

        
    def squere_weight(self, board, color):
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
        for i in range(1, 9):
            for j in range(1, 9):
                if board.get_square_color(i, j) == color:
                    max_score += weight[i-1][j-1]
                elif board.get_square_color(i, j) == board._opponent(color):
                    min_score += weight[i-1][j-1]

        return max_score - min_score

    def score(self, board, color):
        max_player = 0
        min_player = 1
        if color == board.BLACK:
            max_player, min_player = min_player, max_player
        
        score = board.score()
        discs = score[0] + score[1]

        coin_weight = 25
        mobility_weight = 5
        corner_weight = 30
        square_weight = 1
        stability_weight = 25
        
        coin_score = coin_weight * self.coin_parity(board, max_player, min_player)
        mobility_score = mobility_weight * self.mobility(board, color)
        corner_score = corner_weight * self.corner_captured(board, color)
        square_weight_score = square_weight * self.squere_weight(board, color)
        stability_score = stability_weight * self.stability(board, color)

        scores = [coin_score, mobility_score, corner_score, square_weight_score, stability_score]
        return coin_score + mobility_score + corner_score + square_weight_score + stability_score, scores 
    