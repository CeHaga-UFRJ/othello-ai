from models.move import Move


class HumanPlayer:
    def __init__(self, color):
        self.color = color

    def play(self, board):
        movimentos = board.valid_moves(self.color)
        if movimentos:
            print(board)
            print('Movimentos possiveis [linha, coluna]: ', end="")
            for i in movimentos:
                print(f"[{i}] ", end="")
            print()
        while(True):
            try:
                rowInp = int(input("Linha: "))
                colInp = int(input("Coluna: "))
                break
            except ValueError:
                continue

        print()
        return Move(rowInp, colInp)
