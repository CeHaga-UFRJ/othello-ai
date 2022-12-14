import random
import numpy as np

class othello:
    """
    Encapsulation of the board game othello
    """

    class ai:
        """
        Class to encapsulate all ai behaviour, more to come.
        To create a new ai, make a child class :)
        """

        def __init__(self, marker):
            self.name = "base_ai"
            self.marker = marker
            self.depth = 0
            self.search_mode = 0

        def peekScore(self, board, x, y):
            dupeBoard = self._duplicateBoard_(board)
            self._makeMove_(dupeBoard, self.marker, x, y)
            score = self.getCurrentScore(dupeBoard)[str(self.marker)]
            return score

        def createNewBoardState(self, board, x, y):
            dupeBoard = self._duplicateBoard_(board)
            self._makeMove_(dupeBoard, self.marker, x, y)
            return dupeBoard

        def createChildBoardState(self, board, x, y, marker):
            dupeBoard = self._duplicateBoard_(board)
            self._makeMove_(dupeBoard, marker, x, y)
            return dupeBoard

        def _makeMove_(self, board, tile, xstart, ystart):

            tilesToFlip = self.isValidMove(board, tile, xstart, ystart)
            if tilesToFlip == False:
                return False

            board[xstart, ystart] = tile
            for x, y in tilesToFlip:
                board[x, y] = tile
            return True

        def _duplicateBoard_(self, board):
            b = np.copy(board)
            return b

        def isOnCorner(self, x, y):
            # Returns True if the position is in one of the four corners.
            return (
                (x == 1 and y == 1)
                or (x == 8 and y == 1)
                or (x == 1 and y == 8)
                or (x == 8 and y == 8)
            )

        def getCurrentScore(self, board):
            # Determine the score by counting the tiles. Returns a dictionary with keys 'X' and 'O'.
            xscore = 0
            oscore = 0
            for x in range(8):
                for y in range(8):
                    if board[x, y] == "1":
                        xscore += 1

                    if board[x, y] == "-1":
                        oscore += 1

            return {"1": xscore, "-1": oscore}

        def _isOnBoard_(self, x, y):
            return x >= 0 and x <= 7 and y >= 0 and y <= 7

        def isValidMove(self, board, tile, xstart, ystart):
            # Returns False if the player's move on space xstart, ystart is invalid.
            # If it is a valid move, returns a list of spaces that would become the player's if they made a move here.
            if board[xstart, ystart] != 0 or not self._isOnBoard_(xstart, ystart):
                return False

            board[xstart, ystart] = tile  # temporarily set the tile on the board.
            if tile == 1:
                otherTile = -1
            else:
                otherTile = 1
            tilesToFlip = []
            for xdirection, ydirection in [
                [0, 1],
                [1, 1],
                [1, 0],
                [1, -1],
                [0, -1],
                [-1, -1],
                [-1, 0],
                [-1, 1],
            ]:
                x, y = xstart, ystart
                x += xdirection  # first step in the direction
                y += ydirection  # first step in the direction
                if self._isOnBoard_(x, y) and board[x, y] == otherTile:
                    # There is a piece belonging to the other player next to our piece.
                    x += xdirection
                    y += ydirection
                    if not self._isOnBoard_(x, y):
                        continue

                    while board[x, y] == otherTile:
                        x += xdirection
                        y += ydirection

                        if not self._isOnBoard_(
                            x, y
                        ):  # break out of while loop, then continue in for loop
                            break

                    if not self._isOnBoard_(x, y):
                        continue

                    if board[x, y] == tile:
                        # There are pieces to flip over. Go in the reverse direction until we reach the original space, noting all the tiles along the way.
                        while True:
                            x -= xdirection
                            y -= ydirection
                            if x == xstart and y == ystart:
                                break
                            tilesToFlip.append([x, y])

            board[xstart, ystart] = 0  # restore the empty space
            if (
                len(tilesToFlip) == 0
            ):  # If no tiles were flipped, this is not a valid move.
                return False
            return tilesToFlip

        def getLegalMoves(self, board, tile):
            # Returns a list of [x,y] lists of valid moves for the given player on the given board.
            validMoves = []
            for x in range(8):
                for y in range(8):
                    if self.isValidMove(board, tile, x, y) != False:
                        validMoves.append([x, y])
            return validMoves

        def getMove(self, board):
            print("WARNING: this is the base ai class, use a custom class please.")
            return

    def __init__(self, bot1=ai("X"), bot2=ai("O"), verbose=True):
        self.mainBoard = self.getNewBoard()
        self.resetBoard(self.mainBoard)
        self.verbose = verbose
        self.bot1 = bot1
        self.bot2 = bot2
        if not self.checkBotLegality(self.bot1, self.bot2):
            print("Bots not compatiable: Fatal error")
            return

    def startgame(self, start_move=0):
        # Welcome message for bots, comment out lines if you want to surpress terminal outputs
        if self.verbose:
            self.welcomeMessage(self.bot1)
            self.welcomeMessage(self.bot2)

        # Randomly initialise board (not implemented)
        self.mainBoard = self.createRandomBoard(start_move)

        # TODO: reimplement random player start
        random_num = random.randint(0, 1)
        if random_num == 0:
            while True:
                turn_state = self.takeTurn(self.bot1, self.bot2, verbose=self.verbose)
                if turn_state == 1:
                    return
                elif turn_state == -1:
                    break

                turn_state = self.takeTurn(self.bot2, self.bot1, verbose=self.verbose)
                if turn_state == 1:
                    return
                elif turn_state == -1:
                    break
        else:
            while True:
                turn_state = self.takeTurn(self.bot2, self.bot1, verbose=self.verbose)
                if turn_state == 1:
                    return
                elif turn_state == -1:
                    break

                turn_state = self.takeTurn(self.bot1, self.bot2, verbose=self.verbose)
                if turn_state == 1:
                    return
                elif turn_state == -1:
                    break

        # Game finished, show results
        if self.verbose:
            self.displayResults(self.bot1)
        return self.getScoreOfBoard(self.mainBoard)

    def createRandomBoard(self, turns_in):
        while True:
            B = self.getNewBoard()
            self.resetBoard(B)

            for i in range(turns_in):
                legal_moves = self.getValidMoves(B, 1)
                if legal_moves == []:
                    continue
                move = random.choice(legal_moves)
                self.makeMove(B, 1, move[0], move[1])

                legal_moves = self.getValidMoves(B, -1)
                if legal_moves == []:
                    continue
                move = random.choice(legal_moves)
                self.makeMove(B, -1, move[0], move[1])

            # Check to see if any moves can stil be made on board
            if self.getValidMoves(B, 1) == []:
                continue
            if self.getValidMoves(B, -1) == []:
                continue
            return B

    def takeTurn(self, bot, bot_other, verbose=True):
        if verbose:
            self.drawBoard(self.mainBoard)
            self.showPoints(self.mainBoard)

        move = bot.getMove(self.mainBoard)

        if move == "quit":
            return 1
        else:
            self.makeMove(self.mainBoard, bot.marker, move[0], move[1])

        if self.getValidMoves(self.mainBoard, bot_other.marker) == []:
            return -1
        else:
            return 0

    def checkBotLegality(self, bot1, bot2):
        if bot1.marker == self.bot2.marker:
            return False
        return True

    def welcomeMessage(self, bot):
        if bot.name == "base_ai":
            print("Welcome base ai I am treating you like a human.")
        elif bot.name == "human":
            print("Welcome human.")
        else:
            print("I only know how to deal with humans at the moment, sorry.")
            return

    def displayResults(self, bot):
        self.drawBoard(self.mainBoard)
        scores = self.getScoreOfBoard(self.mainBoard)
        #print("X scored %s points. O scored %s points." % (scores["1"], scores["-1"]))
        print(self.bot1.name,' score: ', str(scores[str(self.bot1.marker)]))
        print(self.bot2.name,' score: ', str(scores[str(self.bot2.marker)]))
        # if scores[str(bot.marker)] > scores[str(self.bot2.marker)]:
        #     print(
        #         "You beat the computer by %s points! Congratulations!"
        #         % (scores[str(bot.marker)] - scores[str(self.bot2.marker)])
        #     )
        # elif scores[str(bot.marker)] < scores[str(self.bot2.marker)]:
        #     print(
        #         "You lost. The computer beat you by %s points."
        #         % (scores[str(self.bot2.marker)] - scores[str(bot.marker)])
        #     )
        # else:
        #     print("The game was a tie!")

    def drawBoard(self, board):
        # This function prints out the board that it was passed. Returns None.
        HLINE = "  +---+---+---+---+---+---+---+---+"
        VLINE = "  |   |   |   |   |   |   |   |   |"
        print("    1   2   3   4   5   6   7   8")
        print(HLINE)

        for y in range(8):
            print(VLINE)
            print(y + 1, end=" ")
            for x in range(8):
                if board[x, y] == 0:
                    print("|  ", end=" ")
                else:
                    print("| %d" % (board[x, y] + 2), end=" ")
            print("|")
            print(VLINE)
            print(HLINE)

    def resetBoard(self, board):
        # Blanks out the board it is passed, except for the original starting position.
        for x in range(8):
            for y in range(8):
                board[x, y] = 0
                # Starting pieces:
                board[3, 3] = 1
                board[3, 4] = -1
                board[4, 3] = -1
                board[4, 4] = 1

    def getNewBoard(self):
        # Creates a brand new, blank board data structure.
        board = np.zeros((8, 8),dtype=int)
        # board = []
        # for i in range(8):
        #     board.append([" "] * 8)
        return board

    def isValidMove(self, board, tile, xstart, ystart):
        # Returns False if the player's move on space xstart, ystart is invalid.
        # If it is a valid move, returns a list of spaces that would become the player's if they made a move here.
        if board[xstart, ystart] != 0 or not self.isOnBoard(xstart, ystart):
            return False

        board[xstart, ystart] = tile  # temporarily set the tile on the board.
        if tile == 1:
            otherTile = -1
        else:
            otherTile = 1
        tilesToFlip = []
        for xdirection, ydirection in [
            [0, 1],
            [1, 1],
            [1, 0],
            [1, -1],
            [0, -1],
            [-1, -1],
            [-1, 0],
            [-1, 1],
        ]:
            x, y = xstart, ystart
            x += xdirection  # first step in the direction
            y += ydirection  # first step in the direction
            if self.isOnBoard(x, y) and board[x, y] == otherTile:
                # There is a piece belonging to the other player next to our piece.
                x += xdirection
                y += ydirection
                if not self.isOnBoard(x, y):
                    continue

                while board[x, y] == otherTile:
                    x += xdirection
                    y += ydirection

                    if not self.isOnBoard(
                        x, y
                    ):  # break out of while loop, then continue in for loop
                        break

                if not self.isOnBoard(x, y):
                    continue

                if board[x, y] == tile:
                    # There are pieces to flip over. Go in the reverse direction until we reach the original space, noting all the tiles along the way.
                    while True:
                        x -= xdirection
                        y -= ydirection
                        if x == xstart and y == ystart:
                            break
                        tilesToFlip.append([x, y])

        board[xstart, ystart] = 0  # restore the empty space
        if len(tilesToFlip) == 0:  # If no tiles were flipped, this is not a valid move.
            return False
        return tilesToFlip

    def isOnBoard(self, x, y):
        # Returns True if the coordinates are located on the board.
        return x >= 0 and x <= 7 and y >= 0 and y <= 7

    def getValidMoves(self, board, tile):
        # Returns a list of [x,y] lists of valid moves for the given player on the given board.
        validMoves = []
        for x in range(8):
            for y in range(8):
                if self.isValidMove(board, tile, x, y) != False:
                    validMoves.append([x, y])
        return validMoves

    def getScoreOfBoard(self, board):
        # Determine the score by counting the tiles. Returns a dictionary with keys 'X' and 'O'.
        xscore = 0
        oscore = 0
        for x in range(8):
            for y in range(8):
                if board[x, y] == 1:
                    xscore += 1

                if board[x, y] == -1:
                    oscore += 1

        return {"1": xscore, "-1": oscore}

    def makeMove(self, board, tile, xstart, ystart):
        # Place the tile on the board at xstart, ystart, and flip any of the opponent's pieces.
        # Returns False if this is an invalid move, True if it is valid.
        tilesToFlip = self.isValidMove(board, tile, xstart, ystart)
        if tilesToFlip == False:
            return False

        board[xstart, ystart] = tile
        for x, y in tilesToFlip:
            board[x, y] = tile
        return True

    def duplicateBoard(self, board):
        b = np.copy(board)
        return b

    def isOnCorner(self, x, y):
        # Returns True if the position is in one of the four corners.
        return (
            (x == 1 and y == 1)
            or (x == 8 and y == 1)
            or (x == 1 and y == 8)
            or (x == 8 and y == 8)
        )

    def getComputerMove(self, board, computerTile):
        # Given a board and the computer's tile, determine where to
        # move and return that move as a [x, y] list.
        possibleMoves = self.getValidMoves(board, computerTile)
        # randomize the order of the possible moves
        random.shuffle(possibleMoves)
        # always go for a corner if available.
        for x, y in possibleMoves:
            if self.isOnCorner(x, y):
                return [x, y]

        # Go through all the possible moves and remember the best scoring move
        bestScore = -1
        for x, y in possibleMoves:
            dupeBoard = self.duplicateBoard(board)
            self.makeMove(dupeBoard, computerTile, x, y)
            score = self.getScoreOfBoard(dupeBoard)[str(computerTile)]
            if score > bestScore:
                bestMove = [x, y]
            bestScore = score
        return bestMove

    def showPoints(self, mainBoard):
        # Prints out the current score.
        scores = self.getScoreOfBoard(mainBoard)
        # print(
        #     "You have %d points. The computer has %d points."
        #     % (scores[str(self.bot1.marker)], scores[str(self.bot2.marker)])
        # )
        print(self.bot1.name,' score: ', str(scores[str(self.bot1.marker)]))
        print(self.bot2.name,' score: ', str(scores[str(self.bot2.marker)]))

class decisionRule_ai(othello.ai):
        def __init__(self, marker):
            self.name = "Greedy"
            self.marker = marker
            self.search_mode = 'Greedy'
            self.depth = 0

        def score(self, board, color):
            if color == board.WHITE:
                return board.score()[0]
            else:
                return board.score()[1]

        def getMove(self, board, color):
            possibleMoves = board.valid_moves(color)
            # randomize possible moves
            random.shuffle(possibleMoves)

            # Corners are opimial always
            for move in possibleMoves:
                if self.isOnCorner(move.x, move.y):
                    return move

            # Go through all the possible moves and remember the best scoring move
            bestScore = -1
            bestMove = None
            for move in possibleMoves:
                copy_board = board.get_clone()
                copy_board.play(move, color)
                score = self.score(copy_board, color)
                if score > bestScore:
                    bestMove = move
                    bestScore = score
            return bestMove