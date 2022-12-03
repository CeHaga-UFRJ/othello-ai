from players import *

if __name__ == "__main__":
    
    n_games = 5000
    wins = 0
    opponent = RandomPlayer()
    kindofplayer = "l" # Change the kind of player that will be used {"h": human player, "r": random player, "c": corner player, "g": greedy player, "l": reinforcement learning player}
    visible = False

    if kindofplayer == "h":
        player = HumanPlayer()
        visible = True
        n_games = 1
    elif kindofplayer == "c":
        player = CornerPlayer()
    elif kindofplayer == "r":
        player = RandomPlayer()
    elif kindofplayer == "g":
        player = GreedyPlayer()
    elif kindofplayer == "l":
        player = RLPlayer(0.1, 1)
        player.policy_net.load("weights-10000-games")
        player.epsilon = 0.0
    else:
        player = RandomPlayer()

    # Runs 'n_games' games and print the scores for each one
    for i in range(n_games):
        mainBoard = getNewBoard()
        resetBoard(mainBoard)
        playerTile, computerTile = ["X", "O"]
        turn = whoGoesFirst()       
        can_play = True
            
        while can_play:
            if turn == "player":
                if getValidMoves(mainBoard, "X"):
                    movex, movey = player.getMove(mainBoard, "X")
                    makeMove(mainBoard, "X", movex, movey)
                else:
                    can_play = False if not getValidMoves(mainBoard, "O") else True
            else:
                if getValidMoves(mainBoard, "O"):
                    movex, movey = opponent.getMove(mainBoard, "O")
                    makeMove(mainBoard, "O", movex, movey)
                else:
                    can_play = False if not getValidMoves(mainBoard, "X") else True

            turn = "player" if turn == "computer" else "computer"

        # Display the final score.
        scores = getScoreOfBoard(mainBoard)
        print(f'Game: {i+1} => Player scored {scores["X"]} points. Opponent scored {scores["O"]} points.')
        if( scores["X"] > scores["O"]):
            wins += 1

    print( f"Player won {wins} out of {n_games} games. ({wins/n_games*100:.2f}%)" )