from views.console_board_view import ConsoleBoardView
from models.board import Board
import time
import glob


class BoardController:
    def __init__(self):
        self.board = Board(None)
        self.view = ConsoleBoardView(self.board)

    def init_game(self, players):
        # print(
        #     """\
        #       ___  _   _          _ _
        #      / _ \| |_| |__   ___| | | ___
        #     | | | | __| '_ \ / _ \ | |/ _ \\
        #     | |_| | |_| | | |  __/ | | (_) |
        #      \___/ \__|_| |_|\___|_|_|\___/ ufrj
        #       """
        # )

        self.black_player = self._select_player(Board.BLACK, players[0])
        self.white_player = self._select_player(Board.WHITE, players[1])
        self.atual_player = self.black_player

        self.finish_game = 0
        self.view.update_view()

        while self.finish_game != 2:
            try:
                self._main_loop()
            except KeyboardInterrupt:
                return
        return self._end_game()

    def _main_loop(self):
        atual_color = self.atual_player.color
        name = 'BRANCO' if atual_color == Board.WHITE else 'PRETO'
        #print(f'\nJOGADOR: {name} ({atual_color})')

        start = time.time()
        valid_moves = self.board.valid_moves(atual_color)
        if valid_moves.__len__() > 0:
            player_move = self.atual_player.play(self.board.get_clone())

            while(player_move not in valid_moves):
                #print("JOGADA INVÁLIDA, TENTE NOVAMENTE.\n")
                #print(f'\nJOGADOR: {name} ({atual_color})')
                player_move = self.atual_player.play(self.board.get_clone())

            self.board.play(player_move, atual_color)
            self.view.update_view()
            self.finish_game = 0
        else:
            #print(
            #    f'Sem movimentos para o jogador {name}')
            self.finish_game += 1
        end = round(time.time() - start)

        #print(
        #    f"Tempo de jogada: {end}s\n")
        #time.sleep(0.7)

        self.atual_player = self._opponent(self.atual_player)

    def _end_game(self):
        score = self.board.score()
        if score[0] > score[1]:
            #print("")
            #print('Jogador ' + self.white_player.__class__.__name__ +
             #     ' de BRANCO ' + '('+Board.WHITE+') Ganhou')
            return -1
        elif score[0] < score[1]:
            #print("")
            #print('Jogador ' + self.black_player.__class__.__name__ +
            #      ' de PRETO ' + '('+Board.BLACK+') Ganhou')
            return 1
        else:
            # print("")
            # print('Jogo terminou empatado')
            return 0

    def _opponent(self, player):
        if player.color == Board.WHITE:
            return self.black_player

        return self.white_player

    def _select_player(self, color, player):
        players = glob.glob('Othello/tabuleiro-othello/models/players/*_player.py')
        if color == Board.WHITE:
            name = 'BRANCO'
        else:
            name = 'PRETO'

        while True:
            # print(
            #     f'\nSelecione um dos players abaixo para ser o jogador {name} ({color})')

            # for idx, player in enumerate(players):
            #     print(idx.__str__() + " - " +
            #           player.split('\\')[-1].split('/')[-1].split('_')[0])
            # try:
            #     player = int(
            #         input("\nDigite o numero do player que voce deseja: "))
            # except ValueError:
            #     print("Escolha deve ser um inteiro")
            #     continue
            
            if 0 <= player < len(players):
                break

        #     print("Escolha inválida")
        # print()
        module_globals = {}
        exec(open(players[int(player)]).read(), module_globals)
        return module_globals[list(module_globals.keys())[len(module_globals.keys()) - 1]](color)

