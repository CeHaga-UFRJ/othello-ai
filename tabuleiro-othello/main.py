from controllers.board_controller import BoardController
from models.move import Move
from models.board import Board

players = [
  'CNN_DEMO_player',
  'CNN_player',
  'corner_player',
  'QL_player',
  'random_player',
  'wise_player',
]

mode = int(input("Qual modo?\n1 - Apenas 1 jogo\n2 - Todos os jogos\n"))

if mode == 1:
  controller = BoardController()
  print('Opções de jogadores:')
  for i, player in iter(enumerate(players)):
    print(f'{i} - {player}')
    
  player1 = int(input('Entre com o jogador 1: '))
  player2 = int(input('Entre com o jogador 2: '))
  winner = controller.init_game((player1, player2))
  if(winner == 1):
    print(f'Jogador 1 ({players[player1]}) ganhou!')
  elif(winner == -1):
    print(f'Jogador 2 ({players[player2]}) ganhou!')
  else:
    print('Empate!')
  
else:
  

  games = [(i1,i2) for i1 in range(len(players)) for i2 in range(len(players)) if i1 != i2]

  qtd_games = 100

  with open('results.txt', 'w') as file:
    print('Quantidade de jogos: ' + str(qtd_games))
    file.write('Quantidade de jogos: ' + str(qtd_games) + '\n')
    print('')
    file.write('\n')
    for game in games:
      print(f'JOGO: {players[game[0]]} x {players[game[1]]}')
      file.write(f'JOGO: {players[game[0]]} x {players[game[1]]}\n')
      scores = {1: 0, -1: 0, 0: 0}
      for i in range(qtd_games):
        controller = BoardController()
        winner = controller.init_game(game)
        scores[winner] += 1
      print(players[game[0]], 'x', players[game[1]], '-> {', scores[1],',',scores[-1], '}')
      file.write(players[game[0]] + ' x ' + players[game[1]] + ' -> {' + str(scores[1]) + ',' + str(scores[-1]) + '}\n')
      print('')
      file.write('\n')