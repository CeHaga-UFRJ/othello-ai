# Tabuleiro Othello

## _Campeonato de Inteligências Artificiais - UFRJ_

Código elaborado por [victorlcampos](https://github.com/victorlcampos/TabuleiroOthello) e [rodrigomorgado](https://github.com/rodrigomorgado). O código foi feito em Python 2.x e modificado por mim, para Python 3.x. Além disso, novas funções fonram adicionadas e um novo visual.

## Descrição

Tudo que será descrito está de acordo com as [Regras Oficiais do Jogo](https://www.worldothello.org/about/about-othello/othello-rules/portugues-brasil).

Othello é um jogo de dois jogadores com cada jogador tendo uma cor (normalmente preto e branco), e o objetivo do jogo é terminar com mais peças da sua cor no tabuleiro do que as peças da cor de seu oponente.

O jogo acontece em uma tabuleiro de 8x8 com uma área 2x2 no meio preenchida com peças de cores alternadas. Os jogadores se revezam colocando as peças de suas respectivas cores. Se a colocação da sua peça cria uma linha de peças consecutivas do adversário delimitada por peças de sua cor, no início e no final da linha, todas as peças na linha “viram” para se tornar da sua cor. Note que essa "linha" pode ser feita na vertical, horizontal ou diagonal.

Para que um movimento seja válido, ele deve criar uma das linhas acima mencionadas. Se não houver movimentos válidos, você deve passar sua vez para o seu oponente. O jogo termina quando ambos os jogadores não podem se mover, e o jogador com mais peças de sua cor no tabuleiro vence.

## Representação do tabuleiro

O tabuleiro é representado como uma matriz, que inclui cada placa do tabuleiro, bem como a borda externa. Um tabuleiro inicial contém quatro peças no centro:

```
? ? ? ? ? ? ? ? ? ?
? . . . . . . . . ?
? . . . . . . . . ?
? . . . . . . . . ?
? . . . ● ○ . . . ?
? . . . ○ ● . . . ?
? . . . . . . . . ?
? . . . . . . . . ?
? . . . . . . . . ?
? ? ? ? ? ? ? ? ? ?
```

A borda externa é marcada pelo símbolo "**?**", e as placas vazias são "**.**", o preto é "○" e o branco é "●". As peças pretas e brancas representam os dois jogadores.

```
EMPTY, BLACK, WHITE, OUTER = '.', '○', '●', '?'
```

**OBS.: Dependendo do contraste do shell, o símbolo "●" pode ser confundido com o preto. Uma maneira de evitar confusões, é lembrar que a bolinha vazada é sempre a preta e a preenchida é sempre a branca.**

## Instruções

Execute em qualquer interpretador Python 3.x. Aparecerá os agentes disponiveis para jogar. Escolha primeiro o jogador para ser a peça preta e, em seguida, o jogador para a peça branca.

Você será presenteado com um tabuleiro de 8x8 na tela com um quadrado de 2x2 de peças alternados no meio, conforme descrito acima. Haverá um placar na parte superior do tabuleiro para acompanhar a pontuação (quantidade de peças de cada jogador) e indicar de quem é a vez. Além disso, ao fim de cada jogada será apresentado na parte inferior do tabuleiro um cronometro indicando quanto tempo levou a jogada. Lembrando que o tempo máximo é de `10 segundos`.

Por padrão o jogo possui 3 agentes:

- **coner_player** - Uma estratégia que prioriza os cantos do tabuleiro.
- **human_player** - Jogador humano.
- **random_player** - Uma estratégia que sempre escolhe um movimento aleatório válido.

## Criação do Agente

A criação de um novo agente deve ser feita na pasta **"controllers/models/players/"** e o nome do arquivo deve possuir o sufixo `_player`, ou seja, `exemplo_player.py`. O conteúdo do arquivo deve ser obrigatóriamente um classe, contendo a função `play` que receberá o tabuleiro atual.

```python
class ExemploPlayer:
    def __init__(self, color):
        self.color = color
    def play(self, board):
        return minimax_funcao()
```

## Objetivo

Você deve criar um agente capaz de decidir os melhores movimentos a serem realizados durante as jogadas. A função `play` será responsavel por iniciar a lógica da sua I.A e retornar o melhor movimento a ser feito. Para isso, você deverá implementar obrigatóriamente o algoritmo **Minimax**.

A classe Board (tabuleiro) possui algumas funções úteis que irá te auxiliar na elaboração das haurísticas:

- **play(move, color)** - Retorna o tabuleiro atualizado com o movimento e peça passada.
- **get_square_color(l, c)** - Retorna o símbolo na coordenada passada, ou seja, '.', '○', '●', '?'.
- **get_clone()** - Retorna uma cópia do tabuleiro.
- **valid_moves(color)** - Retorna uma lista de todos os movimentos válidos para o jogador.
- **score()** - Retorna uma lista com a quantidade de peças de cada cor: [branco, preto].
- **\_opponent(color)** - Retorna o oponente da peça passada.
