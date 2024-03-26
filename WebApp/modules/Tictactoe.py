import numpy as np
import random as rd
from modules.alphazero import NeuralNetworkWrapper


class TicTacToe:
    def __init__(self, player1, player2):
        self.board = np.zeros((3, 3), dtype=int)
        players_refs = {"0": PlayerHuman, "1": PlayerHasard}

        self.player1 = players_refs[player1](self)
        self.player2 = players_refs[player2](self)
        self.player1.num = 1
        self.player2.num = -1
        self.player = self.player1

    def print_board(self):
        for row in self.board:
            print(" ".join(["X" if case == 1 else "O" if case == -1 else "-" for case in row]))

    def get_possible_move(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def move(self, cell_id=None):
        if cell_id is not None:
            cell_id = int(cell_id)
            cell = ((cell_id - 1) // 3, (cell_id - 1) % 3)
        else:
            cell = self.player.make_move()

        self.board[cell[0], cell[1]] = self.player.num
        self.print_board()
        if self.player == self.player1:
            self.player = self.player2
        else:
            self.player = self.player1
        return cell[0] * 3 + cell[1] + 1

    def is_winner(self):
        for i in range(3):
            if np.sum(self.board[i, :]) in [3, -3]:
                return [i * 3 + j + 1 for j in range(3)]

            elif np.sum(self.board[:, i]) in [3, -3]:
                return [i + j * 3 + 1 for j in range(3)]

            elif np.trace(self.board) in [3, -3]:
                return [1, 5, 9]

            elif np.trace(self.board[::-1]) in [3, -3]:
                return [3, 5, 7]


class PlayerHuman:
    def __init__(self, game):
        self.game = game
        self.num = None


class PlayerHasard:
    def __init__(self, game):
        self.game = game
        self.num = None

    def make_move(self):
        return rd.choice(self.game.get_possible_move())


class PlayerNN:
    def __init__(self, game, level=0):
        self.game = game
        if level == 0:
            self.network = NeuralNetworkWrapper(game)
