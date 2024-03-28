import numpy as np
import random as rd
from modules.alphazero import NeuralNetworkWrapper


class TicTacToe:
    def __init__(self, player1, player2):
        self.config = (player1, player2)
        self.board = np.zeros((3, 3), dtype=int)
        players_refs = {"0": PlayerHuman, "1": PlayerHasard, "2": PlayerNN}

        self.player1 = players_refs[player1](self)
        self.player2 = players_refs[player2](self)
        self.player1.num = 1
        self.player2.num = -1
        self.player = self.player1

    def change_player(self):
        if self.player == self.player1:
            self.player = self.player2
        else:
            self.player = self.player1

    def clone(self):
        clone_game = TicTacToe(self.config[0], self.config[1])
        clone_game.board = np.copy(self.board)
        clone_game.player = self.player
        return clone_game

    def print_board(self):
        for row in self.board:
            print(" ".join(["X" if case == 1 else "O" if case == -1 else "-" for case in row]))

    def get_possible_move(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def get_valid_move(self):
        valid_moves = []
        possible_actions = self.get_possible_move()
        for x in range(self.board.shape[0]):
            for y in range(self.board.shape[1]):
                if (x, y) in possible_actions:
                    valid_moves.append((1, x, y))
                else:
                    valid_moves.append((0, None, None))
        return np.array(valid_moves)

    def move(self, cell_id=None):
        if cell_id is not None:
            cell_id = int(cell_id)
            cell = ((cell_id - 1) // 3, (cell_id - 1) % 3)
        else:
            cell = self.player.make_move()

        print(cell)
        self.board[cell[0], cell[1]] = self.player.num
        self.print_board()
        self.change_player()
        return int(cell[0] * 3 + cell[1] + 1)

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

    def check_winner(self, player=1):
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):

                return True, 1
            elif np.all(self.board[i, :] == -player) or np.all(self.board[:, i] == -player):

                return True, -1

        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):

            return True, 1
        elif np.all(np.diag(self.board) == -player) or np.all(np.diag(np.fliplr(self.board)) == -player):

            return True, -1

        if np.all(self.board != 0):
            return True, 0

        return False, 0

    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row, col] = self.player.num
            self.change_player()
            return True
        else:
            return False

    def is_valid_move(self, row, col):
        return 3 > row >= 0 == self.board[row, col] and 0 <= col < 3


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
            self.network = NeuralNetworkWrapper(game, model="./models2/")

    def make_move(self):
        state = self.game.board
        move_probs = self.network.play(self.game.clone())
        print(move_probs)
        best_move_index = np.argmax(move_probs)
        print(best_move_index)
        best_move = (best_move_index // 3, best_move_index % 3)
        print(best_move)
        return (best_move[0],best_move[1])
        return best_move_index
