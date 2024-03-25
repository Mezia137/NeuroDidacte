import numpy as np
import random as rd
from modules.alphazero import NeuralNetworkWrapper

class TicTacToe:
    def __init__(self, player1, player2):
        self.board = np.zeros((3, 3), dtype=int)
        players_refs = {"0":PlayerHuman(), "1":PlayerHasard()}
        try:
            self.player1 = players_refs[player1]
            self.player2 = players_refs[player2]
        except:
            print('Erreur selection players')
        self.player = self.player1

    def clone(self):
        clone_game = TicTacToe()
        clone_game.board = np.copy(self.board)
        clone_game.current_player = self.current_player
        return clone_game

    def restart_game(self, P=1):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = P

    def print_board(self):
        for row in self.board:
            print(" ".join(["X" if case == 1 else "O" if case == -1 else "-" for case in row]))

    def get_possible_move(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]
    
    def get_valid_move(self):
        valid_moves = []
        possible_actions = self.get_possible_move()
        for x in range(self.row):
            for y in range(self.column):
                if ((x,y) in possible_actions):
                    valid_moves.append((1, x, y))
                else:
                    valid_moves.append((0, None, None)) 
        return np.array(valid_moves)

    def is_valid_move(self, row, col):
        return 0 <= row < 3 and 0 <= col < 3 and self.board[row, col] == 0

    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row, col] = self.current_player
            self.current_player *= -1
            return True
        else:
            return False

    def check_winner(self, player = 1):
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

    def play(self, player1, player2):
        while True:
            if self.current_player == 1:
                player1.make_move(self)
            else:
                player2.make_move(self)
            print(self.board)
            w,winner  = self.check_winner()
            if w is True:
                
                return winner
        
    def move(self, cell_id=None):
        if cell_id is not None:
            cell = ((cell_id-1)//3, (cell_id-1)%3)
        else:
            cell = self.player.make_move()
            
        self.make_move(cell)
        return cell[0]*3+cell[1]+1

class PlayerHuman:
    def __init__(self, game):
        self.game = game
    
class PlayerHasard:
    def __init__(self, game):
        self.game = game
    
    def make_move(self):
        return rd.choice(self.game.get_possible_move())
    
class PlayerNN:
    def __init__(self, game, level=0):
        self.game = game
        if level == 0:
            self.network = NeuralNetworkWrapper(game)
            
    


    


