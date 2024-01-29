
import numpy as np
import random as rd
from pprint import pprint

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 for X, -1 for O
    def get_board(self):
        l = []
        for row in self.board:
            l.append([case for case in row])
        return [l[i][k]for i in range(3)for k in range(3)]
    def print_board(self):
        for row in self.board:
            print(" ".join(["X" if case == 1 else "O" if case == -1 else "-" for case in row]))

    def is_valid_move(self, row, col):
        return self.board[row, col] == 0

    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row, col] = self.current_player
            self.current_player *= -1  
            return True
        else:
            print("Invalid move. Try again.")
            return False

    def check_winner(self):
        for i in range(3):

            if np.all(self.board[i, :] == 1) or np.all(self.board[:, i] == 1):
                return 1
            elif np.all(self.board[i, :] == -1) or np.all(self.board[:, i] == -1):
                return -1 


        if np.all(np.diag(self.board) == 1) or np.all(np.diag(np.fliplr(self.board)) == 1):
            return 1  

        if np.all(np.diag(self.board) == -1) or np.all(np.diag(np.fliplr(self.board)) == -1):
            return -1 
        


        if np.all(self.board != 0):
            return 0 

        return None 

    def play(self, player1, player2):
        while True:
            self.print_board()
            if self.current_player == 1:
                print("Player X's turn.")
                player1.make_move(self)
            else:
                print("Player O's turn.")
                player2.make_move(self)

            winner = self.check_winner()
            if winner is not None:
                self.print_board()
                if winner == 1:
                    print("Player X wins!")
                elif winner == -1:
                    print("Player O wins!")
                else:
                    print("It's a tie!")
                break
    def get_3matrice(self):
        M1 = np.zeros((3, 3), dtype=int)
        M2 = np.zeros((3, 3), dtype=int)
        M3 = np.zeros((3, 3), dtype=int)
        for i,j in enumerate(self.board) :
            for h ,k in enumerate(j) :
                if k == self.current_player :
                    M1[i,h] = 1
                elif k == self.current_player*(-1) :
                    M2[i,h] = 1
                else :
                    M3[i,h] = 1
        return M1 , M2 , M3
    
class HumanPlayer:
    def make_move(self, game):
        while True:
            try:
                row = int(input("Enter row (0, 1, or 2): "))
                col = int(input("Enter column (0, 1, or 2): "))
                if game.is_valid_move(row, col):
                    game.make_move(row, col)
                    break
                else:
                    print("Invalid move. Try again.")
            except ValueError:
                print("Invalid input. Enter numbers only.")

                
class Hasard:
    def make_move(self, game):
        while True:
            nb = rd.randrange(1, 10)
            row = nb // 3
            col = nb % 3
            if game.is_valid_move(row, col):
                game.make_move(row, col)
                break

class NNplayer:
    def __init__(self, reseau):
        self.NN = reseau
    def make_move(self, game):
        while True:
            nb = self.NN.predict(game.get_3matrice)
            row = nb // 3
            col = nb % 3
            if game.is_valid_move(row, col):
                game.make_move(row, col)
                break


if __name__ == "__main__":

    game = TicTacToe()
    player1 = NNplayer()
    player2 = Hasard()

    game.play(player1, player2)