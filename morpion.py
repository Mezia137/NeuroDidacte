import numpy as np
import random as rd
from alphazero import NeuralNetworkWrapper
import os
from alphazero import CFG
from alphazero import Train
class TicTacToe:
    def __init__(self, P=1):
        self.board = np.zeros((3, 3), dtype=int)
        self.row = 3
        self.column = 3
        self.action_size = 9
        self.current_player = P

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
            random_move = rd.choice(game.get_possible_move())
            if game.is_valid_move(*random_move):
                game.make_move(*random_move)
                break

class NNplayer:
    def __init__(self, game):
        self.NN = NeuralNetworkWrapper(game)

    def make_move(self, game):
        state = game.board
        move_probs, _ = self.NN.predict(state)
        h = 0
        for i in range(len(state)):
            for j in range(len(state[i])):
                if state[i, j] != 0:
                    move_probs = np.delete(move_probs, i+j+2-h)
                    h += 1
                    
        valid_moves = game.get_possible_move()
        best_move_index = np.argmax(move_probs)
        best_move = valid_moves[best_move_index]
        game.make_move(best_move[0],best_move[1])

def train() :
    game = TicTacToe()
    net = NeuralNetworkWrapper(game)

    # Initialize the network with the best model.
    if CFG.load_model:
        file_path = CFG.model_directory + "best_model.meta"
        if os.path.exists(file_path):
            net.load_model("best_model")
        else:
            print("Trained model doesn't exist. Starting from scratch.")
    else:
        print("Trained model not loaded. Starting from scratch.")

    # Play vs the AI as a human instead of training.
    train = Train(game, net)
    train.start()
def main():
    game = TicTacToe(P=1)
    player1 = NNplayer(game)
    player2 = HumanPlayer()
    v1 = 0
    v2 = 0
    for i in range (100):
        wineur =  game.play(player1, player2)
        if wineur == 1:
            v1 += 1
        elif wineur == -1:
            v2 += 1
        game.restart_game()
        print(i+1) 
    
    print (v1, v2)
if __name__ == "__main__":
    train()
    #main()
    


    


