
import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 for X, -1 for O

    def print_board(self):
        for row in self.board:
            print(" ".join(["X" if cell == 1 else "O" if cell == -1 else "-" for cell in row]))

    def is_valid_move(self, row, col):
        return self.board[row, col] == 0

    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row, col] = self.current_player
            self.current_player *= -1  # Switch player
            return True
        else:
            print("Invalid move. Try again.")
            return False

    def check_winner(self):
        # Check rows, columns, and diagonals for a winner
        for i in range(3):
            if np.all(self.board[i, :] == 1) or np.all(self.board[:, i] == 1):
                return 1  # Player X wins
            elif np.all(self.board[i, :] == -1) or np.all(self.board[:, i] == -1):
                return -1  # Player O wins

        if np.all(np.diag(self.board) == 1) or np.all(np.diag(np.fliplr(self.board)) == 1):
            return 1  # Player X wins

        if np.all(np.diag(self.board) == -1) or np.all(np.diag(np.fliplr(self.board)) == -1):
            return -1  # Player O wins

        # Check for a tie
        if np.all(self.board != 0):
            return 0  # It's a tie

        return None  # No winner yet

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

if __name__ == "__main__":
    game = TicTacToe()
    player1 = HumanPlayer()
    player2 = HumanPlayer()

    game.play(player1, player2)
