
import numpy as np
import random as rd

class Tictactoe:
    symboles = {0:"·", -1:"0", 1:"X"}
    def __init__(self, randmod=False):
        self.board = np.zeros((3, 3))
        self.free

    def randmove(self):
        return rd.choice(np.where(plateau_de_jeu == 0))

    def play(self):


    def __str__(self):
        return "".join([[" "+symboles[square] for square in row] for row in board])