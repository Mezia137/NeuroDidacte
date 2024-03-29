import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

import seaborn as sns
import numpy as np
import random as rd
import os

LIGHT_COLOR = to_rgba('#BFBFBF')


class ReseauSimple:
    def __init__(self, archi=None, xmod=False, init_met="default", seed=42, idi=None):
        if archi is None:
            archi = [2, 3, 1]
        np.random.seed(seed)
        rd.seed(seed)
        if xmod:
            self.Xtrash = np.random.randn(archi[0], 1)

        if init_met == "he":
            self.W_1 = np.random.randn(archi[1], archi[0]) * np.sqrt(2.0 / archi[0])
            self.W_2 = np.random.randn(archi[1], archi[2]) * np.sqrt(2.0 / archi[1])
            self.b_1 = np.random.randn(archi[1], archi[2]) * np.sqrt(2.0 / archi[1])
            self.b_2 = rd.normalvariate(mu=0, sigma=1)
        elif init_met == "xavier":
            self.W_1 = np.random.randn(archi[1], archi[0]) * np.sqrt(1.0 / archi[0])
            self.W_2 = np.random.randn(archi[1], archi[2]) * np.sqrt(1.0 / archi[1])
            self.b_1 = np.random.randn(archi[1], archi[2]) * np.sqrt(1.0 / archi[1])
            self.b_2 = rd.normalvariate(mu=0, sigma=1)
        elif init_met == "default":
            self.W_1 = np.random.randn(archi[1], archi[0])
            self.W_2 = np.random.randn(archi[1], archi[2])
            self.b_1 = np.random.randn(archi[1], archi[2])
            self.b_2 = rd.normalvariate(mu=0, sigma=1)
        else:
            raise ValueError("Invalid weight_init option. Choose 'default', 'he' or 'xavier'.")

        self.stock = [[np.copy(self.W_1), np.copy(self.W_2), np.copy(self.b_1), np.copy(self.b_2)]]
        self.life = 0
        self.age = 0

        if id is not None:
            self.idi = idi

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def ReLU(self, z):
        return np.maximum(0, z)

    def hidden(self, X, W_1, b):
        return self.sigmoid(W_1 @ X.reshape(-1, 1) + b)

    def output(self, H, W_2, b):
        return self.sigmoid(W_2.T @ H + b)

    def delta_2(self, o, y):
        return float((o - y) * o * (1 - o))

    def delta_1(self, W_2, h, d2):
        return (W_2 * d2 * h * (1 - h)).reshape(-1, 1)

    def dJdW_2(self, h, d2):
        return d2 * h

    def dJdW_1(self, X, d1):
        return d1 @ X.T

    def train(self, X, y, passages=1, lr=0.01, show_loading_bar=False):
        for _ in range(passages):
            self.life += 1
            self.age = self.life
            if show_loading_bar:
                n = round(100 * (self.life + 1) / passages)
                print(f'{n * "▣"}{(100 - n) * "▢"}', end='\r')
            for i in range(X.shape[0]):
                h = self.hidden(X[i], self.W_1, self.b_1)
                o = self.output(h, self.W_2, self.b_2)

                d2 = self.delta_2(o, y[i])
                djdb_2 = d2
                d1 = self.delta_1(self.W_2, h, d2)
                djdb_1 = d1
                dj2 = self.dJdW_2(h, d2)
                dj1 = self.dJdW_1(X[i].reshape(-1, 1), d1)

                self.W_2 -= lr * dj2
                self.W_1 -= lr * dj1
                self.b_2 -= lr * djdb_2
                self.b_1 -= lr * djdb_1

            self.stock.append([np.copy(self.W_1), np.copy(self.W_2), np.copy(self.b_1), np.copy(self.b_2)])

    def predict(self, X, age=-1):
        h = self.hidden(X.reshape(-1, 1), self.stock[age][0], self.stock[age][2])
        o = self.output(h, self.stock[age][1], self.stock[age][3])
        return float(o)

    def predict_list(self, X, age=-1):
        y_hat = []
        for x in X:
            y_hat.append(self.predict(x, age=age))
        return y_hat

    def visualisation(self, X, y, savemod=False, folder="Saves", img_name=None, age=-1, nb_levels=10):
        plt.figure()

        x_r = X[:, 0].max() - X[:, 0].min()
        y_r = X[:, 1].max() - X[:, 1].min()
        x_min, x_max = X[:, 0].min() - x_r / 10, X[:, 0].max() + x_r / 10
        y_min, y_max = X[:, 1].min() - y_r / 10, X[:, 1].max() + y_r / 10
        xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / 100),
                             np.arange(y_min, y_max, (y_max - y_min) / 100))
        Z = np.array(self.predict_list(np.c_[xx.ravel(), yy.ravel()], age=age))
        Z = Z.reshape(xx.shape)

        contour = plt.contourf(xx, yy, Z, alpha=1, levels=[i / nb_levels for i in range(nb_levels + 1)],
                               cmap='plasma')

        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y).set_aspect('equal')

        plt.colorbar(contour)
        if savemod:
            if img_name is None:
                img_name = f'fig{age:03d}.svg'
            img_path = os.path.join(folder, img_name)
            plt.savefig(img_path, format='svg', transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()
            return img_name
        else:
            plt.show()

    def history(self):
        return self.stock

    def get_w(self, age=-1):
        w = self.stock[age]
        return {'w000': w[0][0, 0].item(), 'w001': w[0][0, 1].item(), 'w010': w[0][1, 0].item(),
                'w011': w[0][1, 1].item(), 'w020': w[0][2, 0].item(), 'w021': w[0][2, 1].item(),
                'w10': w[1][0].item(), 'w11': w[1][1].item(), 'w12': w[1][2].item()}
