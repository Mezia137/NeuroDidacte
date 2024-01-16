import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import seaborn as sns


# FORWARD PROPAGATION
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def hidden(X, W_1, b):
    # W_1[1]=-W_1[1]
    return sigmoid(W_1 @ X + b)


def output(H, W_2, b):
    # W_2[1]=-W_2[1]
    return sigmoid(W_2.T @ H + b)


# BACKWARD PROPAGATION
def delta_2(o, y):
    return float((o - y) * o * (1 - o))


def delta_1(W_2, h, d2):
    d1 = np.zeros(shape=(3, 1))
    for j in range(d1.shape[0]):
        d1[j] = W_2[j] * d2 * h[j] * (1 - h[j])

    return d1


def dJdW_2(h, d2):
    dj = np.zeros(shape=(3, 1))
    for j in range(dj.shape[0]):
        dj[j] = d2 * h[j]

    return dj


def dJdW_1(X, d1):
    dj = np.zeros(shape=(3, 2))
    for j in range(dj.shape[0]):
        for i in range(dj.shape[1]):
            dj[j, i] = d1[j] * X[i]

    return dj


# TRAINING
def train_model(X, W_1, W_2, b_1, b_2, y, lr):
    for i in range(X.shape[0]):
        # forward propagation
        h = hidden(X[i].reshape(-1, 1), W_1, b_1)
        o = output(h, W_2, b_2)

        # backward propagation
        d2 = delta_2(o, y[i])
        djdb_2 = d2
        d1 = delta_1(W_2, h, d2)
        djdb_1 = d1
        dj2 = dJdW_2(h, d2)
        dj1 = dJdW_1(X[i].reshape(-1, 1), d1)
        W_2 -= lr * dj2
        W_1 -= lr * dj1
        b_2 -= lr * djdb_2
        b_1 -= lr * djdb_1

    return W_1, W_2, b_1, b_2


def cluster(pos, size=100, etendue=(2, 2)):
    cluster = []
    for n in range(1, size + 1):
        x = rd.gauss(pos[0], etendue[0])
        y = rd.gauss(pos[1], etendue[1])
        cluster.append((x, y))

    return cluster


# GENERATING
def predict_model(X, W_1, W_2, b_1, b_2):
    y_hat = []
    for i in range(X.shape[0]):
        h = hidden(X[i].reshape(-1, 1), W_1, b_1)
        o = output(h, W_2, b_2)
        y_hat.append(float(o))

    return y_hat


def accuracy(y_hat, y):
    return len([1 for i in range(y_hat.shape[0]) if round(y_hat[i]) == round(y[i])]) / len(y)


def visualisation(data, labels, W_1, W_2, b_1, b_2, savemod=False, passage=0, nb_levels=10):
    plt.figure(figsize=(12, 8))
    hh = .02
    x_r = data[:, 0].max() - data[:, 0].min()
    y_r = data[:, 1].max() - data[:, 1].min()
    x_min, x_max = data[:, 0].min() - x_r / 10, data[:, 0].max() + x_r / 10
    y_min, y_max = data[:, 1].min() - y_r / 10, data[:, 1].max() + y_r / 10
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / 100),
                         np.arange(y_min, y_max, (y_max - y_min) / 100))

    Z = np.array(predict_model(np.c_[xx.ravel(), yy.ravel()], W_1, W_2, b_1, b_2))
    Z = Z.reshape(xx.shape)

    contour = plt.contourf(xx, yy, Z, alpha=1, levels=[i / nb_levels for i in range(nb_levels + 1)])
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels).set_aspect('equal')

    plt.colorbar(contour, label='prediction')
    if savemod:
        plt.savefig(f'./Saves/fig{passage:03d}.png')
    else:
        plt.show()


# DEMONSTRATION

np.random.seed(42)
X = np.random.standard_normal(size=(2, 1))
W_1 = np.random.standard_normal(size=(3, 2))
W_2 = np.random.standard_normal(size=(3, 1))
b_1 = np.random.standard_normal(size=(3, 1))
b_2 = rd.normalvariate(mu=0, sigma=1)
y = rd.normalvariate(mu=0, sigma=1)

data = np.array(cluster((20, 20), size=1000) + cluster((-20, -20), size=1000))
labels = [1 for _ in range(1000)] + [0 for _ in range(1000)]

lr = 0.001
passages = 300
for i in range(passages):
    W_1, W_2, b_1, b_2 = train_model(data, W_1, W_2, b_1, b_2, labels, lr)

visualisation(data, labels, W_1, W_2, b_1, b_2, nb_levels=10)
