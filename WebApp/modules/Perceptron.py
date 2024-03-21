import numpy as np
import os
import random as rd
import matplotlib.pyplot as plt
import seaborn as sns





# Classe du Perceptron
class Perceptron:

    # Initialisation avec des paramètres par défaut
    def __init__(self):
        self.activation_func = self.sigmoid  # Fonction d'activation
        self.weights = None  # Poids du perceptron
        self.biais = 0  # Biais du perceptron

        self.stock = [[np.copy(self.weights), np.copy(self.biais)]]
        self.age = 0
        self.life = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Méthode pour entraîner le perceptron
    def train(self, X, y, passages=1, lr=0.01):
        n_samples, n_features = X.shape
        if self.weights is None:
            self.weights = np.zeros(n_features)
        # Conversion de la sortie y en 0 et 1
        y_ = np.where(y > 0, 1, 0)

        # Apprentissage des poids
        for i in range(passages):
            self.life += 1
            self.age = self.life
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.biais
                y_predicted = self.activation_func(linear_output)

                # Règle de mise à jour du perceptron
                update = lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.biais += update

            self.stock.append([np.copy(self.weights), np.copy(self.biais)])

    # Méthode pour prédire les sorties
    def predict(self, X, age=-1):
        linear_output = np.dot(X, self.stock[age][0]) + self.stock[age][1]
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def predict_list(self, X, age=-1):
        y_hat = []
        for x in X:
            y_hat.append(self.predict(x, age=age))
        return y_hat

    # Méthode pour visualiser les étapes d'apprentissage
    def visualiser(self, X, y, etape, savemod=True, folder="../", img_name=None, nb_levels=10):
        plt.figure(figsize=(12, 8))

        # Calcul de la frontière de décision
        xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
                             np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Tracé de la frontière de décision
        contour = plt.contourf(xx, yy, Z, alpha=1, cmap='plasma', levels=np.linspace(0, 1, nb_levels + 1))

        # Tracé des données
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y).set_aspect('equal')

        plt.colorbar(contour, label='prediction')

        # Paramètres esthétiques
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Perceptron - Étape {etape}')
        plt.legend(loc='best')

        # Sauvegarde ou affichage de l'image
        if savemod:
            if img_name is None:
                img_name = f'fig{etape:03d}.svg'
            img_path = os.path.join(folder, img_name)
            plt.savefig(img_path, format='svg', transparent=True, bbox_inches='tight', pad_inches=0)
            print(f'{img_name} generated')
        else:
            plt.show()
        plt.close()

    def visualisation(self, data, labels, savemod=True, folder="../", img_name=None, age=-1, nb_levels=10,
                      show_data=True, show_previsu=True):
        plt.figure(figsize=(12, 8))
        if show_previsu:
            hh = .02
            x_r = data[:, 0].max() - data[:, 0].min()
            y_r = data[:, 1].max() - data[:, 1].min()
            x_min, x_max = data[:, 0].min() - x_r / 10, data[:, 0].max() + x_r / 10
            y_min, y_max = data[:, 1].min() - y_r / 10, data[:, 1].max() + y_r / 10
            xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / 100),
                                 np.arange(y_min, y_max, (y_max - y_min) / 100))
            Z = np.array(self.predict_list(np.c_[xx.ravel(), yy.ravel()], age=age))
            Z = Z.reshape(xx.shape)

            # plt.title('Prédiction du réseau en fonction de la position', fontsize=16, color=LIGHT_COLOR)

            contour = plt.contourf(xx, yy, Z, alpha=1, levels=[i / nb_levels for i in range(nb_levels + 1)],
                                   cmap='plasma')

        if show_data:
            sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels).set_aspect('equal')

        plt.colorbar(contour, label='prediction')
        if savemod:
            if img_name is None:
                img_name = f'fig{age:03d}.svg'
            img_path = os.path.join(folder, img_name)
            plt.savefig(img_path, format='svg', transparent=True, bbox_inches='tight', pad_inches=0)
            # print(f'{img_name} generated')
            return img_name
        else:
            plt.show()
        plt.close()

    def get_w(self, age=-1):
        s = self.stock[age]
        return {'w0':s[0][0].item(),
                'w1':s[0][1].item(),
                'b':s[1].item()}


def cluster(pos, size=100, etendue=(2, 2)):
    c = []
    for n in range(1, size + 1):
        # Modifier les décalages des moyennes pour déplacer les points
        x = rd.gauss(pos[0] + 5, etendue[0])  # Déplacement sur l'axe x
        y = rd.gauss(pos[1] - 5, etendue[1])  # Déplacement sur l'axe y
        c.append((x, y))
    return c


def clans1v1(center=(0, 0), dist=10, etendue=(2, 2)):
    # Décalage des centres des clusters
    p = np.array(cluster((center[0] + dist / 2, center[1] - dist / 2), size=1000, etendue=etendue) +
                 cluster((center[0] - dist / 2, center[1] + dist / 2), size=1000, etendue=etendue))
    l = [1 for _ in range(1000)] + [0 for _ in range(1000)]
    return p, l
