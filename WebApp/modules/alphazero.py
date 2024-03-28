import os

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense
tf.disable_v2_behavior()
import random as rd

import math

from copy import deepcopy


class NeuralNetwork(object):
    """
    Représente le ResNet pour la politique et la valeur.
    Attributs:
        row: Un entier indiquant la longueur de la ligne de la grille.
        column: Un entier indiquant la longueur de la colonne de la grille.
        action_size: Un entier indiquant le nombre total de cases de la grille.
        pi: Un tenseur TF pour les probabilités de recherche.
        v: Un tenseur TF pour les valeurs de recherche.
        states: Un tenseur TF avec les dimensions de la grille.
        training: Un scalaire booléen TF.
        train_pis: Un tenseur TF pour les probabilités cibles de recherche.
        train_vs: Un tenseur TF pour les valeurs cibles de recherche.
        loss_pi: Un tenseur TF pour la sortie de l'entropie croisée softmax sur pi.
        loss_v: Un tenseur TF pour la sortie de l'erreur quadratique moyenne sur v.
        total_loss: Un tenseur TF pour stocker l'addition des pertes pi et v.
        train_op: Un tenseur TF pour la sortie de l'optimiseur d'entraînement.
        saver: Un sauvegardateur TF pour écrire les points de contrôle de l'entraînement.
        sess: Une session TF pour exécuter les opérations sur le graphe.
    """

    def __init__(self, game):
        """Initialise NeuralNetwork avec le graphe du réseau ResNet."""
        self.row, self.column = game.board.shape
        self.action_size = game.board.size
        self.pi = None
        self.v = None

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.states = tf.placeholder(tf.float32, shape=[None, self.row, self.column])
            self.training = tf.placeholder(tf.bool)

            # Couche d'entrée
            input_layer = tf.reshape(self.states, [-1, self.row, self.column, 1])

            # Bloc de convolution
            conv1 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", strides=1)(input_layer)
            batch_norm1 = BatchNormalization()(conv1)
            relu1 = tf.nn.relu(batch_norm1)
            resnet_in_out = relu1

            # Tour résiduel
            for i in range(CFG.resnet_blocks):
                # Bloc résiduel
                conv2 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", strides=1)(resnet_in_out)
                batch_norm2 = BatchNormalization()(conv2)
                relu2 = tf.nn.relu(batch_norm2)

                conv3 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", strides=1)(relu2)
                batch_norm3 = BatchNormalization()(conv3)

                resnet_skip = tf.add(batch_norm3, resnet_in_out)
                resnet_in_out = tf.nn.relu(resnet_skip)

            # Tête de politique
            conv4 = Conv2D(filters=2, kernel_size=(1, 1), padding="same", strides=1)(resnet_in_out)
            batch_norm4 = BatchNormalization()(conv4)
            relu4 = tf.nn.relu(batch_norm4)
            relu4_flat = tf.reshape(relu4, [-1, self.row * self.column * 2])
            logits = Dense(units=self.action_size)(relu4_flat)
            self.pi = tf.nn.softmax(logits)

            # Tête de valeur
            conv5 = Conv2D(filters=1, kernel_size=(1, 1), padding="same", strides=1)(resnet_in_out)
            batch_norm5 = BatchNormalization()(conv5)
            relu5 = tf.nn.relu(batch_norm5)
            relu5_flat = tf.reshape(relu5, [-1, self.action_size])
            dense1 = Dense(units=256)(relu5_flat)
            relu6 = tf.nn.relu(dense1)
            dense2 = Dense(units=1)(relu6)
            self.v = tf.nn.tanh(dense2)

            # Fonction de perte
            self.train_pis = tf.placeholder(tf.float32, shape=[None, self.action_size])
            self.train_vs = tf.placeholder(tf.float32, shape=[None])
            self.loss_pi = tf.losses.softmax_cross_entropy(self.train_pis, self.pi)
            self.loss_v = tf.losses.mean_squared_error(self.train_vs, tf.reshape(self.v, shape=[-1, ]))
            self.total_loss = self.loss_pi + self.loss_v

            optimizer = tf.train.MomentumOptimizer(learning_rate=CFG.learning_rate, momentum=CFG.momentum,
                                                   use_nesterov=False)
            self.train_op = optimizer.minimize(self.total_loss)

            # Créer un sauvegardateur pour écrire les points de contrôle de l'entraînement.
            self.saver = tf.train.Saver()

            # Créer une session pour exécuter les Ops sur le Graph.
            self.sess = tf.Session()

            # Initialiser la session.
            self.sess.run(tf.global_variables_initializer())


class NeuralNetworkWrapper(object):
    """
    Classe enveloppe pour la classe NeuralNetwork.
    Attributs:
        game: Un objet contenant l'etat du jeu.
        net: Un objet contenant le reseau neuronal.
        sess: Une session TF pour executer les operations sur le graphe.
    """

    def __init__(self, game, model = "./models/"):
        """Initialise NeuralNetworkWrapper avec l'etat du jeu et la session TF."""
        self.game = game
        self.net = NeuralNetwork(self.game)
        self.sess = self.net.sess
        self.modeldirectory = model

    def play(self, game) :
        mcts = MonteCarloTreeSearch(self)
        node = TreeNode()
        best_child = mcts.search(game, node, CFG.temp_init)
        return best_child.parent.child_psas

    def predict(self, state):
        """
        Predit les probabilites de mouvement et les valeurs d'etat etant donne un etat du jeu.
        Args:
            state: Une liste contenant l'etat du jeu sous forme de matrice.
        Returns:
            Un vecteur de probabilite et une valeur scalaire.
        """
        state = np.array(state)
        state = state[np.newaxis, :, :]
        pi, v = self.sess.run([self.net.pi, self.net.v],
                              feed_dict={self.net.states: state,
                                         self.net.training: False})
        # Masquage des probabilités des coups impossibles
        impossible_moves = np.where(state != 0)  # Trouver les coups déjà joués
        pi[0, 3*impossible_moves[1]+ impossible_moves[2]] = -np.inf
        return pi[0], v[0][0]


    def train(self, training_data):
        """
        Entraine le reseau en utilisant les etats, les pis et les vs des jeux en auto-apprentissage.
        Args:
            training_data: Une liste contenant des etats, des pis et des vs.
        """
        print("\nEntrainement du reseau.\n")

        for epoch in range(CFG.epochs):
            print("epoque", epoch + 1)

            examples_num = len(training_data)
            # Divise l'epoque en lots.
            for i in range(0, examples_num, CFG.batch_size):
                states, pis, vs = map(list,
                                      zip(*training_data[i:i + CFG.batch_size]))

                feed_dict = {self.net.states: states,
                             self.net.train_pis: pis,
                             self.net.train_vs: vs,
                             self.net.training: True}

                self.sess.run(self.net.train_op, feed_dict=feed_dict)

                pi_loss, v_loss = self.sess.run(
                    [self.net.loss_pi, self.net.loss_v],
                    feed_dict=feed_dict)

                # Enregistre la perte de pi et de v dans un fichier.
                if CFG.record_loss:
                    # Cree le repertoire s'il n'existe pas.
                    if not os.path.exists(self.modeldirectory):
                        os.mkdir(self.modeldirectory)

                    file_path = self.modeldirectory + CFG.loss_file

                    with open(file_path, 'a') as loss_file:
                        loss_file.write('%f,%f\n' % (pi_loss, v_loss))

        print("\n")

    def save_model(self, filename="current_model"):
        """
        Enregistre le modele de reseau au chemin de fichier donne.
        Args:
            filename: Une chaine representant le nom du modele.
        """
        # Cree le repertoire s'il n'existe pas.
        if not os.path.exists(self.modeldirectory):
            os.mkdir(self.modeldirectory)

        file_path = self.modeldirectory + filename

        print("Enregistrement du modele:", filename, "a", self.modeldirectory)
        self.net.saver.save(self.sess, file_path)

    def load_model(self, filename="current_model"):
        """
        Charge le modele de reseau au chemin de fichier donne.
        Args:
            filename: Une chaine representant le nom du modele.
        """
        file_path = self.modeldirectory + filename

        print("Chargement du modele:", filename, "depuis", self.modeldirectory)
        self.net.saver.restore(self.sess, file_path)





class Train(object):
    """
    Classe avec des fonctions pour entrainer le reseau neuronal en utilisant MCTS.
    Attributs:
        game: Un objet contenant l'etat du jeu.
        net: Un objet contenant le reseau neuronal.
    """

    def __init__(self, game, net):
        """Initialise Train avec l'etat du jeu et le reseau neuronal."""
        self.game = game
        self.net = net
        self.eval_net = NeuralNetworkWrapper(game)

    def start(self):
        """Boucle d'entrainement principale."""
        num_games = CFG.num_eval_games
        eval_win_rate = CFG.eval_win_rate
        for i in range(CFG.num_iterations):
            print("Iteration", i + 1)

            training_data = []  # liste pour stocker les etats, pis et vs de l'auto-apprentissage

            for j in range(CFG.num_games):
                print("Demarrage de l'entrainement du jeu auto-apprentissage", j + 1)
                game = self.game.clone()  # Cree un clone frais pour chaque jeu.
                self.play_game(game, training_data)
            for j in range(CFG.num_gamesh):
                print("Demarrage de l'entrainement du jeu auto-apprentissage contre le hasard", j + 1)
                game = self.game.clone()  # Cree un clone frais pour chaque jeu.
                self.play_gameh(game, training_data)
            # Enregistre le modele de reseau neuronal actuel.
            self.net.save_model()

            # Charge le modele recemment enregistre dans le reseau evaluateur.
            self.eval_net.load_model()

            # Entraine le reseau en utilisant les valeurs de l'auto-apprentissage.
            self.net.train(training_data)

            # Initialise les objets MonteCarloTreeSearch pour les deux reseaux.
            current_mcts = MonteCarloTreeSearch(self.net)
            eval_mcts = MonteCarloTreeSearch(self.eval_net)

            evaluator = Evaluate(current_mcts=current_mcts, eval_mcts=eval_mcts, game=self.game)
            wins, losses = evaluator.evaluate()
            if losses == 0:
                win_rate = wins
            else:
                win_rate = wins / losses

            print("taux de victoire:", win_rate)

            if win_rate > eval_win_rate:
                # Enregistre le modele actuel comme meilleur modele.
                print("Nouveau modele enregistre comme meilleur modele.")
                self.net.save_model("best_model")
            else:
                print("Nouveau modele abandonne et modele precedent charge.")
                # Abandonne le modele actuel et utilise le meilleur modele precedent.
                self.net.load_model()
            
            with open("alfazero_winning_rate.csv", 'a') as win_rate_file:
                win_rate_file.write('%d,%d,%d,%f\n' % (i + 1, wins, losses, win_rate))

    def play_game(self, game, training_data):
        """
        Boucle pour chaque jeu en auto-apprentissage.
        Execute MCTS pour chaque etat de jeu et joue un mouvement en fonction de la sortie de MCTS.
        S'arrete lorsque le jeu est termine et imprime un gagnant.
        Args:
            game: Un objet contenant l'etat du jeu.
            training_data: Une liste pour stocker les etats, pis et vs de l'auto-apprentissage.
        """
        mcts = MonteCarloTreeSearch(self.net)

        game_over = False
        value = 0
        count = 0
        self_play_data = []

        node = TreeNode()

        # Continue a jouer jusqu'a ce que le jeu soit dans un etat terminal.
        while not game_over:
            # Simulations MCTS pour obtenir le meilleur noeud enfant.
            if count < CFG.temp_thresh:
                best_child = mcts.search(game, node, CFG.temp_init)
            else:
                best_child = mcts.search(game, node, CFG.temp_final)
            if best_child == None:
                game_over, value = True, -1 * game.player.num
                break
            
            # Stocke l'etat, la probabilite et v pour l'entrainement.
            self_play_data.append([deepcopy(game.board), deepcopy(best_child.parent.child_psas), 0])

            action = best_child.action

            game.make_move(action[1],action[2])  # Joue l'action du noeud enfant.
            count += 1
            game_over, value = game.check_winner(game.player.num)
            best_child.parent = None
            node = best_child  # Fait du noeud enfant le noeud racine.

        # Met a jour v comme la valeur du resultat du jeu.
        for game_state in self_play_data:
            value = -value
            game_state[2] = value
            self.augment_data(game_state, training_data)
    def play_gameh(self, game, training_data):

        mcts = MonteCarloTreeSearch(self.net)

        game_over = False
        value = 0
        count = 0
        self_play_data = []

        node = TreeNode()

        # Continue a jouer jusqu'a ce que le jeu soit dans un etat terminal.
        while not game_over:
            # Simulations MCTS pour obtenir le meilleur noeud enfant.
            if game.player.num == -1 :
                if count < CFG.temp_thresh:
                    best_child = mcts.search(game, node, CFG.temp_init)
                else:
                    best_child = mcts.search(game, node, CFG.temp_final)
                if best_child == None:
                    game_over, value = True, -1 * game.player.num
                    break

                # Stocke l'etat, la probabilite et v pour l'entrainement.
                self_play_data.append([deepcopy(game.board), deepcopy(best_child.parent.child_psas), 0])
                action = best_child.action
                game.make_move(action[1],action[2])  # Joue l'action du noeud enfant.
                count += 1
                game_over, value = game.check_winner(game.player.num)
                best_child.parent = None
                node = best_child  # Fait du noeud enfant le noeud racine.
            else : 
                random_move = rd.choice(game.get_possible_move())
                game.make_move(random_move[0],random_move[1])  
                count += 1
            
        # Met a jour v comme la valeur du resultat du jeu.
        for game_state in self_play_data:
            value = -value
            game_state[2] = value
            self.augment_data(game_state, training_data)
    def augment_data(self, game_state, training_data):
        """
        Boucle pour chaque jeu en auto-apprentissage.
        Execute MCTS pour chaque etat de jeu et joue un mouvement en fonction de la sortie de MCTS.
        S'arrete lorsque le jeu est termine et imprime un gagnant.
        Args:
            game_state: Un objet contenant l'etat, les pis et la valeur.
            training_data: Une liste pour stocker les etats, pis et vs de l'auto-apprentissage.
        """
        state = deepcopy(game_state[0])
        psa_vector = deepcopy(game_state[1])

        training_data.append([state, psa_vector, game_state[2]])


















# Class to represent a configuration file.
class CFG(object):
    """
    Represents a static configuration file used through the application.
    Attributes:
        num_iterations: Number of iterations.
        num_games: Number of self play games played during each iteration.
        num_mcts_sims: Number of MCTS simulations per game.
        c_puct: The level of exploration used in MCTS.
        l2_val: The level of L2 weight regularization used during training.
        momentum: Momentum Parameter for the momentum optimizer.
        learning_rate: Learning Rate for the momentum optimizer.
        t_policy_val: Value for policy prediction.
        temp_init: Initial Temperature parameter to control exploration.
        temp_final: Final Temperature parameter to control exploration.
        temp_thresh: Threshold where temperature init changes to final.
        epochs: Number of epochs during training.
        batch_size: Batch size for training.
        dirichlet_alpha: Alpha value for Dirichlet noise.
        epsilon: Value of epsilon for calculating Dirichlet noise.
        model_directory: Name of the directory to store models.
        num_eval_games: Number of self-play games to play for evaluation.
        eval_win_rate: Win rate needed to be the best model.
        load_model: Binary to initialize the network with the best model.
        resnet_blocks: Number of residual blocks in the resnet.
        record_loss: Binary to record policy and value loss to a file.
        loss_file: Name of the file to record loss.
    """
    
    
    num_iterations = 1000       # Nombre d'iterations. Il specifie le nombre de fois que certaines operations doivent etre repetees dans l'application.
    num_games = 30             # Nombre de parties jouees en auto-apprentissage pendant chaque iteration. Cela determine la quantite de donnees generees pour l'entrainement.     
    epochs = 10                 # Nombre d'epoques pendant l'entrainement. Une epoque correspond a un passage complet a travers l'ensemble de donnees d'entrainement.
    num_eval_games = 20         # Nombre de parties en auto-apprentissage a jouer pour l'evaluation. Il determine le nombre de parties utilisees pour evaluer les performances du modele.
    num_gamesh = 30


    c_puct = 2                  # Niveau d'exploration utilise dans MCTS. Il controle le degre d'exploration versus l'exploitation dans l'algorithme de recherche MCTS.
    temp_init = 1               # Parametre de temperature initial pour controler l'exploration. Il est utilise dans les strategies d'exploration-exploitation dans l'apprentissage par renforcement.
    temp_final = 0.001            # Parametre de temperature final pour controler l'exploration. Il represente la temperature apres laquelle l'exploration est reduite.   
    temp_thresh = 10             # Seuil ou l'initialisation de la temperature passe a la finale. Il definit le point a partir duquel la temperature initiale est reduite jusqu'a la temperature finale.
    dirichlet_alpha = 0.5       # Valeur alpha pour le bruit de Dirichlet. Le bruit de Dirichlet est utilise pour augmenter la diversite des coups explores lors de la recherche MCTS.
    epsilon = 0.25               # Valeur d'epsilon pour calculer le bruit de Dirichlet. Il controle l'importance du bruit de Dirichlet dans la recherche MCTS.
    num_mcts_sims = 30          # Nombre de simulations MCTS (Monte Carlo Tree Search) par partie. MCTS est une technique utilisee dans les jeux pour decider du meilleur coup a jouer en simulant des sequences de jeu.

    l2_val = 0.0001           # Niveau de regularisation des poids L2 utilise pendant l'entrainement. Il permet de limiter la complexite du modele et d'eviter le surapprentissage.
    momentum = 0.9              # Parametre de momentum pour l'optimiseur de momentum. Le momentum est une technique d'optimisation qui accelere la convergence en accumulant une moyenne exponentielle des gradients precedents.
    learning_rate = 0.01         # Taux d'apprentissage pour l'optimiseur de momentum. Il controle la taille des pas effectues lors de la mise a jour des poids du reseau neuronal pendant l'entrainement.
    t_policy_val = 0.0001       # Valeur pour la prediction de politique. C'est un parametre utilise dans la prediction des politiques dans un modele d'apprentissage par renforcement.
    resnet_blocks = 5           # Nombre de blocs residuels dans le ResNet. Le ResNet est une architecture de reseau de neurones profonds utilisee dans de nombreuses taches de vision par ordinateur.
    batch_size = 128            # Taille du lot pour l'entrainement. Il specifie le nombre d'exemples d'entrainement utilises dans une seule iteration de l'algorithme d'optimisation.


    eval_win_rate = 1        # Taux de victoire necessaire pour etre le meilleur modele. Il definit le seuil de performance pour qu'un modele soit considere comme le meilleur.
    load_model = 1              # Binaire pour initialiser le reseau avec le meilleur modele. Il indique si le modele doit etre initialise avec le meilleur modele precedemment sauvegarde.
    record_loss = 0             # Binaire pour enregistrer la perte de politique et de valeur dans un fichier. Il specifie si les pertes doivent etre enregistrees pendant l'entrainement.
    loss_file = "alfazero_loss.csv"     # Nom du fichier pour enregistrer la perte. Il specifie le nom du fichier ou les pertes seront enregistrees pendant l'entrainement.







# Classe pour evaluer le reseau.
class Evaluate(object):
    """
    Represente le Resnet de la politique et de la valeur.
    Attributs:
        current_mcts: Un objet pour le MCTS du reseau actuel.
        eval_mcts: Un objet pour le MCTS du reseau d'evaluation.
        game: Un objet contenant l'etat du jeu.
    """

    def __init__(self, current_mcts, eval_mcts, game):
        """Initialise Evaluate avec les MCTS des deux reseaux et l'etat du jeu."""
        self.current_mcts = current_mcts
        self.eval_mcts = eval_mcts
        self.game = game

    def evaluate(self):
        """
        Joue des parties d'auto-apprentissage entre les deux reseaux et enregistre les statistiques du jeu.
        Returns:
            Le nombre de victoires et de defaites du point de vue du reseau actuel.
        """
        wins = 0
        losses = 0

        # Boucle d'auto-apprentissage
        for i in range(CFG.num_eval_games):
            print("Debut de l'auto-apprentissage d'evaluation du jeu:", i+1)

            game = self.game.clone()  # Cree un clone frais pour chaque jeu.
            game_over = False
            value = 0
            node = TreeNode()

            
            
            game.change_player()

            
            # Continue a jouer jusqu'a ce que le jeu soit dans un etat terminal.
            while not game_over:
                # Simulations MCTS pour obtenir le meilleur noeud enfant.
                # Si player_to_eval est 1, joue en utilisant le reseau actuel
                # Sinon, joue en utilisant le reseau d'evaluation.
                if game.player.num == 1:
                    best_child = self.current_mcts.search(game, node, CFG.temp_final)
                else:
                    best_child = self.eval_mcts.search(game, node, CFG.temp_final)
                
                action = best_child.action
                game.make_move(action[1], action[2])  # Joue l'action du noeud enfant.

                game_over, value = game.check_winner(1)
                print(game.board)
                best_child.parent = None
                node = best_child  # Fait du noeud enfant le noeud racine.
            print(game.board)
            if value == 1:
                wins += 1
            elif value == -1:
                losses += 1
        return wins, losses






# Classe pour representer un etat du plateau et stocker les statistiques pour les actions a cet etat.
class TreeNode(object):
    """
    Represente un etat du plateau et stocke les statistiques pour les actions a cet etat.
    Attributs:
        Nsa: Un entier pour le nombre de visites.
        Wsa: Un flottant pour la valeur totale de l'action.
        Qsa: Un flottant pour la valeur moyenne de l'action.
        Psa: Un flottant pour la probabilite prioritaire d'atteindre ce noeud.
        action: Un tuple (ligne, colonne) du mouvement prioritaire pour atteindre ce noeud.
        children: Une liste qui stocke les noeuds enfants.
        child_psas: Un vecteur contenant les probabilites enfants.
        parent: Un TreeNode representant le noeud parent.
    """

    def __init__(self, parent=None, action=None, psa=0.0, child_psas=[]):
        """Initialise TreeNode avec les statistiques initiales et les donnees."""
        self.Nsa = 0
        self.Wsa = 0.0
        self.Qsa = 0.0
        self.Psa = psa
        self.action = action
        self.children = []
        self.child_psas = child_psas
        self.parent = parent

    def is_not_leaf(self):
        """
        Verifie si un TreeNode est une feuille.
        Returns:
            Une valeur booleenne indiquant si un TreeNode est une feuille.
        """
        if len(self.children) > 0:
            return True
        return False

    def select_child(self):
        """
        Selectionne un noeud enfant en fonction de la formule AlphaZero PUCT.
        Returns:
            Un TreeNode enfant qui est le plus prometteur selon PUCT.
        """
        c_puct = CFG.c_puct

        highest_uct = 0
        highest_index = 0

        # Selectionne l'enfant avec la plus haute valeur Q + U
        for idx, child in enumerate(self.children):
            uct = child.Qsa + child.Psa * c_puct * (
                    math.sqrt(self.Nsa) / (1 + child.Nsa))
            if uct > highest_uct:
                highest_uct = uct
                highest_index = idx

        return self.children[highest_index]

    def expand_node(self, game, psa_vector):
        """
        etend le noeud actuel en ajoutant les mouvements valides comme enfants.
        Args:
            game: Un objet contenant l'etat du jeu.
            psa_vector: Une liste contenant les probabilites de mouvement pour chaque mouvement.
        """
        self.child_psas = deepcopy(psa_vector)
        valid_moves = game.get_valid_move()
        for idx, move in enumerate(valid_moves):
            if move[0] != 0 :
                action = deepcopy(move)
                self.add_child_node(parent=self, action=action, psa=psa_vector[idx])

    def add_child_node(self, parent, action, psa=0.0):
        """
        Cree et ajoute un TreeNode enfant au noeud actuel.
        Args:
            parent: Un TreeNode qui est le parent de ce noeud.
            action: Un tuple (ligne, colonne) du mouvement prioritaire pour atteindre ce noeud.
            psa: Un flottant representant la probabilite brute de mouvement pour ce noeud.
        Returns:
            Le TreeNode enfant nouvellement cree.
        """

        child_node = TreeNode(parent=parent, action=action, psa=psa)
        self.children.append(child_node)
        return child_node

    def back_prop(self, wsa, v):
        """
        Met a jour les statistiques du noeud actuel en fonction du resultat du jeu.
        Args:
            wsa: Un flottant representant la valeur de l'action pour cet etat.
            v: Un flottant representant la valeur du reseau de cet etat.
        """
        self.Nsa += 1
        self.Wsa = wsa + v
        self.Qsa = self.Wsa / self.Nsa











# Classe pour representer un algorithme de recherche Monte Carlo Tree Search.
class MonteCarloTreeSearch(object):
    """
    Represente un algorithme de recherche Monte Carlo Tree Search.
    Attributs:
        root: Un TreeNode representant l'etat du plateau et ses statistiques.
        game: Un objet contenant l'etat du jeu.
        net: Un objet contenant le reseau neuronal.
    """

    def __init__(self, net):
        """Initialise TreeNode avec le TreeNode, le plateau et le reseau neuronal."""
        self.root = None
        self.game = None
        self.net = net

    def search(self, game, node, temperature):
        """
        Boucle MCTS pour obtenir le meilleur mouvement qui peut etre joue a un etat donne.
        Args:
            game: Un objet contenant l'etat du jeu.
            node: Un TreeNode representant l'etat du plateau et ses statistiques.
            temperature: Un flottant pour controler le niveau d'exploration.
        Returns:
            Un noeud enfant representant le meilleur mouvement a jouer a cet etat.
        """
        self.root = node
        self.game = game

        for _ in range(CFG.num_mcts_sims):
            node = self.root
            game = self.game.clone()  # Creez un clone frais pour chaque boucle.

            # Boucle lorsque le noeud n'est pas une feuille.
            while node.is_not_leaf():
                node = node.select_child()
                game.make_move(node.action[1], node.action[2])

            # Obtenez les probabilites de mouvement et les valeurs du reseau pour cet etat.
            psa_vector, v = self.net.predict(game.board)

            # Ajoutez du bruit de Dirichlet au vecteur de probabilite de mouvement du noeud racine.
            if node.parent is None:
                psa_vector = self.add_dirichlet_noise(game, psa_vector)

            valid_moves = game.get_valid_move()

            for idx, move in enumerate(valid_moves):
                if move[0] == 0:
                    psa_vector[idx] = 0

            psa_vector_sum = sum(psa_vector)

            # Renormalisez le vecteur psa
            if psa_vector_sum > 0:
                psa_vector /= psa_vector_sum

            # Essayez d'etendre le noeud actuel.
            node.expand_node(game=game, psa_vector=psa_vector)
            _, wsa = game.check_winner(game.player.num)

            # Propager les statistiques du noeud vers le haut jusqu'au noeud racine.
            while node is not None:
                wsa = -wsa
                v = -v
                node.back_prop(wsa, v)
                node = node.parent

        highest_nsa = 0
        highest_index = 0

        # Selectionnez le mouvement de l'enfant en utilisant un parametre de temperature.
        for idx, child in enumerate(self.root.children):
            temperature_exponent = int(1 / temperature)

            if child.Nsa ** temperature_exponent > highest_nsa:
                highest_nsa = child.Nsa ** temperature_exponent
                highest_index = idx

        if len(self.root.children) == 0:
            return None
        return self.root.children[highest_index]

    def add_dirichlet_noise(self, game, psa_vector):
        """
        Ajoutez du bruit de Dirichlet au vecteur de probabilite de mouvement du noeud racine.
        C'est pour une exploration supplementaire.
        Args:
            game: Un objet contenant l'etat du jeu.
            psa_vector: Un vecteur de probabilite.
        Returns:
            Un vecteur de probabilite auquel du bruit de Dirichlet a ete ajoute.
        """
        dirichlet_input = [CFG.dirichlet_alpha for x in range(game.board.size)]

        dirichlet_list = np.random.dirichlet(dirichlet_input)
        noisy_psa_vector = []

        for idx, psa in enumerate(psa_vector):
            noisy_psa_vector.append(
                (1 - CFG.epsilon) * psa + CFG.epsilon * dirichlet_list[idx])

        return noisy_psa_vector

