from flask import Flask, render_template, jsonify, send_file, url_for
from flask_socketio import SocketIO, Namespace, emit
import webbrowser
import os
import numpy as np
import random as rd
from sklearn.datasets import make_circles

from modules.ReseauSimple import ReseauSimple
from modules.Perceptron import Perceptron
from modules.Tictactoe import TicTacToe

app = Flask(__name__)
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/reseausimple')
def reseausimple():
    return render_template('reseausimple.html')


@app.route('/perceptron')
def perceptron():
    return render_template('perceptron.html')


@app.route('/tictactoe')
def tictactoe():
    return render_template('tictactoe.html')


class PerceptronNamespace(Namespace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = Perceptron()
        self.dots, self.labels = clans1v1(size=10, dist=5)

        os.makedirs('./static/svg', exist_ok=True)

    def on_init(self):
        emit('init_bar', {'life': self.network.life, 'age': self.network.age})
        self.on_update({'age': self.network.age})

    def get_image_name(self, age=None):
        if age is None:
            age = self.network.age

        return f'tmp-0-{age:03d}.svg'

    def generate_image(self, img_name=None, folder="static/svg", age=None):
        if img_name is None:
            img_name = self.get_image_name()

        if age is None:
            age = self.network.age

        self.network.visualisation(self.dots, self.labels, savemod=True, folder=folder, img_name=img_name, age=age)
        return url_for('static', filename=f'svg/{img_name}')

    def on_training(self, data):
        passages = data['passages']
        for _ in range(passages):
            self.network.train(self.dots, self.labels)
            image_path = self.generate_image()
            emit('nouvelle_image', {'image_path': image_path})
            emit('avancement', self.network.life)
            emit('update_net', self.network.get_w())

    def on_resume_training(self):
        self.network = Perceptron()
        self.on_init()

    def on_get_image(self, data):
        age = data['step']
        img_name = f'tmp-0-{age:03d}.svg'
        img_path = url_for('static', filename=f'svg/{img_name}')

        if not os.path.exists('.' + img_path):
            self.generate_image(img_name, age=age)

        emit('nouvelle_image', {'image_path': img_path})
        emit('update_net', self.network.get_w())

    def on_update(self, data):
        age = data['age']
        self.network.age = age
        if age == 0:
            emit('nouvelle_image', {'image_path': url_for('static', filename=f'svg/0-000.svg')})

        else:
            img_name = self.get_image_name()
            img_path = url_for('static', filename=f'svg/{img_name}')

            if not os.path.exists('.' + img_path):
                self.generate_image()

            emit('nouvelle_image', {'image_path': img_path})
            emit('update_net', self.network.get_w(age))


class ReseauSimpleNamespace(Namespace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network1 = ReseauSimple(xmod=True, idi='1')
        self.network2 = ReseauSimple(xmod=True, idi='2')

        self.network = None
        self.dots, self.labels = None, None

        os.makedirs('./static/svg', exist_ok=True)

    def on_init(self, net_id):
        if net_id == '1':
            self.network = self.network1
            self.dots, self.labels = clans1v1(size=1000)
        elif net_id == '2':
            self.network = self.network2
            self.dots, self.labels = circles()

        emit('init_bar', {'life': self.network.life, 'age': self.network.age})
        self.on_update({'age': self.network.age})

    def get_image_name(self, age=None):
        if age is None:
            age = self.network.age

        return f'tmp-{self.network.idi}-{age:03d}.svg'

    def generate_image(self, img_name=None, folder="static/svg", age=None):
        if img_name is None:
            img_name = self.get_image_name()

        if age is None:
            age = self.network.age

        self.network.visualisation(self.dots, self.labels, savemod=True, folder=folder, img_name=img_name, age=age)
        return url_for('static', filename=f'svg/{img_name}')

    def on_training(self, data):
        passages = data['passages']
        for _ in range(passages):
            self.network.train(self.dots, self.labels)
            image_path = self.generate_image()
            emit('nouvelle_image', {'image_path': image_path})
            emit('avancement', self.network.life)
            emit('update_net', self.network.get_w())

    def on_resume_training(self):
        if self.network.idi == '1':
            self.network1 = ReseauSimple(xmod=True, idi='1')
        elif self.network.idi == '2':
            self.network2 = ReseauSimple(xmod=True, idi='2')
        self.on_init(self.network.idi)

    def on_update(self, data):
        age = data['age']
        self.network.age = age
        if age == 0:
            emit('nouvelle_image', {'image_path': url_for('static', filename=f'svg/{self.network.idi}-000.svg')})

        else:
            img_name = self.get_image_name()
            img_path = url_for('static', filename=f'svg/{img_name}')

            if not os.path.exists('.' + img_path):
                self.generate_image()

            emit('nouvelle_image', {'image_path': img_path})
            emit('update_net', self.network.get_w(age))


class TicTacToeNamespace(Namespace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.game = None

    def on_play(self, data):
        self.game = TicTacToe(data["p1"], data["p2"])

    def on_move(self, data=None):
        move_data, net_data = self.game.move(data)
        emit('moved', move_data)

    def on_is_winner(self):
        ans = self.game.is_winner()
        if ans is not None:
            emit('winner', ans)


@socketio.on('close-app')
def handle_closing_page():
    print("Page is closing. Stopping the Python program.")
    for file in os.listdir('./static/svg'):
        if file.startswith('tmp-'):
            os.remove(os.path.join('./static/svg', file))

    os._exit(0)


def cluster(pos, size=100, etendue=(2, 2)):
    c = []
    for n in range(1, size + 1):
        x = rd.gauss(pos[0], etendue[0])
        y = rd.gauss(pos[1], etendue[1])
        c.append((x, y))

    return c


def clans1v1(center=(0, 0), dist=40, etendue=(2, 2), size=100, invert=1):
    p = np.array(cluster((center[0] + invert * (dist / 2), center[1] + dist / 2), size=size, etendue=etendue) +
                 cluster((center[0] - invert * (dist / 2), center[1] - dist / 2), size=size, etendue=etendue))
    l = np.array([1 for _ in range(size)] + [0 for _ in range(size)])
    return p, l


def circles():
    return make_circles(5000, noise=0.07, factor=0.1, random_state=42)


if __name__ == '__main__':
    socketio.on_namespace(ReseauSimpleNamespace('/reseausimple'))
    socketio.on_namespace(PerceptronNamespace('/perceptron'))
    socketio.on_namespace(TicTacToeNamespace('/tictactoe'))

    webbrowser.open('http://127.0.0.1:5000/')
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, use_reloader=False)
    handle_closing_page()
