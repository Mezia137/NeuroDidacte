from flask import Flask, render_template, jsonify, send_file, url_for
from flask_socketio import SocketIO, Namespace, emit
import webbrowser
import os
from modules.ReseauSimple import ReseauSimple, clans1v1, clans2v2, circles
from modules.Perceptron import Perceptron

from pprint import pprint

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


class ReseauSimpleNamespace(Namespace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = ReseauSimple(xmod=True)
        self.dots, self.labels = clans1v1(size=1000)

    def generate_image(self, img_name, folder="static/svg", show_data=True, show_previsu=True, step=-1):
        self.network.visualisation(self.dots, self.labels, savemod=True, folder=folder, img_name=img_name, age=step)
        return url_for('static', filename=f'svg/{img_name}')

    def on_training(self, data):
        os.makedirs('./static/svg', exist_ok=True)
        passages = data['passages']
        dp = self.network.nb_pass
        for i in range(dp + 1, passages + dp + 1):
            self.network.train(self.dots, self.labels)
            # image_name = er.visualisation(data, label, savemod=True, folder="static/svg", step=i)
            image_path = self.generate_image(f'tmp-1-{i:03d}.svg')
            emit('nouvelle_image', {'image_path': image_path})
            emit('avancement', i)
            # pprint(er.get_w(i))
            emit('update_net', self.network.get_w(i))

    def on_resume_training(self):
        self.network = ReseauSimple(xmod=True)

    def on_get_image(self, data):
        step = data['step']
        img_name = f'tmp-1-{step:03d}.svg'
        img_path = url_for('static', filename=f'svg/{img_name}')

        if not os.path.exists('.' + img_path):
            self.generate_image(img_name, step=step)

        emit('nouvelle_image', {'image_path': img_path})
        emit('update_net', self.network.get_w(step))


class PerceptronNamespace(Namespace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = Perceptron()
        self.dots, self.labels = clans1v1(size=1000, etendue=(10, 10))

    def generate_image(self, img_name, folder="static/svg", show_data=True, show_previsu=True, step=-1):
        self.network.visualisation(self.dots, self.labels, savemod=True, folder=folder, img_name=img_name, age=step)
        return url_for('static', filename=f'svg/{img_name}')

    def on_training(self, data):
        os.makedirs('./static/svg', exist_ok=True)
        passages = data['passages']
        dp = self.network.nb_pass
        for i in range(dp + 1, passages + dp + 1):
            self.network.train(self.dots, self.labels)
            # image_name = er.visualisation(data, label, savemod=True, folder="static/svg", step=i)
            image_path = self.generate_image(f'tmp-0-{i:03d}.svg')
            emit('nouvelle_image', {'image_path': image_path})
            emit('avancement', i)
            # pprint(er.get_w(i))
            emit('update_net', self.network.get_w())

    def on_resume_training(self):
        self.network = Perceptron()

    def on_get_image(self, data):
        step = data['step']
        img_name = f'tmp-0-{step:03d}.svg'
        img_path = url_for('static', filename=f'svg/{img_name}')

        if not os.path.exists('.' + img_path):
            self.generate_image(img_name, step=step)

        emit('nouvelle_image', {'image_path': img_path})
        emit('update_net', self.network.get_w())


if __name__ == '__main__':
    socketio.on_namespace(ReseauSimpleNamespace('/reseausimple'))
    socketio.on_namespace(PerceptronNamespace('/perceptron'))

    webbrowser.open('http://127.0.0.1:5000/')
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, use_reloader=False)
