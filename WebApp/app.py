from flask import Flask, render_template, jsonify, send_file, url_for
from flask_socketio import SocketIO
import webbrowser
import os
from modules.ReseauSimple import ReseauSimple, clans1v1, clans2v2, circles

from pprint import pprint

app = Flask(__name__)
socketio = SocketIO(app)

reseauclans = ReseauSimple(xmod=True)
reseaucirlces = ReseauSimple(xmod=True)

data, label = clans1v1()
#data, label = circles()

er = reseauclans


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/reseausimple')
def reseausimple():
    return render_template('reseausimple.html')


@socketio.on('connect', namespace='/reseausimple')
def test_connect():
    print("Client connected")


@socketio.on('disconnect', namespace='/reseausimple')
def test_disconnect():
    print("Client disconnected")


@socketio.on('closing_page', namespace='/reseausimple')
def handle_closing_page():
    print("Page is closing. Stopping the Python program.")
    socketio.stop()
    for svgimage in os.listdir('./static/svg'):
        os.remove(os.path.join('./static/svg', svgimage))


def generate_image(img_name, folder="static/svg", show_data=True, show_previsu=True, etape=-1):
    er.visualisation(data, label, savemod=True, folder=folder, img_name=img_name, show_data=show_data,
                     show_previsu=show_previsu, etape=etape)
    return url_for('static', filename=f'svg/{img_name}')


@socketio.on('start_training', namespace='/reseausimple')
def emit_training(webdata):
    global er
    os.makedirs('./static/svg', exist_ok=True)
    passages = webdata['passages']
    dp = er.nb_pass
    for i in range(dp + 1, passages + dp + 1):
        er.train(data, label)
        # image_name = er.visualisation(data, label, savemod=True, folder="static/svg", etape=i)
        image_path = generate_image(f'fig{i:03d}.svg')
        socketio.emit('nouvelle_image', {'image_path': image_path}, namespace='/reseausimple')
        socketio.emit('avancement', i, namespace='/reseausimple')
        # pprint(er.get_w(i))
        socketio.emit('update_net', er.get_w(i), namespace='/reseausimple')


@socketio.on('get_image', namespace='/reseausimple')
def emit_image(webdata):
    etape = webdata['etape']
    img_name = f'fig{etape:03d}.svg'
    img_path = url_for('static', filename=f'svg/{img_name}')

    if not os.path.exists('.' + img_path):
        generate_image(img_name, etape=etape)

    socketio.emit('nouvelle_image', {'image_path': img_path}, namespace='/reseausimple')
    socketio.emit('update_net', er.get_w(etape), namespace='/reseausimple')


@socketio.on('resume_training', namespace='/reseausimple')
def resume_training():
    global er
    er = ReseauSimple(xmod=True)
    img_path = generate_image('fig000.svg', folder="static/svg", show_data=True, show_previsu=True)
    socketio.emit('nouvelle_image', {'image_path': img_path}, namespace='/reseausimple')
    socketio.emit('update_net', er.get_w(0), namespace='/reseausimple')


if __name__ == '__main__':
    webbrowser.open('http://127.0.0.1:5000/')
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, use_reloader=False)
