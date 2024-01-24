
from flask import Flask, render_template, jsonify, send_file,url_for
from flask_socketio import SocketIO
import webbrowser
from modules.ReseauSimple import ReseauSimple, clans1v1, clans2v2



app = Flask(__name__)
socketio = SocketIO(app)

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

@socketio.on('update_image', namespace='/reseausimple')
def generer_images():
    er = ReseauSimple(xmod=True)
    data, label = clans1v1()
    for i in range(100):
        print('dsfgdsfgdsfgdsfgdsfgsdfgdsfgdsf')
        er.train(data, label, passages=1)
        image_name = er.visualisation(data, label, savemod=True, folder="static/svg", etape=i)
        image_path = url_for('static', filename=f'svg/{image_name}')
        print(image_path)
        socketio.emit('nouvelle_image', {'image_path': image_path}, namespace='/reseausimple')


if __name__ == '__main__':
    webbrowser.open('http://127.0.0.1:5000/')
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)