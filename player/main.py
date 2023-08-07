from flask import Flask, send_from_directory

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('C:/Users/Administrator/Desktop/deep_signer/player/webgl', 'index.html')

@app.route('/<path:path>')
def server_statio(path):
    return send_from_directory('C:/Users/Administrator/Desktop/deep_signer/player/webgl', path)

app.run()