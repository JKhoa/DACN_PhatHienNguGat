#!/usr/bin/env python
"""Minimal Flask-SocketIO test with custom namespace"""
from flask import Flask
from flask_socketio import SocketIO, Namespace, emit
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
socketio = SocketIO(
    app,
    cors_allowed_origins='*',
    ping_interval=10,
    ping_timeout=30,
)

@socketio.on('connect', namespace='/test')
def test_connect():
    logging.info('[TestNamespace] Client connected')
    emit('hello', {'msg': 'Hello from TestNamespace'})

@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    logging.info('[TestNamespace] Client disconnected')

@socketio.on('test', namespace='/test')
def test_handler(data):
    logging.info(f'[TestNamespace] Received test: {data}')
    emit('test_response', {'msg': 'Test received'})

@app.route('/health')
def health():
    return {'ok': True}

if __name__ == '__main__':
    logging.info('Starting minimal Flask-SocketIO test server')
    socketio.run(app, host='127.0.0.1', port=5002, debug=False, allow_unsafe_werkzeug=True)
