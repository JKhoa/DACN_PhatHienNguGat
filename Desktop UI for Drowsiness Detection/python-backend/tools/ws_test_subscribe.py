import sys
import time
import socketio

# Simple test client for /ws/camera namespace
# Usage: python ws_test_subscribe.py <camera_id>

URL = 'http://127.0.0.1:5000/ws/camera'

if __name__ == '__main__':
    cam_id = sys.argv[1] if len(sys.argv) > 1 else 'cam1'
    sio = socketio.Client(logger=True, engineio_logger=False, reconnection=True)

    @sio.event(namespace='/ws/camera')
    def connect():
        print('[client] connected, subscribing to', cam_id)
        sio.emit('subscribe', {'camera_id': cam_id}, namespace='/ws/camera')

    @sio.event(namespace='/ws/camera')
    def subscribed(data):
        print('[client] subscribed ack:', data)

    @sio.on('update', namespace='/ws/camera')
    def on_update(data):
        persons = data.get('persons', [])
        w = data.get('frame_width')
        h = data.get('frame_height')
        fps = data.get('fps')
        print(f"[update] cam={data.get('camera_id')} persons={len(persons)} size={w}x{h} fps={fps}")

    @sio.event(namespace='/ws/camera')
    def disconnect():
        print('[client] disconnected')

    sio.connect(URL, transports=['websocket'])
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        sio.disconnect()
