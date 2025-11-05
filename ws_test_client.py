import base64
import time
import socketio
from pathlib import Path

sio = socketio.Client()

received = {
    'count': 0,
    'done': False,
}

@sio.event(namespace='/ws/detect')
def connect():
    print('[client] connected to /ws/detect')

@sio.on('hello', namespace='/ws/detect')
def on_hello(data):
    print('[client] hello:', data)

@sio.on('result', namespace='/ws/detect')
def on_result(data):
    received['count'] += 1
    persons = data.get('persons') or []
    fps = data.get('fps')
    print(f"[client] result {received['count']}: persons={len(persons)} fps={fps}")
    # Stop after first result
    received['done'] = True
    sio.disconnect()

@sio.event(namespace='/ws/detect')
def disconnect():
    print('[client] disconnected')


def main():
    url = 'http://127.0.0.1:5000'
    img_path = Path('data_raw/cap_000000.jpg')
    if not img_path.exists():
        print(f'Image not found: {img_path.resolve()}')
        return
    img_bytes = img_path.read_bytes()
    img_b64 = base64.b64encode(img_bytes).decode('ascii')
    dataurl = 'data:image/jpeg;base64,' + img_b64
    print('[client] connecting...')
    sio.connect(url, namespaces=['/ws/detect'])
    print('[client] sending frame...')
    sio.emit('frame', {'frame': dataurl, 'camera_id': 'webcam'}, namespace='/ws/detect')
    # Wait a bit
    t0 = time.time()
    while not received['done'] and time.time() - t0 < 5:
        time.sleep(0.1)
    if not received['done']:
        print('[client] timeout waiting for result')
        try:
            sio.disconnect()
        except Exception:
            pass

if __name__ == '__main__':
    main()
