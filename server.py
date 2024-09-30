from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import base64
from face import recognize_faces, load_encodings
from spoofing import anti_spoofing
import time
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)

# Global variables for encodings
encodings_file = "/home/bahrombek/Desktop/face_spoofing1/encodings.csv"
known_encodings, known_names = load_encodings(encodings_file)

def reload_encodings():
    global known_encodings, known_names
    while True:
        time.sleep(60)  # Har bir daqiqada qayta yuklash
        print("Reloading encodings...")
        known_encodings, known_names = load_encodings(encodings_file)

# Encodings-larni qayta yuklash uchun alohida thread
encodings_thread = threading.Thread(target=reload_encodings, daemon=True)
encodings_thread.start()

def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_recognition')
def start_recognition():
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    anti_spoofing_active = False
    recognized_name = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if time.time() - start_time >= 10:
            break

        if time.time() - start_time >= 2:
            anti_spoofing_active = True

        if anti_spoofing_active:
            frame, is_real_face = anti_spoofing(frame)
            if is_real_face:
                frame, name = recognize_faces(frame, known_encodings, known_names)
                if name and name != "Unknown":
                    recognized_name = name
                    break  # Exit the loop once a face is recognized

        frame_encoded = encode_frame(frame)
        emit('update_frame', {'image': frame_encoded, 'name': recognized_name}, broadcast=True)
        socketio.sleep(0.1)

    cap.release()
    if recognized_name:
        emit('face_recognized', {'name': recognized_name}, broadcast=True)
    else:
        emit('recognition_timeout', broadcast=True)

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)