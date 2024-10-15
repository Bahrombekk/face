# Face Recognition and Anti-Spoofing Application

## Overview
This application uses Flask with Flask-SocketIO to create a real-time face recognition and anti-spoofing system. It captures video from a webcam, verifies if the face is real (anti-spoofing), and then performs face recognition using pre-stored face encodings.

## Features
- **Face Recognition:** Recognizes known faces from a CSV file containing face encodings.
- **Anti-Spoofing:** Detects if a face is real or fake using a pre-trained anti-spoofing model.
- **Live Reload of Encodings:** The application reloads face encodings every minute in a separate thread.
- **Web Interface:** Real-time video feed and recognition results are displayed in a browser.

## How It Works
1. The application uses a webcam to capture frames.
2. After 2 seconds, the anti-spoofing feature is activated.
3. If the face is real, the face recognition process checks if the face matches any known faces.
4. The recognized face name is displayed on the web interface, and if no face is recognized within 10 seconds, it times out.

## Project Structure
- **server.py:** The main application file that manages video capture, face recognition, and anti-spoofing.
- **face.py:** Contains the logic for loading face encodings and performing face recognition.
- **spoofing.py:** Implements the anti-spoofing check using a pre-trained model.
- **templates/index.html:** Frontend that shows the live video feed and recognition results.

## Running the Application
1. Install dependencies:
    ```bash
    pip install flask flask-socketio opencv-python face-recognition
    ```

2. Run the Flask application:
    ```bash
    python server.py
    ```

3. Open your browser and navigate to `http://localhost:8080` to see the live feed and recognition results.

## Configuration
- **Encodings File:** The file path to the CSV containing face encodings should be specified in `server.py`.
- **Anti-Spoofing Models:** Ensure the anti-spoofing models are in the correct directory (`resources/anti_spoof_models`).

## Notes
- The application is designed to work with a webcam (ID: 0). You can adjust this in `server.py`.
- Ensure your webcam and OpenCV work properly on your system for capturing frames.





After running the `alig.py` script, images of individuals whose faces need to be recognized can be added through the interface available at `http://127.0.0.1:5000/`.

![ ](https://github.com/Bahrombekk/face_spoofing1/raw/main/Alignment/uploads/Pasted%20image%201.png)

Once the information is added, the `server.py` script is executed, and you can access the live recognition system by navigating to `http://127.0.0.1:8080/`.

![](https://github.com/Bahrombekk/face_spoofing1/blob/main/Alignment/uploads/Pasted%20image.png)

The interface will display the real-time video feed for recognition, as shown in the images.

![](https://github.com/Bahrombekk/face_spoofing1/blob/main/Alignment/uploads/Pasted%20image%202.png)

Make sure all necessary adjustments are made for a seamless workflow between these stages.

![](https://github.com/Bahrombekk/face_spoofing1/blob/main/Alignment/uploads/Pasted%20image%203.png)

