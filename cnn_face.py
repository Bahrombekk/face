import cv2
import face_recognition
import numpy as np
import pickle
import logging
from collections import Counter
from spoofing import anti_spoofing
import time
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_encodings(encodings_file):
    try:
        with open(encodings_file, 'rb') as f:
            known_encodings, known_names = pickle.load(f)
        return known_encodings, known_names
    except Exception as e:
        logger.error(f"Kodlarni yuklashda xatolik: {e}")
        return None, None

def recognize_faces(frame, known_encodings, known_names, tolerance=0.5, min_detections=5):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
        name = "Noma'lum"

        if True in matches:
            matched_indices = [i for i, match in enumerate(matches) if match]
            distances = face_recognition.face_distance(np.array(known_encodings)[matched_indices], face_encoding)
            best_match_index = np.argmin(distances)
            if distances[best_match_index] <= tolerance:
                name = known_names[matched_indices[best_match_index]]

        face_names.append(name)

    # Izchil tanib olishni ta'minlash uchun
    if hasattr(recognize_faces, 'previous_names'):
        recognize_faces.previous_names.append(face_names)
        if len(recognize_faces.previous_names) > min_detections:
            recognize_faces.previous_names.pop(0)
        
        consistent_names = []
        for i, name in enumerate(face_names):
            names_for_face = [names[i] for names in recognize_faces.previous_names if i < len(names)]
            most_common = Counter(names_for_face).most_common(1)
            if most_common and most_common[0][1] >= min_detections // 2:
                consistent_names.append(most_common[0][0])
            else:
                consistent_names.append("Noma'lum")
        face_names = consistent_names
    else:
        recognize_faces.previous_names = [face_names]

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

    return frame

def main():
    encodings_file = "/home/bahrombek/Desktop/face_spoofing/encoding/encodings1.pkl"
    known_encodings, known_names = load_encodings(encodings_file)
    
    if known_encodings is None or known_names is None:
        logger.error("Kodlarni yuklashda xatolik yuz berdi. Dastur to'xtatilmoqda.")
        return

    cap = cv2.VideoCapture(0)
    start_time = time.time()  # Record the start time
    anti_spoofing_active = False  # Flag to control anti-spoofing

    while True:
        ret, frame = cap.read()
        
        if not ret:
            logger.error("Kadrni olishda xatolik")
            break
        if time.time() - start_time >= 2:
            anti_spoofing_active = True

        if anti_spoofing_active:
            frame, is_real_face = anti_spoofing(frame)
            if is_real_face:
                frame = recognize_faces(frame, known_encodings, known_names)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()