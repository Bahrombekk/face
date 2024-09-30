import cv2
import face_recognition
import os
import numpy as np
import csv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_encodings(encodings_file):
    if not os.path.exists(encodings_file):
        logger.error(f"{encodings_file} file not found.")
        return None, None
    
    known_encodings = []
    known_names = []
    
    try:
        with open(encodings_file, 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                name = row[0]
                encoding = [float(x) for x in row[1].strip('[]').split(', ')]
                known_names.append(name)
                known_encodings.append(encoding)
        return known_encodings, known_names
    except Exception as e:
        logger.error(f"Error loading encodings: {e}")
        return None, None

def recognize_faces(frame, known_encodings, known_names, threshold=0.5):
    if frame is None or known_encodings is None or known_names is None:
        logger.error("Invalid input parameters")
        return frame, None
    
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            name = "Unknown"
            if len(known_encodings) > 0:
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index] and face_distances[best_match_index] < threshold:
                    name = known_names[best_match_index]
            
            #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            #cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 2)
        
        return frame, name # Return the frame and the recognized name
    except Exception as e:
        logger.error(f"Error in face recognition: {e}")
        return frame, None # Return the frame and None if no face was recognized