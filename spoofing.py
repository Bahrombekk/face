import os
import cv2
import numpy as np
import time
import warnings
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')

# Yuzni aniqlash uchun Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Yuzni aniqlash va anti-spoofing tahlili
def anti_spoofing(frame):
    model_dir = "./resources/anti_spoof_models"  # Model kutubxonasi
    device_id = 0  # GPU ID

    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    
    # Rangli tasvirni kulrangga aylantirish
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Yuzlarni aniqlash
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    is_real_face = False  # Soxta yoki haqiqiy yuz aniqlanganini kuzatib borish uchun flag

    for (x, y, w, h) in faces:
        # Yuzni kesish
        face = frame[y:y+h, x:x+w]
        # Yuzni o'lchash
        face_resized = cv2.resize(face, (60, 80), interpolation=cv2.INTER_LINEAR)
        
        # Anti-spoofing tahlili
        image_bbox = (x, y, w, h)  # Yuzning koordinatalari
        prediction = np.zeros((1, 3))
        
        # Modelni ishlatish
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": face_resized,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            start = time.time()
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))
            # Test tezligini hisoblash (ixtiyoriy)
            test_speed = time.time() - start

        # Natijani chizish
        label = np.argmax(prediction)
        value = prediction[0][label] / 2
        
        # Threshold qiymatini o'zgartirish
        threshold = 0.1  # O'zgartirilgan threshold
        if value < threshold:
            label = 0  # Soxta yuz deb belgilash
        
        if label == 1:
            result_text = "Haqiqiy Yuz: {:.2f}".format(value)
            color = (255, 0, 0)  # Qizil
            is_real_face = True  # Haqiqiy yuz topildi
        else:
            result_text = "Soxta Yuz: {:.2f}".format(value)
            color = (0, 0, 255)  # Ko'k
        
        # Yuzga to'rtburchak va matn qo'shish
        #cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        #cv2.putText(frame, result_text, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5 * frame.shape[0] / 1024, color)
        is_real_face = True
    return frame, is_real_face  # Yangilangan frame va haqiqiylik flagi qaytariladi

