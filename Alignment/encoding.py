import pandas as pd
import face_recognition
import os

def save_encodings(image_path, encodings_file):
    known_image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(known_image)
    
    if not face_encodings:
        print(f"Xato: {image_path} da yuz topilmadi.")
        return
    
    known_encoding = face_encodings[0]
    encoding_entry = {
        'name': os.path.splitext(os.path.basename(image_path))[0],
        'encoding': str(known_encoding.tolist())
    }

    if os.path.exists(encodings_file):
        df = pd.read_csv(encodings_file)
        known_names = df['name'].tolist()
        if encoding_entry['name'] not in known_names:
            new_df = pd.DataFrame([encoding_entry])
            new_df.to_csv(encodings_file, mode='a', header=False, index=False)
            print(f"{encoding_entry['name']} kodlari '{encodings_file}' fayliga qo'shildi.")
        else:
            print(f"{encoding_entry['name']} allaqachon mavjud.")
    else:
        new_df = pd.DataFrame([encoding_entry])
        new_df.to_csv(encodings_file, index=False)
        print(f"{encoding_entry['name']} kodlari '{encodings_file}' fayliga saqlandi.")