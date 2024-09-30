from flask import Flask, render_template, request, jsonify
import os
from Alignment import run_alignment
from encoding import save_encodings

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
OUTPUT_FOLDER = '/home/bahrombek/Desktop/face_spoofing1/datasets'
ENCODINGS_FILE = '/home/bahrombek/Desktop/face_spoofing1/encodings.csv'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('alignment.html')

@app.route('/upload-and-align', methods=['POST'])
def upload_and_align():
    if 'images' not in request.files:
        return jsonify({"success": False, "message": "Fayl topilmadi"}), 400
    
    files = request.files.getlist('images')
    
    if not files or all(file.filename == '' for file in files):
        return jsonify({"success": False, "message": "Fayl nomi bo'sh"}), 400
    
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        if not os.path.exists(file_path):
            file.save(file_path)
        else:
            return jsonify({"success": False, "message": f"{file.filename} allaqachon mavjud."}), 400

    alignment_results = run_alignment(UPLOAD_FOLDER, OUTPUT_FOLDER)
    
    # Encoding qismi
    for result in alignment_results['details']:
        if result.startswith("Saved aligned face to"):
            image_path = result.split(" ")[-1]
            save_encodings(image_path, ENCODINGS_FILE)
    
    return jsonify(alignment_results)

if __name__ == '__main__':
    app.run(debug=True)